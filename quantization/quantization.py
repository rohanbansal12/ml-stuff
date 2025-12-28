"""
Quantization Implementations - Learning Edition

This module contains function stubs for you to implement various quantization
techniques. Each function has detailed docstrings explaining the math and
algorithm, plus assertions to verify your implementation.

Work through these in order:
1. Basic utilities (scale/zero-point computation)
2. Quantize/dequantize operations
3. RTN (round-to-nearest) - simplest full method
4. Absmax - slight improvement
5. GPTQ - calibration-based, much better for 4-bit
6. AWQ - activation-aware, state-of-the-art

Run this file directly to test your implementations:
    python quantization.py
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable
from dataclasses import dataclass
from copy import deepcopy
from tqdm import tqdm


# Type alias for layer filter functions
LayerFilter = Callable[[str], bool]


@dataclass
class QuantConfig:
    """Configuration for quantization."""

    bits: int = 8  # Quantization bits (8, 4, 2)
    symmetric: bool = True  # Symmetric vs asymmetric quantization
    per_channel: bool = False  # Per-channel vs per-tensor scales
    group_size: Optional[int] = None  # Group quantization (e.g., 128 for GPTQ)
    
    # Layer filtering: functions that take layer name and return bool
    # Layer is quantized if: include(name) and not exclude(name)
    include: Optional[LayerFilter] = None  # If None, include all
    exclude: Optional[LayerFilter] = None  # If None, exclude none
    
    def should_quantize(self, layer_name: str, module: nn.Module) -> bool:
        """
        Check if a layer should be quantized based on type and include/exclude filters.
        
        Args:
            layer_name: Name of the layer (e.g., "model.layers.0.mlp.gate_proj")
            module: The module instance
        
        Returns:
            True if layer should be quantized
        
        Examples:
            # Only quantize attention layers
            config = QuantConfig(include=lambda n: 'attn' in n)
            
            # Quantize everything except lm_head and embeddings
            config = QuantConfig(exclude=lambda n: 'lm_head' in n or 'embed' in n)
            
            # Quantize only MLP layers, but not the final one
            config = QuantConfig(
                include=lambda n: 'mlp' in n,
                exclude=lambda n: 'layers.31' in n
            )
        """
        # Only quantize Linear layers
        if not isinstance(module, nn.Linear):
            return False
        # Check include filter (None means include all)
        if self.include is not None and not self.include(layer_name):
            return False
        # Check exclude filter (None means exclude none)
        if self.exclude is not None and self.exclude(layer_name):
            return False
        return True


# =============================================================================
# PART 1: Basic Quantization Math
# =============================================================================
#
# Quantization maps continuous values to discrete integers.
#
# For b-bit quantization:
#   - Symmetric: integers in [-2^(b-1), 2^(b-1) - 1], e.g., [-128, 127] for 8-bit
#   - Asymmetric: integers in [0, 2^b - 1], e.g., [0, 255] for 8-bit
#
# The key parameters are:
#   - scale (s): maps between float and int ranges
#   - zero_point (z): offset for asymmetric quantization
#
# Symmetric quantization (simpler, what most weight quantization uses):
#   q = round(x / s)           # quantize
#   x_hat = q * s              # dequantize
#
#   where s = max(|x|) / qmax
#
# Asymmetric quantization (better for activations with non-zero mean):
#   q = round(x / s) + z       # quantize
#   x_hat = (q - z) * s        # dequantize
#
#   where s = (xmax - xmin) / (qmax - qmin)
#         z = qmin - round(xmin / s)
# =============================================================================


def compute_qrange(bits: int, symmetric: bool) -> tuple[int, int]:
    """
    Compute the integer range [qmin, qmax] for quantization.

    Args:
        bits: Number of bits (e.g., 8, 4)
        symmetric: If True, use signed range centered at 0
                   If False, use unsigned range starting at 0

    Returns:
        (qmin, qmax) tuple
    """
    if symmetric:
        return (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)

    return (0, 2**bits - 1)


def compute_scale_symmetric(
    tensor: Tensor,
    bits: int,
    per_channel: bool = False,
    channel_dim: int = 0,
) -> Tensor:
    """
    Compute scale for symmetric quantization.

    For symmetric quantization, we map [-amax, amax] -> [qmin, qmax]
    where amax = max(|tensor|).

    Scale s = amax / qmax

    Args:
        tensor: Weights to quantize, typically [out_features, in_features]
        bits: Quantization bits
        per_channel: If True, compute separate scale per output channel
        channel_dim: Which dimension is the channel dimension

    Returns:
        scale: Scalar tensor or [n_channels, 1, ...] shaped tensor
    """
    _, qmax = compute_qrange(bits, symmetric=True)

    if per_channel:
        reduce_dims = [d for d in range(tensor.ndim) if d != channel_dim]
        amax = tensor.abs().amax(dim=reduce_dims, keepdim=True)
    else:
        amax = tensor.abs().amax()

    amax = torch.clamp(amax, min=1e-8)
    return amax / qmax


def compute_scale_zeropoint_asymmetric(
    tensor: Tensor,
    bits: int,
    per_channel: bool = False,
    channel_dim: int = 0,
) -> tuple[Tensor, Tensor]:
    """
    Compute scale and zero-point for asymmetric quantization.

    For asymmetric quantization, we map [xmin, xmax] -> [qmin, qmax].

    scale = (xmax - xmin) / (qmax - qmin)
    zero_point = qmin - round(xmin / scale)

    The zero_point shifts the range so that 0.0 maps to an integer.
    This is important for activations that are mostly positive (like ReLU outputs).

    Args:
        tensor: Values to quantize
        bits: Quantization bits
        per_channel: If True, compute separate scale/zp per channel
        channel_dim: Which dimension is the channel dimension

    Returns:
        (scale, zero_point) tuple
    """
    qmin, qmax = compute_qrange(bits, symmetric=False)
    if per_channel:
        reduce_dims = [d for d in range(tensor.ndim) if d != channel_dim]
        xmin = tensor.amin(dim=reduce_dims, keepdim=True)
        xmax = tensor.amax(dim=reduce_dims, keepdim=True)
    else:
        xmin = tensor.amin()
        xmax = tensor.amax()

    scale = torch.clamp((xmax - xmin) / (qmax - qmin), min=1e-8)
    zero_point = torch.clamp(qmin - torch.round(xmin / scale), min=qmin, max=qmax)

    return (scale, zero_point)


def quantize_tensor(
    tensor: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    bits: int,
    symmetric: bool,
) -> Tensor:
    """
    Quantize a floating-point tensor to integers.

    Symmetric:   q = clamp(round(x / s), qmin, qmax)
    Asymmetric:  q = clamp(round(x / s) + z, qmin, qmax)

    Args:
        tensor: Float tensor to quantize
        scale: Scale factor(s)
        zero_point: Zero point(s) - ignored for symmetric
        bits: Number of bits
        symmetric: Whether using symmetric quantization

    Returns:
        Integer tensor (stored as float for convenience, but values are integers)

    Note: We return float dtype because PyTorch operations are easier,
    but the values will all be integers in [qmin, qmax].
    """
    qmin, qmax = compute_qrange(bits, symmetric=symmetric)
    q = torch.round(tensor / scale)
    if not symmetric:
        q = torch.round(tensor / scale) + zero_point
    q = torch.clamp(q, min=qmin, max=qmax)
    return q


def dequantize_tensor(
    q: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    symmetric: bool,
) -> Tensor:
    """
    Dequantize integers back to floating-point.

    Symmetric:   x_hat = q * s
    Asymmetric:  x_hat = (q - z) * s

    Args:
        q: Quantized integer tensor
        scale: Scale factor(s) used in quantization
        zero_point: Zero point(s) - ignored for symmetric
        symmetric: Whether using symmetric quantization

    Returns:
        Dequantized float tensor (approximately equal to original)
    """
    if symmetric:
        x_hat = q * scale
    else:
        x_hat = (q - zero_point) * scale

    return x_hat


# =============================================================================
# PART 2: Round-to-Nearest (RTN) Quantization
# =============================================================================
#
# RTN is the simplest quantization method:
# 1. Compute scale (and zero_point for asymmetric)
# 2. Quantize weights
# 3. Immediately dequantize back to float
#
# The model runs in float, but weights have reduced precision.
# This is called "fake quantization" or "simulated quantization".
#
# Why dequantize immediately?
# - PyTorch operations expect float tensors
# - True integer inference needs special kernels (e.g., cuBLAS INT8)
# - Simulated quantization lets us measure quality impact easily
# =============================================================================


def quantize_rtn(
    tensor: Tensor,
    config: QuantConfig,
) -> tuple[Tensor, dict]:
    """
    Apply round-to-nearest quantization to a tensor.

    This is "fake quantization": quantize then immediately dequantize.
    The result is a float tensor with reduced effective precision.

    Args:
        tensor: Tensor to quantize (typically Linear.weight)
        config: Quantization configuration

    Returns:
        (dequantized_tensor, state_dict) where state_dict contains:
            - 'scale': the scale factor(s)
            - 'zero_point': the zero point(s)
            - 'config': the config used
    """
    if config.symmetric:
        scale = compute_scale_symmetric(
            tensor, bits=config.bits, per_channel=config.per_channel
        )
        zero_point = torch.zeros_like(scale)
    else:
        scale, zero_point = compute_scale_zeropoint_asymmetric(
            tensor, bits=config.bits, per_channel=config.per_channel
        )

    q = quantize_tensor(
        tensor,
        scale=scale,
        zero_point=zero_point,
        bits=config.bits,
        symmetric=config.symmetric,
    )
    dq = dequantize_tensor(
        q, scale=scale, zero_point=zero_point, symmetric=config.symmetric
    )

    return (dq, {"scale": scale, "zero_point": zero_point, "config": config})


def quantize_linear_rtn(
    linear: nn.Linear,
    config: QuantConfig,
) -> nn.Linear:
    """
    Quantize a Linear layer's weights using RTN.

    Args:
        linear: Linear layer to quantize
        config: Quantization config

    Returns:
        New Linear layer with quantized weights (bias unchanged)

    Note: We create a new layer rather than modifying in-place to
    preserve the original for comparison.
    """
    layer = deepcopy(linear)
    dq_weight, _ = quantize_rtn(tensor=linear.weight, config=config)
    layer.weight.data.copy_(dq_weight)
    return layer


def quantize_model_rtn(
    model: nn.Module,
    config: QuantConfig,
) -> nn.Module:
    """
    Apply RTN quantization to all Linear layers in a model.

    Args:
        model: Model to quantize
        config: Quantization config

    Returns:
        New model with quantized weights
        
    Note:
        Uses config.should_quantize(layer_name, module) to filter which layers
        get quantized. See QuantConfig for include/exclude examples.
    """
    new_model = deepcopy(model)

    with torch.no_grad():
        for name, module in new_model.named_modules():
            if not config.should_quantize(name, module):
                continue
            quantized_layer = quantize_linear_rtn(module, config=config)
            module.weight.data.copy_(quantized_layer.weight)

    return new_model


# =============================================================================
# PART 3: Absmax Quantization
# =============================================================================
#
# Absmax is just symmetric RTN with per-channel scales.
# Per-channel quantization uses a separate scale for each output channel,
# which significantly improves quality for weight matrices.
#
# For a weight matrix W of shape [out_features, in_features]:
# - Per-tensor: one scale for entire matrix
# - Per-channel: one scale per row (output channel)
#
# Per-channel is better because different output neurons may have
# very different weight magnitudes.
# =============================================================================


def quantize_absmax(
    tensor: Tensor,
    config: QuantConfig,
) -> tuple[Tensor, dict]:
    """
    Absmax quantization - symmetric per-channel quantization.

    This is the same as RTN with symmetric=True, per_channel=True.
    Separating it out because "absmax" is a common term in the literature.

    Args:
        tensor: Tensor to quantize
        config: Config (bits used, per_channel forced True, symmetric forced True)

    Returns:
        (dequantized_tensor, state_dict)
    """
    config.symmetric = True
    config.per_channel = True
    return quantize_rtn(tensor, config=config)


def quantize_model_absmax(
    model: nn.Module,
    config: QuantConfig,
) -> nn.Module:
    """Apply absmax quantization to all Linear layers."""
    config.symmetric = True
    config.per_channel = True
    return quantize_model_rtn(model, config=config)


# =============================================================================
# PART 4: GPTQ - Optimal Brain Quantization
# =============================================================================
#
# RTN quantizes each weight independently. But weights interact!
# Changing one weight affects the optimal values of others.
#
# GPTQ insight: Quantize weights one at a time, and adjust the remaining
# weights to compensate for the quantization error.
#
# For a linear layer y = Wx:
# - We want quantized W_q such that W_q @ x ≈ W @ x
# - The "Hessian" H = X^T @ X tells us how input features interact
# - When we quantize column j, we update columns k>j to minimize error
#
# Algorithm (simplified):
# 1. Collect calibration activations X from running real data
# 2. Compute H = X^T @ X (shape: [in_features, in_features])
# 3. For each column j (in order of increasing H[j,j]):
#    a. Quantize W[:, j]
#    b. Compute error: delta = W[:, j] - W_q[:, j]
#    c. Update remaining columns: W[:, k>j] -= (delta @ H[j, k>j]) / H[j,j]
#
# The update step is derived from Newton's method / optimal brain surgeon.
# It minimizes the increase in loss when we quantize column j.
# =============================================================================


def collect_calibration_activations(
    model: nn.Module,
    calibration_data: list[Tensor],
    layer_name: str,
) -> Tensor:
    """
    Collect input activations to a specific layer during forward passes.

    Args:
        model: The model
        calibration_data: List of input tensors to run through model
        layer_name: Name of layer to collect activations for (e.g., "model.layers.0.mlp.gate_proj")

    Returns:
        X: Collected activations, shape [n_samples * seq_len, in_features]
    """
    layer = dict(model.named_modules())[layer_name]

    activations = []

    def hook(module, input, output):
        x = input[0].detach()
        if x.dim() == 3:
            x = x.reshape(-1, x.size(-1))
        activations.append(x)

    handle = layer.register_forward_hook(hook)

    with torch.no_grad():
        for entry in calibration_data:
            _ = model(entry)

    handle.remove()

    return torch.cat(activations, dim=0)


def compute_hessian(X: Tensor) -> Tensor:
    """
    Compute the Hessian approximation for GPTQ.

    H = X^T @ X  (shape: [in_features, in_features])

    This measures how input features co-vary. H[i,j] large means
    features i and j often activate together, so quantizing one
    affects the other more.

    H[j,j] (the diagonal) measures the "importance" of feature j.
    Features with larger H[j,j] should be quantized more carefully.

    Args:
        X: Activations of shape [n_samples, in_features]

    Returns:
        H: Hessian of shape [in_features, in_features]

    Note: Add small diagonal for numerical stability: H += damping * I
          Use damping = 0.01 * mean(diag(H)) as a good default
    """
    # Work in float32 for numerical stability
    X = X.float()
    
    # Check for NaN/Inf in input
    if torch.isnan(X).any() or torch.isinf(X).any():
        print("  Warning: NaN/Inf in activations, cleaning...")
        X = torch.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    H = X.T @ X
    
    return H


def quantize_gptq_layer(
    weight: Tensor,
    hessian: Tensor,
    config: QuantConfig,
    block_size: int = 64,
) -> tuple[Tensor, dict]:
    """
    Memory-efficient GPTQ using Cholesky decomposition.
    
    This version computes H_inv blocks on-demand by solving triangular systems,
    avoiding storage of the full H^{-1} matrix.
    
    Memory usage:
    - L (Cholesky factor): n² × 4 bytes
    - Per block: block_size × n × 4 bytes for triangular solve
    - No full H_inv storage
    
    For n=8960, block_size=128:
    - L: 320 MB
    - Per block workspace: ~4.5 MB
    - Total: ~325 MB (vs 640 MB+ with full H_inv)

    Args:
        weight: Weight matrix [out_features, in_features]
        hessian: Hessian matrix [in_features, in_features]
        config: Quantization config  
        block_size: Number of columns to process together

    Returns:
        (dequantized_weight, state_dict)
    """
    W = weight.clone().float()
    n_out, n_in = W.shape
    device = W.device
    dtype = torch.float32
    
    Q = torch.zeros_like(W)
    zero_point = torch.zeros(1, device=device, dtype=dtype)
    
    H = hessian.detach().float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    diag_mean = H.diag().mean()
    if diag_mean <= 0 or torch.isnan(diag_mean):
        diag_mean = 1.0
    damping = 0.01 * diag_mean
    diag = torch.arange(H.size(0), device=H.device)
    H[diag, diag] += damping

    perm = torch.argsort(H.diag())
    inv_perm = torch.argsort(perm)  # To restore original order at the end
    W = W[:, perm]
    H = H[perm][:, perm]
    
    # Cholesky: H = L @ L^T
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    H_inv = H

    total_error = 0.0
    
    # Process columns in blocks
    for block_start in range(0, n_in, block_size):
        block_end = min(block_start + block_size, n_in)
        block_len = block_end - block_start

        H_inv_block = H_inv[block_start:block_end, block_start:block_end]
        H_inv_cross = H_inv[block_start:block_end, block_end:]

        W_block = W[:, block_start:block_end]
        Err_block = torch.zeros_like(W_block)

        for j in range(block_len):
            col_idx = block_start + j

            w = W_block[:, j].clone()  # Clone to avoid aliasing issues

            scale = compute_scale_symmetric(w, bits=config.bits, per_channel=False)
            q = quantize_tensor(w, scale, zero_point, config.bits, config.symmetric)
            w_q = dequantize_tensor(q, scale, zero_point, config.symmetric)

            Q[:, col_idx] = w_q

            d = H_inv_block[j, j].item()
            raw_error = w - w_q
            total_error += raw_error.pow(2).sum().item()
            err_scaled = torch.clamp(raw_error / d, -1e3, 1e3)
            
            Err_block[:, j] = err_scaled
            
            # Update remaining columns in this block
            if j + 1 < block_len:
                W_block[:, (j + 1):].addr_(err_scaled, H_inv_block[j, (j + 1):], alpha=-1.0)
        
        if block_end < n_in:
            W[:, block_end:] -= torch.matmul(Err_block, H_inv_cross)
    
    Q = Q[:, inv_perm]
    return Q.to(weight.dtype), {"method": "gptq_memeff", "block_size": block_size, "total_error": total_error}


def collect_all_layer_activations(
    model: nn.Module,
    calibration_data: list[Tensor],
    layer_names: list[str],
    store_on_cpu: bool = True,
) -> dict[str, Tensor]:
    """
    Collect activations for multiple layers in a single forward pass.
    
    This is much faster than calling collect_calibration_activations
    separately for each layer, as we only run the model once.
    
    Args:
        model: The model
        calibration_data: List of input tensors
        layer_names: Names of layers to collect activations for
        store_on_cpu: If True, store activations on CPU to avoid GPU OOM.
                     They'll be moved back to GPU when needed.
    
    Returns:
        Dict mapping layer_name -> activations tensor [n_samples * seq_len, in_features]
    """
    # Create a dict to store activations for each layer
    layer_activations: dict[str, list[Tensor]] = {name: [] for name in layer_names}
    
    # Get all layers by name
    name_to_layer = dict(model.named_modules())
    
    # Create hooks for all layers
    handles = []
    
    def make_hook(layer_name):
        def hook(module, input, output):
            x = input[0].detach()
            if x.dim() == 3:
                x = x.reshape(-1, x.size(-1))
            # Move to CPU immediately to avoid GPU OOM
            if store_on_cpu:
                x = x.cpu()
            layer_activations[layer_name].append(x)
        return hook
    
    for name in layer_names:
        layer = name_to_layer[name]
        handle = layer.register_forward_hook(make_hook(name))
        handles.append(handle)
    
    # Run all calibration data through the model
    with torch.no_grad():
        for entry in calibration_data:
            _ = model(entry)
    
    # Remove all hooks
    for handle in handles:
        handle.remove()
    
    # Concatenate activations for each layer
    result = {}
    for name in layer_names:
        result[name] = torch.cat(layer_activations[name], dim=0)
        # Free the list of tensors
        layer_activations[name] = None
    
    return result


def quantize_model_gptq(
    model: nn.Module,
    calibration_data: list[Tensor],
    config: QuantConfig,
    sequential: bool = True,
) -> nn.Module:
    """
    Apply GPTQ quantization to all Linear layers.

    Args:
        model: Model to quantize
        calibration_data: List of input tensors for calibration (e.g., 128 samples)
        config: Quantization config
        sequential: If True, re-collect activations after each layer (more accurate but slower).
                   If False, collect all activations once upfront (faster but slightly less accurate).

    Returns:
        Quantized model

    Note: Uses config.should_quantize(layer_name, module) to filter layers.
    
    Speed comparison (for a model with N layers, M calibration samples):
        - sequential=True:  O(N * M) forward passes
        - sequential=False: O(M) forward passes (N times faster!)
    
    The accuracy difference is usually negligible because:
    1. Quantization error at each layer is small
    2. Errors tend to average out across layers
    """
    new_model = deepcopy(model)
    
    # Get all Linear layers that should be quantized
    linear_layers = [
        (name, module) 
        for name, module in new_model.named_modules() 
        if config.should_quantize(name, module)
    ]
    
    layer_names = [name for name, _ in linear_layers]
    print(f"GPTQ: Quantizing {len(linear_layers)} layers...")
    
    if sequential:
        # Original sequential approach - more accurate but slower
        for layer_name, layer in tqdm(linear_layers, desc="GPTQ (sequential)"):
            activations = collect_calibration_activations(
                new_model, 
                calibration_data=calibration_data, 
                layer_name=layer_name
            )
            
            hessian = compute_hessian(activations)
            
            dq_weight, state = quantize_gptq_layer(
                layer.weight, 
                hessian=hessian, 
                config=config
            )
            
            with torch.no_grad():
                layer.weight.data.copy_(dq_weight)
            
            del activations, hessian
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        # Fast approach - collect all activations in one pass, store on CPU
        print("  Collecting activations for all layers in one pass...")
        all_activations = collect_all_layer_activations(
            new_model, 
            calibration_data, 
            layer_names,
            store_on_cpu=True
        )
        print(f"  Collected activations for {len(all_activations)} layers (stored on CPU)")
        
        device = next(new_model.parameters()).device
        nan_layers = []
        
        # Quantize each layer using cached activations
        for layer_name, layer in tqdm(linear_layers, desc="GPTQ"):
            # Move activations to GPU for this layer only
            activations = all_activations[layer_name].to(device)
            
            # Free CPU copy immediately
            del all_activations[layer_name]
            
            hessian = compute_hessian(activations)
            
            # Free activations after computing Hessian
            del activations
            
            # Check Hessian for issues
            if torch.isnan(hessian).any() or torch.isinf(hessian).any():
                print(f"  Warning: Bad Hessian for {layer_name}, using RTN")
                nan_layers.append(layer_name)
                dq_weight, _ = quantize_rtn(layer.weight, config)
                del hessian
            else:
                dq_weight, state = quantize_gptq_layer(
                    layer.weight, 
                    hessian=hessian, 
                    config=config
                )
                del hessian
            
            # Check output for NaN
            if torch.isnan(dq_weight).any() or torch.isinf(dq_weight).any():
                print(f"  Warning: NaN in quantized weight for {layer_name}, keeping original")
                nan_layers.append(layer_name)
            else:
                with torch.no_grad():
                    layer.weight.data.copy_(dq_weight)
            
            del dq_weight
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if nan_layers:
            print(f"  Warning: {len(nan_layers)} layers had NaN issues")

    return new_model



# =============================================================================
# PART 5: AWQ - Activation-Aware Weight Quantization
# =============================================================================
#
# AWQ insight: Not all weights are equally important. Weights that
# process large activations matter more than weights that process
# small activations.
#
# Key idea: Protect important weights by scaling them up before quantization.
#
# If weight channel j sees large activations, we want to quantize W[:, j]
# more precisely. We can do this by:
# 1. Scale up: W'[:, j] = W[:, j] * s[j]  (larger values = finer quantization)
# 2. Quantize W'
# 3. At runtime: y = W'_q @ (x / s)  (compensate by scaling input down)
#
# The scaling can be absorbed into the previous layer, so no runtime cost!
#
# Finding optimal scales:
# - s[j] = activation_magnitude[j] ^ alpha
# - Search over alpha in [0, 1] to minimize quantization error
# - alpha=0 means no scaling (vanilla RTN)
# - alpha=1 means scale proportional to activation magnitude
# =============================================================================


class AWQLinear(nn.Module):
    """
    Linear layer with AWQ-style input scaling for quantization.
    
    AWQ quantizes weights after scaling by per-channel factors. At inference,
    the input must be divided by these scales to compensate:
    
        y = W_q @ (x / s) + b
    
    where W_q = Quantize(W * s)
    
    In a full AWQ implementation, the scale division would be fused into the
    previous layer. This class handles it explicitly for correctness.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Quantized weight (already scaled and quantized)
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        
        # AWQ scales - stored as buffer (not a parameter, but saved with model)
        self.register_buffer('awq_scales', torch.ones(in_features, **factory_kwargs))
        
        # Reset parameters (will be overwritten by from_linear)
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        awq_scales: Tensor,
        quantized_weight: Tensor,
    ) -> "AWQLinear":
        """
        Create an AWQLinear from an existing Linear layer.
        
        Args:
            linear: Original Linear layer
            awq_scales: Per-channel scales [in_features]
            quantized_weight: Quantized weight W_q = Quantize(W * s)
        
        Returns:
            AWQLinear layer ready for inference
        """
        has_bias = linear.bias is not None
        device = linear.weight.device
        dtype = linear.weight.dtype
        
        awq_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=has_bias,
            device=device,
            dtype=dtype,
        )
        
        # Copy quantized weight
        with torch.no_grad():
            awq_linear.weight.copy_(quantized_weight)
            if has_bias:
                awq_linear.bias.copy_(linear.bias)
            awq_linear.awq_scales.copy_(awq_scales.to(device=device, dtype=dtype))
        
        return awq_linear
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with input scale compensation.
        
        y = W_q @ (x / s) + b
        
        Args:
            x: Input tensor [..., in_features]
        
        Returns:
            Output tensor [..., out_features]
        """
        # Scale input down to compensate for weight scaling
        x_scaled = x / self.awq_scales
        
        # Standard linear operation
        return nn.functional.linear(x_scaled, self.weight, self.bias)
    
    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bias={self.bias is not None}, awq=True'
        )


def compute_activation_magnitudes(
    model: nn.Module,
    calibration_data: list[Tensor],
    layer_name: str,
) -> Tensor:
    """
    Compute per-channel activation magnitudes for AWQ.

    Args:
        model: The model
        calibration_data: Calibration inputs
        layer_name: Layer to analyze

    Returns:
        magnitudes: Per-channel activation magnitudes [in_features]

    The magnitude for channel i can be computed as:
        mean(|X[:, i]|)  - average absolute activation
    or:
        max(|X[:, i]|)   - maximum absolute activation

    AWQ paper uses mean. Channels with larger magnitudes are more important.
    """
    activations = collect_calibration_activations(
        model, calibration_data=calibration_data, layer_name=layer_name
    )
    return activations.abs().mean(dim=0)


def find_optimal_scales(
    weight: Tensor,
    activation_magnitudes: Tensor,
    config: QuantConfig,
    n_grid: int = 20,
) -> Tensor:
    """
    Find optimal per-channel scaling factors for AWQ via grid search.

    We want to find s* = argmin_s || Q(W * diag(s)) / diag(s) - W ||

    AWQ parameterizes s as: s[j] = activation_magnitudes[j] ^ alpha
    and searches over alpha in [0, 1].

    Args:
        weight: Weight matrix [out_features, in_features]
        activation_magnitudes: Per-channel magnitudes [in_features]
        config: Quantization config
        n_grid: Number of grid points for alpha search

    Returns:
        optimal_scales: Per-channel scales [in_features]
    """
    device = weight.device
    dtype = weight.dtype
    _, in_features = weight.shape
    
    best_error = float('inf')
    best_scales = torch.ones(in_features, device=device, dtype=dtype)
    
    # Ensure activation magnitudes are on same device
    activation_magnitudes = activation_magnitudes.to(device=device, dtype=dtype)
    
    # Normalize activation magnitudes to avoid numerical issues
    # (very large or very small magnitudes can cause problems when raised to power)
    act_mag_normalized = activation_magnitudes / (activation_magnitudes.mean() + 1e-8)
    
    for alpha in torch.linspace(0, 1, n_grid, device=device):
        # Compute scales: s[j] = act_mag[j] ^ alpha
        s = act_mag_normalized ** alpha
        s = s.clamp(min=1e-8)  # Avoid division by zero
        
        # Scale weights up
        W_scaled = weight * s  # Broadcasting: [out, in] * [in] -> [out, in]
        
        # Quantize (using a copy of config to avoid modifying original)
        W_q, _ = quantize_rtn(W_scaled, QuantConfig(
            bits=config.bits,
            symmetric=config.symmetric,
            per_channel=config.per_channel,
        ))
        
        # Unscale to get reconstruction
        W_reconstructed = W_q / s
        
        # Compute reconstruction error
        error = ((W_reconstructed - weight) ** 2).sum().item()
        
        if error < best_error:
            best_error = error
            best_scales = s.clone()
    
    return best_scales


def quantize_awq_layer(
    weight: Tensor,
    activation_magnitudes: Tensor,
    config: QuantConfig,
) -> tuple[Tensor, Tensor, dict]:
    """
    Quantize a weight matrix using AWQ.

    Args:
        weight: Weight matrix [out_features, in_features]
        activation_magnitudes: Per-channel magnitudes [in_features]
        config: Quantization config

    Returns:
        (quantized_weight, scales, state_dict)

        The returned weight is: Q(W * s) where s are the optimal scales.
        To use this weight correctly, you need to scale inputs: y = W_q @ (x / s)

        In practice, the /s can be absorbed into the previous layer's weights.
    """
    # Find optimal scales via grid search
    scales = find_optimal_scales(
        weight, 
        activation_magnitudes=activation_magnitudes, 
        config=config
    )
    
    # Scale weights up before quantization
    W_scaled = weight * scales
    
    # Quantize the scaled weights
    dq_weight, quant_state = quantize_rtn(W_scaled, config=config)
    
    state_dict = {
        "method": "awq",
        "scales_min": scales.min().item(),
        "scales_max": scales.max().item(),
        **quant_state,
    }
    
    return dq_weight, scales, state_dict


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """Replace a module in the model by its name (handles nested modules)."""
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def quantize_model_awq(
    model: nn.Module,
    calibration_data: list[Tensor],
    config: QuantConfig,
    sequential: bool = False,
) -> nn.Module:
    """
    Apply AWQ quantization to all Linear layers.
    
    This replaces nn.Linear layers with AWQLinear layers that handle
    the input scaling compensation properly.

    Args:
        model: Model to quantize
        calibration_data: Calibration inputs
        config: Quantization config
        sequential: If True, collect activations separately for each layer (slower).
                   If False, collect all activations in one pass (faster, default).

    Returns:
        Quantized model with AWQLinear layers
        
    Note: Uses config.should_quantize(layer_name, module) to filter layers.
    
    Unlike GPTQ, AWQ doesn't modify weights in a way that affects subsequent
    layer activations (the scaling is compensated in the forward pass), so
    the fast mode is equally accurate as sequential mode.
    """
    new_model = deepcopy(model)
    
    # Get all Linear layers that should be quantized
    # We need to collect them first because we'll be modifying the model
    linear_layers = [
        (name, module) 
        for name, module in new_model.named_modules() 
        if config.should_quantize(name, module)
    ]
    
    layer_names = [name for name, _ in linear_layers]
    print(f"AWQ: Quantizing {len(linear_layers)} layers...")
    
    if sequential:
        # Original sequential approach
        for layer_name, layer in tqdm(linear_layers, desc="AWQ (sequential)"):
            activation_magnitudes = compute_activation_magnitudes(
                new_model, 
                calibration_data=calibration_data, 
                layer_name=layer_name
            )
            
            quantized_weight, scales, state = quantize_awq_layer(
                layer.weight,
                activation_magnitudes=activation_magnitudes,
                config=config
            )
            
            awq_layer = AWQLinear.from_linear(
                linear=layer,
                awq_scales=scales,
                quantized_weight=quantized_weight,
            )
            
            _set_module_by_name(new_model, layer_name, awq_layer)
            
            del activation_magnitudes
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        # Fast approach - collect all activations in one pass, store on CPU
        print("  Collecting activations for all layers in one pass...")
        all_activations = collect_all_layer_activations(
            new_model, 
            calibration_data, 
            layer_names,
            store_on_cpu=True
        )
        print(f"  Collected activations for {len(all_activations)} layers (stored on CPU)")
        
        device = next(new_model.parameters()).device
        
        # Compute magnitudes and free raw activations
        print("  Computing activation magnitudes...")
        all_magnitudes = {}
        for name in tqdm(layer_names, desc="Computing magnitudes"):
            all_magnitudes[name] = all_activations[name].abs().mean(dim=0)
            del all_activations[name]
        del all_activations
        
        # Quantize each layer
        for layer_name, layer in tqdm(linear_layers, desc="AWQ"):
            activation_magnitudes = all_magnitudes[layer_name].to(device)
            del all_magnitudes[layer_name]
            
            quantized_weight, scales, state = quantize_awq_layer(
                layer.weight,
                activation_magnitudes=activation_magnitudes,
                config=config
            )
            
            awq_layer = AWQLinear.from_linear(
                linear=layer,
                awq_scales=scales,
                quantized_weight=quantized_weight,
            )
            
            _set_module_by_name(new_model, layer_name, awq_layer)
            
            del activation_magnitudes, quantized_weight, scales
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return new_model


# =============================================================================
# Utility Functions (implemented for you)
# =============================================================================


def count_model_bits(
    model: nn.Module,
    config: QuantConfig,
) -> dict:
    """
    Estimate model size under different precisions.
    
    Args:
        model: Model to analyze
        config: Quantization config (uses bits and should_quantize filter)
    
    Returns:
        Dict with:
            - total_params: Total parameter count
            - quantized_params: Parameters that would be quantized
            - fp16_mb: Size in MB at FP16
            - quantized_mb: Size in MB after quantization
            - compression_ratio: FP16 size / quantized size
    """
    total_params = 0
    quantized_params = 0
    unquantized_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            n_weight = module.weight.numel()
            n_bias = module.bias.numel() if module.bias is not None else 0
            
            total_params += n_weight + n_bias
            
            if config.should_quantize(name, module):
                # Weight gets quantized, bias stays fp16
                quantized_params += n_weight
                unquantized_params += n_bias
            else:
                unquantized_params += n_weight + n_bias
    
    # Also count non-Linear parameters (embeddings, norms, etc.)
    all_params = sum(p.numel() for p in model.parameters())
    other_params = all_params - total_params
    unquantized_params += other_params
    
    # Calculate sizes
    fp16_bits = all_params * 16
    quantized_bits = quantized_params * config.bits + unquantized_params * 16
    
    return {
        "total_params": all_params,
        "quantized_params": quantized_params,
        "unquantized_params": unquantized_params,
        "fp16_mb": fp16_bits / 8 / 1e6,
        "quantized_mb": quantized_bits / 8 / 1e6,
        "compression_ratio": fp16_bits / quantized_bits if quantized_bits > 0 else 1.0,
    }


def compute_quantization_error(
    original: Tensor,
    quantized: Tensor,
) -> dict:
    """
    Compute various error metrics between original and quantized tensors.

    Returns dict with:
        - mse: Mean squared error
        - mae: Mean absolute error
        - max_error: Maximum absolute error
        - snr_db: Signal-to-noise ratio in dB (higher is better)
    """
    diff = original - quantized
    mse = (diff**2).mean().item()
    mae = diff.abs().mean().item()
    max_err = diff.abs().max().item()

    signal_power = (original**2).mean()
    noise_power = (diff**2).mean() + 1e-10
    snr = 10 * torch.log10(signal_power / noise_power)

    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_err,
        "snr_db": snr.item(),
    }


def count_unique_values(tensor: Tensor) -> int:
    """Count unique values in a tensor (sanity check for quantization)."""
    return len(torch.unique(tensor))


# =============================================================================
# Tests - Run these to verify your implementations
# =============================================================================


def test_qrange():
    """Test compute_qrange."""
    print("Testing compute_qrange...")

    # 8-bit symmetric
    qmin, qmax = compute_qrange(8, symmetric=True)
    assert qmin == -128, f"Expected -128, got {qmin}"
    assert qmax == 127, f"Expected 127, got {qmax}"

    # 8-bit asymmetric
    qmin, qmax = compute_qrange(8, symmetric=False)
    assert qmin == 0, f"Expected 0, got {qmin}"
    assert qmax == 255, f"Expected 255, got {qmax}"

    # 4-bit symmetric
    qmin, qmax = compute_qrange(4, symmetric=True)
    assert qmin == -8, f"Expected -8, got {qmin}"
    assert qmax == 7, f"Expected 7, got {qmax}"

    # 4-bit asymmetric
    qmin, qmax = compute_qrange(4, symmetric=False)
    assert qmin == 0, f"Expected 0, got {qmin}"
    assert qmax == 15, f"Expected 15, got {qmax}"

    print("  ✓ compute_qrange passed!")


def test_scale_symmetric():
    """Test compute_scale_symmetric."""
    print("Testing compute_scale_symmetric...")

    # Simple case: per-tensor
    x = torch.tensor([[1.0, -2.0], [0.5, 0.5]])
    s = compute_scale_symmetric(x, bits=8, per_channel=False)
    expected = 2.0 / 127  # amax=2.0, qmax=127
    assert torch.allclose(s, torch.tensor(expected), atol=1e-6), (
        f"Expected {expected}, got {s}"
    )

    # Per-channel
    s = compute_scale_symmetric(x, bits=8, per_channel=True, channel_dim=0)
    assert s.numel() == 2, f"Expected 2 scales, got {s.numel()}"
    # Channel 0: amax=2.0, Channel 1: amax=0.5
    expected_0 = 2.0 / 127
    expected_1 = 0.5 / 127
    assert torch.allclose(s.flatten()[0], torch.tensor(expected_0), atol=1e-6), (
        f"Channel 0: expected {expected_0}, got {s.flatten()[0]}"
    )
    assert torch.allclose(s.flatten()[1], torch.tensor(expected_1), atol=1e-6), (
        f"Channel 1: expected {expected_1}, got {s.flatten()[1]}"
    )

    print("  ✓ compute_scale_symmetric passed!")


def test_quantize_dequantize():
    """Test quantize_tensor and dequantize_tensor are approximate inverses."""
    print("Testing quantize/dequantize...")

    x = torch.randn(64, 128)

    # Symmetric 8-bit
    s = compute_scale_symmetric(x, bits=8)
    zp = torch.zeros_like(s)

    q = quantize_tensor(x, s, zp, bits=8, symmetric=True)
    x_hat = dequantize_tensor(q, s, zp, symmetric=True)

    # Check q contains integers in valid range
    assert torch.all(q == q.round()), "Quantized values should be integers"
    assert torch.all(q >= -128) and torch.all(q <= 127), "Values out of range"

    # Check reconstruction error is reasonable for 8-bit
    error = compute_quantization_error(x, x_hat)
    assert error["snr_db"] > 30, f"SNR too low for 8-bit: {error['snr_db']:.1f} dB"

    print(f"  8-bit symmetric SNR: {error['snr_db']:.1f} dB")
    print("  ✓ quantize/dequantize passed!")


def test_rtn():
    """Test RTN quantization end-to-end."""
    print("Testing RTN quantization...")

    x = torch.randn(256, 512)

    # 8-bit per-tensor
    config = QuantConfig(bits=8, symmetric=True, per_channel=False)
    x_q, state = quantize_rtn(x, config)

    assert x_q.shape == x.shape, "Shape mismatch"
    error = compute_quantization_error(x, x_q)
    print(f"  8-bit per-tensor SNR: {error['snr_db']:.1f} dB")
    assert error["snr_db"] > 30, f"8-bit SNR too low: {error['snr_db']:.1f} dB"

    # 8-bit per-channel (should be same or better)
    config = QuantConfig(bits=8, symmetric=True, per_channel=True)
    x_q_pc, state = quantize_rtn(x, config)

    error_pc = compute_quantization_error(x, x_q_pc)
    print(f"  8-bit per-channel SNR: {error_pc['snr_db']:.1f} dB")
    assert error_pc["snr_db"] >= error["snr_db"] - 0.1, (
        "Per-channel should be >= per-tensor"
    )

    # 4-bit per-channel (more degradation expected)
    config = QuantConfig(bits=4, symmetric=True, per_channel=True)
    x_q_4bit, state = quantize_rtn(x, config)

    error_4bit = compute_quantization_error(x, x_q_4bit)
    print(f"  4-bit per-channel SNR: {error_4bit['snr_db']:.1f} dB")

    print("  ✓ RTN passed!")


def test_model_quantization():
    """Test quantizing a simple model."""
    print("Testing model quantization...")

    # Simple MLP
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    )

    # Get original output
    x = torch.randn(8, 64)
    with torch.no_grad():
        y_orig = model(x)

    # Quantize to 8-bit
    config = QuantConfig(bits=8, symmetric=True, per_channel=True)
    model_q = quantize_model_rtn(model, config)

    # Check output similarity
    with torch.no_grad():
        y_quant = model_q(x)

    error = compute_quantization_error(y_orig, y_quant)
    print(f"  Output SNR after 8-bit quantization: {error['snr_db']:.1f} dB")

    # Verify weights were actually modified
    w_orig = model[0].weight.data
    w_quant = model_q[0].weight.data
    assert not torch.allclose(w_orig, w_quant), "Weights should be modified"

    # Verify original model was NOT modified
    with torch.no_grad():
        y_check = model(x)
    assert torch.allclose(y_orig, y_check), "Original model should be unchanged!"

    print("  ✓ Model quantization passed!")


def test_layer_filtering():
    """Test include/exclude layer filtering."""
    print("Testing layer filtering...")
    
    # Create a model with named layers
    class NamedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(64, 128)
            self.mlp1 = nn.Linear(128, 128)
            self.mlp2 = nn.Linear(128, 128)
            self.lm_head = nn.Linear(128, 64)
        
        def forward(self, x):
            x = torch.relu(self.embed(x))
            x = torch.relu(self.mlp1(x))
            x = torch.relu(self.mlp2(x))
            return self.lm_head(x)
    
    model = NamedMLP()
    x = torch.randn(8, 64)
    
    # Test 1: Exclude embed and lm_head (common pattern)
    config = QuantConfig(
        bits=8, 
        symmetric=True, 
        per_channel=True,
        exclude=lambda n: 'embed' in n or 'lm_head' in n
    )
    model_q = quantize_model_rtn(model, config)
    
    # embed and lm_head should be unchanged
    assert torch.allclose(model.embed.weight, model_q.embed.weight), "embed should not be quantized"
    assert torch.allclose(model.lm_head.weight, model_q.lm_head.weight), "lm_head should not be quantized"
    # mlp layers should be different
    assert not torch.allclose(model.mlp1.weight, model_q.mlp1.weight), "mlp1 should be quantized"
    assert not torch.allclose(model.mlp2.weight, model_q.mlp2.weight), "mlp2 should be quantized"
    print("  ✓ Exclude filter works")
    
    # Test 2: Only include mlp layers
    config = QuantConfig(
        bits=8,
        symmetric=True,
        per_channel=True,
        include=lambda n: 'mlp' in n
    )
    model_q = quantize_model_rtn(model, config)
    
    assert torch.allclose(model.embed.weight, model_q.embed.weight), "embed should not be quantized"
    assert torch.allclose(model.lm_head.weight, model_q.lm_head.weight), "lm_head should not be quantized"
    assert not torch.allclose(model.mlp1.weight, model_q.mlp1.weight), "mlp1 should be quantized"
    print("  ✓ Include filter works")
    
    # Test 3: Include mlp but exclude mlp2
    config = QuantConfig(
        bits=8,
        symmetric=True,
        per_channel=True,
        include=lambda n: 'mlp' in n,
        exclude=lambda n: 'mlp2' in n
    )
    model_q = quantize_model_rtn(model, config)
    
    assert torch.allclose(model.mlp2.weight, model_q.mlp2.weight), "mlp2 should not be quantized"
    assert not torch.allclose(model.mlp1.weight, model_q.mlp1.weight), "mlp1 should be quantized"
    print("  ✓ Include + exclude combination works")
    
    print("  ✓ Layer filtering passed!")


def run_all_tests():
    """Run all tests in order."""
    print("=" * 60)
    print("Running Quantization Implementation Tests")
    print("=" * 60)
    print()

    tests = [
        ("Part 1a", test_qrange),
        ("Part 1b", test_scale_symmetric),
        ("Part 1c", test_quantize_dequantize),
        ("Part 2", test_rtn),
        ("Part 2+", test_model_quantization),
        ("Filtering", test_layer_filtering),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test in tests:
        try:
            test()
            passed += 1
            print()
        except NotImplementedError as e:
            print(f"  ⏭ {name}: Not implemented yet - {e}")
            print()
            skipped += 1
        except AssertionError as e:
            print(f"  ✗ {name}: Assertion failed - {e}")
            print()
            failed += 1
        except Exception as e:
            print(f"  ✗ {name}: Unexpected error - {type(e).__name__}: {e}")
            print()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} not implemented")
    print("=" * 60)

    if passed == len(tests):
        print("\n🎉 All basic tests passed!")
        print("Next steps:")
        print("  1. Implement GPTQ (Part 4) - requires calibration data")
        print("  2. Implement AWQ (Part 5) - builds on GPTQ concepts")
        print("  3. Run compare_quant.py to see real model impact")
    elif skipped > 0 and failed == 0:
        print(f"\n📝 {passed} tests passed. Keep implementing!")
        print("Work through the functions in order - each builds on the previous.")
    else:
        print(f"\n🔧 {failed} tests failed. Debug those before moving on.")


if __name__ == "__main__":
    run_all_tests()