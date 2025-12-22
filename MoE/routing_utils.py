from typing import Dict, Any, List, Optional, Callable, Union, Iterable
import torch
import numpy as np
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel
from contextlib import contextmanager
from types import MethodType

def attach_router_hooks(layer_att, routing_store: Dict[int, List[torch.Tensor]]) -> int:
    """Attaches forward hooks to capture routing logits across all MoE layers.

    Iterates through all layers in the model and registers forward hooks on the router
    (gate) modules. The hooks capture routing logits during forward passes and store
    them in the provided routing_store dictionary.

    Args:
        layer_att: The model's layer attribute object containing MoE layers. For base
            models, this is the model itself. For PEFT models, this is model.base_model.model.
        routing_store: Dictionary to store captured routing tensors, mapping layer indices
            to lists of routing logit tensors.

    Returns:
        The total number of layers in the model.

    Note:
        This function assumes the router module is accessible via `layer.mlp.gate`.
        Only layers that have this attribute structure will have hooks attached.
    """
    num_layers = len(layer_att.model.layers)

    for i, layer in enumerate(layer_att.model.layers):
        # Match your current known router path
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            layer.mlp.gate.register_forward_hook(get_router_hook(i, routing_store))

    return num_layers


def get_router_hook(layer_idx: int, store: Dict[int, List[torch.Tensor]]) -> Callable:
    """Creates a forward hook to capture routing logits for a specific layer.

    Args:
        layer_idx: The index of the MoE layer to monitor.
        store: Dictionary to store captured routing tensors, mapping layer indices
            to lists of routing logit tensors.

    Returns:
        A hook function that can be registered with PyTorch's register_forward_hook.
        The hook extracts routing logits and stores them in the provided store dict.
    """

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            data = output[0].detach().cpu()
        else:
            data = output.detach().cpu()
        store.setdefault(layer_idx, []).append(data)
        return output

    return hook


def run_prefill(model, inputs, store: Dict[int, List[torch.Tensor]]) -> None:
    """Runs model prefill (prompt processing) and captures routing data.

    Clears the routing store dictionary and performs a forward pass on the model
    to process the input prompt. Router hooks capture routing logits during this phase.

    Args:
        model: The language model to run inference on.
        inputs: Tokenized input tensors (typically from tokenizer with return_tensors='pt').
        store: Dictionary to store captured routing tensors, mapping layer indices
            to lists of routing logit tensors.
    """
    store.clear()
    with torch.no_grad():
        _ = model(**inputs)


def run_generate(
    model, inputs, max_new_tokens: int, store: Dict[int, List[torch.Tensor]]
) -> None:
    """Runs full model generation (prefill + decode) and captures routing data.

    Clears the routing store dictionary and performs text generation including both
    the prefill phase (prompt processing) and decode phase (token generation). Router hooks
    capture routing logits during both phases.

    Args:
        model: The language model to run generation on.
        inputs: Tokenized input tensors (typically from tokenizer with return_tensors='pt').
        max_new_tokens: Maximum number of tokens to generate.
        store: Dictionary to store captured routing tensors, mapping layer indices
            to lists of routing logit tensors.
    """
    store.clear()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=max_new_tokens)


# -----------------------------
# Ablation hook
# -----------------------------

MaskSpec = Union[int, Iterable[int]]


@contextmanager
def ablate_experts(
    layer_att, expert_mask_by_layer: Dict[int, MaskSpec], *, neg: float = -1e9
):
    """Context manager for temporarily masking (ablating) experts in router logits.

    Registers forward hooks on specified layers to set masked expert logits to a large
    negative value, effectively preventing those experts from being selected by the router.
    The hooks are automatically removed when exiting the context.

    Args:
        layer_att: The model's layer attribute object containing MoE layers with gate modules.
        expert_mask_by_layer: Dictionary mapping layer indices to expert IDs to mask.
            Values can be either a single expert ID (int) or an iterable of expert IDs.
        neg: Large negative value to assign to masked expert logits. Defaults to -1e9.

    Yields:
        None. The context is active while forward hooks are registered.

    Example:
        >>> with ablate_experts(model, {0: 5, 1: [3, 7]}):
        >>>     output = model(input)  # Expert 5 in layer 0 and experts 3,7 in layer 1 are masked
    """
    handles = []

    def make_hook(layer_idx: int, mask_spec: MaskSpec):
        if isinstance(mask_spec, int):
            mask_list = [mask_spec]
        else:
            mask_list = list(mask_spec)

        def hook(module, inputs, output):
            if isinstance(output, tuple):
                logits = output[0]
                rest = output[1:]
            else:
                logits = output
                rest = None

            logits2 = logits.clone()
            for e in mask_list:
                logits2[..., e] = neg

            if rest is None:
                return logits2
            return (logits2, *rest)

        return hook

    for i, layer in enumerate(layer_att.model.layers):
        if hasattr(layer.mlp, "gate") and i in expert_mask_by_layer:
            handles.append(
                layer.mlp.gate.register_forward_hook(
                    make_hook(i, expert_mask_by_layer[i])
                )
            )

    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def scale_router_logits_via_weight(layer_att, alpha: float):
    """Context manager for scaling router logits by scaling the input to the router.

    Temporarily patches router forward methods to scale hidden states by alpha before
    computing logits. Since routing logits are computed as logits = W @ h, scaling the
    input by alpha is equivalent to scaling the logits: softmax(alpha * (W @ h)).

    This approach is more robust than directly scaling weights as it doesn't modify
    model parameters and works consistently across different router implementations.

    Args:
        layer_att: The model's layer attribute object containing MoE layers. For base
            models, this is the model itself. For PEFT models, this is model.base_model.model.
        alpha: Scaling factor applied to hidden states before routing. Higher values create
            sharper routing distributions (logits = W @ (alpha * h)), lower values create
            smoother distributions.

    Yields:
        The number of routers that were successfully patched.

    Raises:
        RuntimeError: If no router modules were found to patch.

    Note:
        The original forward methods are automatically restored when exiting the context.
        This function searches for router modules under layer.mlp.gate attributes.

    Example:
        >>> with scale_router_logits_via_weight(model, alpha=2.0) as n:
        >>>     print(f"Patched {n} routers")
        >>>     output = model(input)  # Routing is sharper due to scaled logits
    """
    patched = []
    n_patched = 0

    # Traverse all modules so we don't rely on attribute names like mlp.gate
    for layer in layer_att.modules():
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            mod = layer.mlp.gate
            # Save original weights
            orig_forward = mod.forward

            def wrapped_forward(self, hidden_states, *_args, __orig=orig_forward, __alpha=alpha, **_kwargs):
                # hidden_states: [..., hidden_dim]
                return __orig(hidden_states * __alpha)

            # bind method to instance
            mod.forward = MethodType(wrapped_forward, mod)
            patched.append((mod, orig_forward))
            n_patched += 1

    if n_patched == 0:
        raise RuntimeError(
            "scale_router_logits_via_weight: patched 0 routers. "
            "Your model may not contain Qwen2MoeTopKRouter modules under layer_att."
        )

    try:
        yield n_patched
    finally:
        for mod, orig_forward in patched:
            mod.forward = orig_forward


def summarize_routing_data(
    routing_data: Dict[int, List[torch.Tensor]],
    *,
    num_layers: int,
    n_experts: int = 60,
    compute_importance: bool = True,
) -> Dict[str, Any]:
    """Converts captured routing logits into additive statistics.

    Processes the routing logits collected by forward hooks and computes aggregated
    statistics that can be used for analysis or subtraction (e.g., decode = generate - prefill).

    Args:
        routing_data: Dictionary mapping layer indices to lists of routing logit tensors.
            Each tensor has shape [T, E] where T is the number of tokens and E is the
            number of experts.
        num_layers: Total number of MoE layers in the model.
        n_experts: Number of experts per layer. Defaults to 60.
        compute_importance: Whether to compute importance metrics (softmax probabilities,
            max probs, margins). Defaults to True.

    Returns:
        A dictionary containing:
            - counts_top1: [L, E] int64 array of top-1 expert selection counts
            - total_events: [L] int64 array of total routing decisions per layer
            - sum_probs: [L, E] float64 array of sum of softmax probs (or None)
            - sum_pmax: [L] float64 array of sum of max probabilities (or None)
            - sum_margin: [L] float64 array of sum of (p1 - p2) margins (or None)
    """
    counts_top1 = np.zeros((num_layers, n_experts), dtype=np.int64)
    total_events = np.zeros(num_layers, dtype=np.int64)

    sum_probs = (
        np.zeros((num_layers, n_experts), dtype=np.float64)
        if compute_importance
        else None
    )
    sum_pmax = np.zeros(num_layers, dtype=np.float64) if compute_importance else None
    sum_margin = np.zeros(num_layers, dtype=np.float64) if compute_importance else None

    for layer_idx, data_list in routing_data.items():
        if layer_idx < 0 or layer_idx >= num_layers:
            continue
        if not data_list:
            continue

        layer_logits = torch.cat(data_list, dim=0)
        if layer_logits.ndim != 2 or layer_logits.shape[-1] != n_experts:
            continue

        layer_logits = layer_logits.to(torch.float32)
        T = int(layer_logits.shape[0])
        if T <= 0:
            continue

        # top-1 counts
        top1 = layer_logits.argmax(dim=-1)
        counts = top1.bincount(minlength=n_experts).cpu().numpy().astype(np.int64)
        counts_top1[layer_idx] += counts
        total_events[layer_idx] += int(counts.sum())

        if compute_importance:
            probs = torch.softmax(layer_logits, dim=-1)  # [T, E]
            sum_probs[layer_idx] += probs.sum(dim=0).cpu().numpy().astype(np.float64)

            p_sorted, _ = probs.sort(dim=-1, descending=True)
            sum_pmax[layer_idx] += float(p_sorted[:, 0].sum().item())
            sum_margin[layer_idx] += float(
                (p_sorted[:, 0] - p_sorted[:, 1]).sum().item()
            )

    return {
        "counts_top1": counts_top1,
        "total_events": total_events,
        "sum_probs": sum_probs,
        "sum_pmax": sum_pmax,
        "sum_margin": sum_margin,
    }


def decode_from_gen_minus_pre(
    gen_sum: Dict[str, Any], pre_sum: Dict[str, Any]
) -> Dict[str, Any]:
    """Extracts decode-only statistics by subtracting prefill from generation statistics.

    Computes the decode phase statistics by subtracting prefill routing statistics from
    the full generation (prefill + decode) statistics. All values are clipped to ensure
    non-negative results.

    Args:
        gen_sum: Dictionary of generation statistics from summarize_routing_data, containing
            the routing statistics for the full generation process (prefill + decode).
        pre_sum: Dictionary of prefill statistics from summarize_routing_data, containing
            the routing statistics for only the prompt processing phase.

    Returns:
        A dictionary containing decode-only statistics with the same structure as the input:
            - counts_top1: [L, E] int64 array of decode-only expert selection counts
            - total_events: [L] int64 array of decode-only routing decisions
            - sum_probs: [L, E] float64 array of decode-only softmax probs (or None)
            - sum_pmax: [L] float64 array of decode-only max probabilities (or None)
            - sum_margin: [L] float64 array of decode-only margins (or None)
    """
    decode_counts = np.clip(gen_sum["counts_top1"] - pre_sum["counts_top1"], 0, None)
    decode_total = np.clip(gen_sum["total_events"] - pre_sum["total_events"], 0, None)

    decode_sum_probs = None
    decode_sum_pmax = None
    decode_sum_margin = None
    if gen_sum["sum_probs"] is not None and pre_sum["sum_probs"] is not None:
        decode_sum_probs = np.clip(
            gen_sum["sum_probs"] - pre_sum["sum_probs"], 0.0, None
        )
    if gen_sum["sum_pmax"] is not None and pre_sum["sum_pmax"] is not None:
        decode_sum_pmax = np.clip(gen_sum["sum_pmax"] - pre_sum["sum_pmax"], 0.0, None)
    if gen_sum["sum_margin"] is not None and pre_sum["sum_margin"] is not None:
        decode_sum_margin = np.clip(
            gen_sum["sum_margin"] - pre_sum["sum_margin"], 0.0, None
        )

    return {
        "counts_top1": decode_counts,
        "total_events": decode_total,
        "sum_probs": decode_sum_probs,
        "sum_pmax": decode_sum_pmax,
        "sum_margin": decode_sum_margin,
    }


def build_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    adapter_path: Optional[str] = None,
) -> tuple:
    """Loads a 4-bit quantized model and tokenizer, optionally with a PEFT adapter.

    Args:
        model_name: HuggingFace model identifier (e.g., "Qwen/Qwen1.5-MoE-A2.7B").
        device: PyTorch device to load the model on (e.g., torch.device("cuda")).
        adapter_path: Optional path to a PEFT/LoRA adapter checkpoint. If provided,
            the adapter will be loaded on top of the base model. Defaults to None.

    Returns:
        A tuple containing:
            - model: The callable model used for forward passes and generation.
            - layer_att: The underlying model object whose layers you attach hooks to.
                For base models, this is the same as model. For PEFT models, this is
                model.base_model.model.
            - tokenizer: HuggingFace tokenizer for the model.
            - suffix: String tag extracted from adapter_path for use in filenames,
                or empty string if no adapter is used.
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        trust_remote_code=True,  # recommended for Qwen MoE
    ).to(device)

    layer_att = model
    suffix = ""

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
        # note: this matches your viz.py pattern
        layer_att = model.base_model.model
        suffix = adapter_path.split("-")[-1]

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    return model, layer_att, tokenizer, suffix
