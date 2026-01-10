"""
Laplace Approximation for Neural Networks

Classic approach (MacKay, 1992) making a comeback with modern variants.

Core idea:
    1. Train network normally to get MAP estimate w*
    2. Approximate posterior as Gaussian centered at w*
    3. Covariance = inverse Hessian of loss at w*

    p(w|D) ≈ N(w*, H^{-1})
    where H = ∇²L(w*) is the Hessian of the loss

Why it works:
    - Taylor expansion of log posterior around MAP
    - Gaussian is exact for quadratic log posterior
    - Good approximation near the mode

The challenge: Computing/storing the Hessian
    - Full Hessian: O(P²) where P = millions of parameters
    - Inversion: O(P³)

Approximations:
    1. Diagonal: Only diagonal of Hessian (independent weights)
    2. KFAC: Kronecker-factored approximate curvature (block diagonal)
    3. Last-layer: Only Laplace on final layer (tractable, often sufficient)
    4. Subnetwork: Laplace on subset of weights

Last-layer Laplace:
    - Treat all but last layer as fixed feature extractor
    - Apply Laplace only to final linear layer
    - Much smaller Hessian: (features × classes)²
    - Often works surprisingly well!

Modern library: https://github.com/aleximmer/laplace

Exercises:
    1. Implement diagonal Laplace for small network
    2. Implement last-layer Laplace (most practical)
    3. Compare to MC Dropout and ensembles
    4. Explore effect of prior precision (regularization strength)
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))

import bayesian.mc_dropout as mc


class DiagonalLaplace:
    """Diagonal Laplace approximation.

    Assumes posterior is N(w*, diag(1/h)) where h is diagonal of Hessian.

    For classification with cross-entropy:
        Hessian diagonal ≈ sum over data of: (p * (1-p)) * x²
        where p = softmax output, x = input to that weight

    Math:
        1. Taylor expand log posterior around MAP: log p(w|D) ≈ const + ½(w-w*)ᵀH(w-w*)
        2. This is log of Gaussian, so p(w|D) ≈ N(w*, H⁻¹)
        3. Approximate H with empirical Fisher: F = Σᵢ gᵢgᵢᵀ (outer product of gradients)
        4. Diagonal approx: only keep diag(F), giving independent Gaussians per weight
        5. Posterior precision for weight i: τᵢ = τ₀ + Σⱼ(∂L/∂wᵢ)² where τ₀ = prior precision
    """

    def __init__(self, model: nn.Module, prior_precision: float = 1.0):
        """
        Args:
            model: Trained neural network (at MAP estimate)
            prior_precision: Prior precision (inverse variance), acts as regularization
        """
        self.model = model
        self.prior_precision = prior_precision
        self.posterior_precision = None  # To be computed
        self.map_weights = None  # Store MAP weights

    def fit(self, train_loader: DataLoader, device: torch.device):
        """Compute diagonal Hessian approximation using empirical Fisher.

        For each weight w_i:
            precision_i = prior_precision + Σ_j (∂log p(y_j|x_j,w) / ∂w_i)²

        This uses the empirical Fisher approximation:
            F_diag ≈ Σ_j diag(g_j ⊙ g_j) where g_j = gradient for sample j
        """
        self.model.to(device)
        self.model.eval()

        # Store MAP weights
        self.map_weights = {name: param.clone() for name, param in self.model.named_parameters()}

        # Initialize precision with prior (like adding λI to Hessian)
        self.posterior_precision = {
            name: torch.ones_like(param) * self.prior_precision
            for name, param in self.model.named_parameters()
        }

        # Accumulate squared gradients (diagonal of empirical Fisher)
        n_samples = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Process each sample individually to get per-sample gradients
            for i in range(inputs.size(0)):
                self.model.zero_grad()
                output = self.model(inputs[i : i + 1])

                # Log likelihood for this sample (negative cross-entropy)
                log_lik = -F.cross_entropy(output, targets[i : i + 1])
                log_lik.backward()

                # Add squared gradient to precision
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.posterior_precision[name] += param.grad.data ** 2

                n_samples += 1

        print(f"Fitted diagonal Laplace on {n_samples} samples")

    @torch.no_grad()
    def sample_weights(self) -> dict:
        """Sample weights from approximate posterior.

        w ~ N(w*, diag(1/posterior_precision))

        For each weight:
            w_i = w*_i + ε_i / √precision_i,  where ε_i ~ N(0,1)

        Returns:
            Dictionary mapping parameter names to sampled values
        """
        if self.posterior_precision is None:
            raise RuntimeError("Must call fit() before sampling")

        sampled = {}
        for name, map_weight in self.map_weights.items():
            precision = self.posterior_precision[name]
            std = 1.0 / torch.sqrt(precision)  # std = 1/√precision
            noise = torch.randn_like(map_weight)
            sampled[name] = map_weight + noise * std

        return sampled

    def _set_weights(self, weights: dict):
        """Set model weights from dictionary."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(weights[name])

    def _restore_map_weights(self):
        """Restore MAP weights."""
        self._set_weights(self.map_weights)

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100):
        """Predict with uncertainty using weight sampling.

        1. Sample weights from posterior
        2. Set model weights to sample
        3. Forward pass
        4. Repeat and aggregate

        Returns:
            mean_probs: Mean predicted probabilities
            std_probs: Std of predicted probabilities (uncertainty)
        """
        if self.posterior_precision is None:
            raise RuntimeError("Must call fit() before predicting")

        self.model.eval()
        all_probs = []

        for _ in range(n_samples):
            # Sample weights from posterior
            sampled_weights = self.sample_weights()
            self._set_weights(sampled_weights)

            # Forward pass
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)

        # Restore original MAP weights
        self._restore_map_weights()

        # Stack and compute statistics
        all_probs = torch.stack(all_probs, dim=0)  # (n_samples, batch, n_classes)
        mean_probs = all_probs.mean(dim=0)
        std_probs = all_probs.std(dim=0)

        return mean_probs, std_probs


class LastLayerLaplace:
    """Laplace approximation on last layer only.

    Much more tractable:
    - Fix all layers except the last as feature extractor
    - Apply full (not diagonal) Laplace to last layer
    - Hessian size: (n_features × n_classes)²

    Often works as well as full-network Laplace!

    Math for multiclass classification:
        - Last layer: f = Φw where Φ ∈ R^{N×D} is features, w ∈ R^{D×C} is weights
        - GGN approximation: H = Σᵢ Jᵢᵀ Λᵢ Jᵢ where Λᵢ = diag(pᵢ) - pᵢpᵢᵀ
        - Posterior: p(w|D) ≈ N(w*, H⁻¹)
    """

    def __init__(self, model: nn.Module, prior_precision: float = 1.0):
        """
        Args:
            model: Trained network. Last layer should be nn.Linear.
            prior_precision: Prior precision for last layer weights
        """
        self.model = model
        self.prior_precision = prior_precision

        # Find the last linear layer
        self.last_layer = self._find_last_layer()
        self.n_features = self.last_layer.in_features
        self.n_classes = self.last_layer.out_features

        # Parameters: weights (D×C) + bias (C) flattened
        self.n_params = self.n_features * self.n_classes
        if self.last_layer.bias is not None:
            self.n_params += self.n_classes

        self.posterior_mean = None  # Last layer weights (flattened)
        self.posterior_cov = None   # Covariance matrix
        self._scale_matrix = None   # Precomputed Σ^{1/2} for sampling
        self.device = None

    def _find_last_layer(self) -> nn.Linear:
        """Find the last linear layer in the model."""
        last_linear = None

        # Check common names first
        for name in ['fc', 'classifier', 'head', 'output', 'linear']:
            if hasattr(self.model, name):
                layer = getattr(self.model, name)
                if isinstance(layer, nn.Linear):
                    return layer
                # Handle Sequential
                if isinstance(layer, nn.Sequential):
                    for sublayer in reversed(list(layer.children())):
                        if isinstance(sublayer, nn.Linear):
                            return sublayer

        # Fallback: find last Linear in module tree
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module

        if last_linear is None:
            raise ValueError("Could not find a Linear layer in the model")

        return last_linear

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features (input to last layer) for given input."""
        features = None

        def hook(module, input, output):
            nonlocal features
            features = input[0].detach()

        handle = self.last_layer.register_forward_hook(hook)
        with torch.no_grad():
            self.model(x)
        handle.remove()

        return features

    def fit(self, train_loader: DataLoader, device: torch.device):
        """Compute posterior covariance for last layer.

        Uses the Generalized Gauss-Newton (GGN) approximation:
            H = Σᵢ Jᵢᵀ Λᵢ Jᵢ + τI

        where:
            - Jᵢ is the Jacobian of outputs w.r.t. last-layer params
            - Λᵢ = diag(pᵢ) - pᵢpᵢᵀ is the Hessian of softmax cross-entropy
            - τ is the prior precision
        """
        self.model.to(device)
        self.model.eval()
        self.device = device

        D, C = self.n_features, self.n_classes
        has_bias = self.last_layer.bias is not None

        # Store MAP estimate
        weight_flat = self.last_layer.weight.detach().view(-1).clone()  # (C*D,)
        if has_bias:
            bias = self.last_layer.bias.detach().clone()  # (C,)
            self.posterior_mean = torch.cat([weight_flat, bias])
        else:
            self.posterior_mean = weight_flat

        # Initialize GGN with prior precision
        H = self.prior_precision * torch.eye(self.n_params, device=device)

        # Accumulate GGN over data
        n_samples = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)

            # Get features and predictions
            features = self._get_features(inputs)  # (B, D)
            with torch.no_grad():
                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=-1)  # (B, C)

            # Accumulate GGN for each sample
            for i in range(features.size(0)):
                phi = features[i]  # (D,)
                p = probs[i]       # (C,)

                # Λ = diag(p) - ppᵀ (Hessian of cross-entropy w.r.t. logits)
                Lambda = torch.diag(p) - torch.outer(p, p)  # (C, C)

                # Build Jacobian of outputs w.r.t. params
                # For output f_c = Σ_d w_{cd} φ_d + b_c
                # df_c/dw_{cd} = φ_d, df_c/db_c = 1

                # Weights contribution: J_w has shape (C, C*D)
                # J_w[c, c*D:(c+1)*D] = phi
                # H_w contribution: J_wᵀ Λ J_w

                # Efficient computation using Kronecker structure:
                # J_w = I_C ⊗ φᵀ, so J_wᵀ Λ J_w = Λ ⊗ (φφᵀ)
                phi_outer = torch.outer(phi, phi)  # (D, D)

                # Kronecker product: Λ ⊗ φφᵀ gives (C*D, C*D) matrix
                H_w = torch.kron(Lambda, phi_outer)

                if has_bias:
                    # Bias contribution: J_b = I_C, so H_b = Λ
                    # Cross terms: J_wᵀ Λ J_b = Λ ⊗ φ (reshaped)
                    H_b = Lambda  # (C, C)

                    # Cross term: for each class c, d(f_c)/d(w_{c,:}) = phi, d(f_c)/d(b_c) = 1
                    # So the cross Jacobian contribution is Λ ⊗ φ
                    H_cross = torch.kron(Lambda, phi.unsqueeze(1))  # (C*D, C)

                    # Assemble full Hessian block
                    H_full = torch.zeros(self.n_params, self.n_params, device=device)
                    H_full[:C*D, :C*D] = H_w
                    H_full[C*D:, C*D:] = H_b
                    H_full[:C*D, C*D:] = H_cross
                    H_full[C*D:, :C*D] = H_cross.T

                    H += H_full
                else:
                    H += H_w

                n_samples += 1

        # Add jitter for numerical stability before inversion
        jitter = 1e-6 * torch.trace(H) / self.n_params
        H += jitter * torch.eye(self.n_params, device=device)

        # Invert to get posterior covariance
        self.posterior_cov = torch.linalg.inv(H)

        # Precompute scale matrix for sampling (using eigendecomposition for robustness)
        # Σ = V @ diag(λ) @ Vᵀ, so Σ^{1/2} = V @ diag(√λ) @ Vᵀ
        eigvals, eigvecs = torch.linalg.eigh(self.posterior_cov)
        # Clamp eigenvalues to be positive (numerical stability)
        eigvals = torch.clamp(eigvals, min=1e-10)
        self._scale_matrix = eigvecs @ torch.diag(torch.sqrt(eigvals))

        print(f"Fitted last-layer Laplace on {n_samples} samples")
        print(f"Posterior covariance shape: {self.posterior_cov.shape}")

    def sample_weights(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample last-layer weights from the posterior.

        Returns:
            weight: Sampled weight matrix (C, D)
            bias: Sampled bias vector (C,) or None
        """
        if self.posterior_cov is None:
            raise RuntimeError("Must call fit() before sampling")

        D, C = self.n_features, self.n_classes
        has_bias = self.last_layer.bias is not None

        # Sample from N(μ, Σ) using precomputed scale matrix
        # w = μ + Σ^{1/2} @ z where z ~ N(0, I)
        z = torch.randn(self.n_params, device=self.device)
        sample = self.posterior_mean + self._scale_matrix @ z

        # Reshape to weight matrix and bias
        weight = sample[:C*D].view(C, D)
        bias = sample[C*D:] if has_bias else None

        return weight, bias

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100,
                                  method: str = "sample"):
        """Predict with uncertainty.

        Two methods:
        1. "sample": Sample last-layer weights, forward pass each
        2. "probit": Closed-form approximation for classification
           (faster but approximate)

        Returns:
            mean_probs: Mean predicted probabilities
            std_probs: Std of predicted probabilities (uncertainty)
        """
        if self.posterior_cov is None:
            raise RuntimeError("Must call fit() before predicting")

        self.model.eval()
        x = x.to(self.device)

        # Extract features
        features = self._get_features(x)  # (B, D)

        if method == "sample":
            return self._predict_sampling(features, n_samples)
        elif method == "probit":
            return self._predict_probit(features)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _predict_sampling(self, features: torch.Tensor, n_samples: int):
        """Predict by sampling weights from posterior."""
        all_probs = []

        for _ in range(n_samples):
            weight, bias = self.sample_weights()

            # Manual forward through last layer with sampled weights
            logits = F.linear(features, weight, bias)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)

        all_probs = torch.stack(all_probs, dim=0)  # (n_samples, B, C)
        mean_probs = all_probs.mean(dim=0)
        std_probs = all_probs.std(dim=0)

        return mean_probs, std_probs

    def _predict_probit(self, features: torch.Tensor):
        """Predict using probit approximation (closed-form).

        For binary case: integrate σ(f) over N(f|μ, σ²)
        Using probit approx: E[σ(f)] ≈ σ(μ / √(1 + π/8 * σ²))

        For multiclass, we apply this to each logit difference.
        This is faster but less accurate than sampling.
        """
        B, D = features.shape
        C = self.n_classes
        has_bias = self.last_layer.bias is not None

        # Compute mean prediction
        weight_mean = self.posterior_mean[:C*D].view(C, D)
        bias_mean = self.posterior_mean[C*D:] if has_bias else None
        mean_logits = F.linear(features, weight_mean, bias_mean)  # (B, C)

        # Compute predictive variance for each logit
        # Var(f_c) = φᵀ Σ_wc φ + Σ_bc (for weight block c and bias c)
        # This requires extracting the right blocks from posterior_cov

        logit_var = torch.zeros(B, C, device=self.device)

        for c in range(C):
            # Weight block for class c: indices c*D to (c+1)*D
            w_start, w_end = c * D, (c + 1) * D

            # Σ_wc is the (D, D) block of covariance for weights of class c
            Sigma_wc = self.posterior_cov[w_start:w_end, w_start:w_end]

            # Var(wᵀφ) = φᵀ Σ φ
            # Vectorized over batch: (B, D) @ (D, D) @ (D, B) -> diagonal
            logit_var[:, c] = (features @ Sigma_wc * features).sum(dim=1)

            if has_bias:
                # Add bias variance
                b_idx = C * D + c
                logit_var[:, c] += self.posterior_cov[b_idx, b_idx]

                # Add weight-bias covariance: 2 * φᵀ Σ_wb
                Sigma_wb = self.posterior_cov[w_start:w_end, b_idx]
                logit_var[:, c] += 2 * (features @ Sigma_wb)

        # Probit approximation: scale logits by uncertainty
        # σ(f/√(1 + π/8 * σ²)) ≈ E[σ(f)] for f ~ N(μ, σ²)
        kappa = 1.0 / torch.sqrt(1.0 + (3.14159 / 8.0) * logit_var)
        scaled_logits = mean_logits * kappa

        mean_probs = F.softmax(scaled_logits, dim=-1)

        # Approximate std using sampling for a few samples
        # (probit doesn't give us std directly)
        std_probs = torch.sqrt(logit_var) * mean_probs * (1 - mean_probs)

        return mean_probs, std_probs


def train_map(
    model: nn.Module,
    train_loader: DataLoader,
    n_epochs: int,
    device: torch.device,
    weight_decay: float = 1e-4,
    lr: float = 1e-3,
    verbose: bool = True,
):
    """Train model to MAP estimate.

    Note: weight_decay corresponds to Gaussian prior with precision = weight_decay.
    This connects L2 regularization to Bayesian inference!

    MAP objective: argmax_w p(w|D) = argmax_w [log p(D|w) + log p(w)]
                 = argmin_w [NLL(w) + (weight_decay/2) ||w||²]

    Args:
        model: Neural network to train
        train_loader: Training data loader
        n_epochs: Number of training epochs
        device: Device to train on
        weight_decay: L2 regularization (= prior precision in Bayesian view)
        lr: Learning rate
        verbose: Print training progress

    Returns:
        model: Trained model (modified in-place)
        losses: List of average losses per epoch
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if verbose and (epoch + 1) % max(1, n_epochs // 10) == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

    model.eval()
    return model, losses


class SimpleMLP(nn.Module):
    """Simple MLP for MNIST classification."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, output_dim: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Named 'fc' for easy detection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc(x)


@torch.no_grad()
def evaluate_model(model, loader, device):
    """Evaluate MAP model accuracy."""
    model.eval()
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    return correct / total


@torch.no_grad()
def evaluate_with_laplace(laplace, loader, device, n_samples: int = 50, method: str = "sample"):
    """Evaluate Laplace model with uncertainty estimates.

    Returns:
        accuracy: Classification accuracy
        mean_confidence: Average max probability
        mean_entropy: Average predictive entropy
    """
    correct = 0
    total = 0
    confidence = 0.0
    entropy = 0.0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        probs, std = laplace.predict_with_uncertainty(inputs, n_samples=n_samples, method=method)

        max_vals, predicted = probs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        confidence += max_vals.sum().item()
        entropy += -(probs * torch.log(probs + 1e-8)).sum(dim=1).sum().item()

    return correct / total, confidence / total, entropy / total


@torch.no_grad()
def collect_laplace_predictions(laplace, loader, device, n_samples: int = 50, method: str = "sample"):
    """Collect predictions for visualization."""
    all_probs = []
    all_stds = []
    all_targets = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        probs, std = laplace.predict_with_uncertainty(inputs, n_samples=n_samples, method=method)
        all_probs.append(probs.cpu())
        all_stds.append(std.cpu())
        all_targets.append(targets)

    return torch.cat(all_probs), torch.cat(all_stds), torch.cat(all_targets)


def compute_ece(probs, targets, n_bins: int = 10):
    """Compute Expected Calibration Error."""
    max_probs, preds = probs.max(dim=1)
    accuracies = (preds == targets).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probs)

    for i in range(n_bins):
        mask = (max_probs >= bin_boundaries[i]) & (max_probs < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean().item()
            bin_conf = max_probs[mask].mean().item()
            ece += (mask.sum().item() / total) * abs(bin_acc - bin_conf)

    return ece


def main():
    """Demo Laplace approximation."""
    import argparse

    parser = argparse.ArgumentParser(description="Laplace Approximation Demo")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (prior precision)")
    parser.add_argument("--prior-precision", type=float, default=1.0, help="Laplace prior precision")
    parser.add_argument("--n-samples", type=int, default=50, help="MC samples for prediction")
    parser.add_argument("--method", type=str, default="sample", choices=["sample", "probit"])
    parser.add_argument("--ood", type=str, default="fashionmnist", choices=["fashionmnist", "noise"])
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, test_loader = mc.get_dataloaders(batch_size=args.batch_size)
    ood_loader = mc.get_ood_loader(args.ood, batch_size=args.batch_size)

    # Create and train model to MAP estimate
    print("\n=== Training MAP Estimate ===")
    model = SimpleMLP(input_dim=784, hidden_dim=256, output_dim=10)
    model, losses = train_map(
        model,
        train_loader,
        n_epochs=args.epochs,
        device=device,
        weight_decay=args.weight_decay,
        lr=args.lr,
    )

    # Evaluate MAP model
    map_acc = evaluate_model(model, test_loader, device)
    print(f"MAP Test Accuracy: {map_acc:.4f}")

    # Apply Last-Layer Laplace
    print("\n=== Fitting Last-Layer Laplace ===")
    laplace = LastLayerLaplace(model, prior_precision=args.prior_precision)
    laplace.fit(train_loader, device)

    # Evaluate on in-distribution data
    print(f"\n=== In-Distribution (MNIST Test) [method={args.method}] ===")
    acc, conf, ent = evaluate_with_laplace(
        laplace, test_loader, device, n_samples=args.n_samples, method=args.method
    )
    print(f"Accuracy: {acc:.4f}")
    print(f"Mean Confidence: {conf:.4f}")
    print(f"Mean Entropy: {ent:.4f}")

    # Evaluate on OOD data
    print(f"\n=== Out-of-Distribution ({args.ood}) ===")
    ood_acc, ood_conf, ood_ent = evaluate_with_laplace(
        laplace, ood_loader, device, n_samples=args.n_samples, method=args.method
    )
    print(f"Accuracy: {ood_acc:.4f} (expected low)")
    print(f"Mean Confidence: {ood_conf:.4f} (should be lower)")
    print(f"Mean Entropy: {ood_ent:.4f} (should be higher)")

    # Summary
    print("\n=== Uncertainty Comparison ===")
    print(f"Confidence drop (ID - OOD): {conf - ood_conf:.4f} (positive = good)")
    print(f"Entropy increase (OOD - ID): {ood_ent - ent:.4f} (positive = good)")

    # Collect predictions for calibration analysis
    print("\nComputing calibration metrics...")
    id_probs, id_stds, id_targets = collect_laplace_predictions(
        laplace, test_loader, device, n_samples=args.n_samples, method=args.method
    )
    ece = compute_ece(id_probs, id_targets)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")

    # Compare with MAP predictions (no uncertainty)
    print("\n=== MAP vs Laplace Comparison ===")
    model.eval()
    map_probs_list = []
    map_targets_list = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs.to(device))
            probs = F.softmax(outputs, dim=-1)
            map_probs_list.append(probs.cpu())
            map_targets_list.append(targets)
    map_probs = torch.cat(map_probs_list)
    map_targets = torch.cat(map_targets_list)
    map_ece = compute_ece(map_probs, map_targets)
    map_entropy = -(map_probs * torch.log(map_probs + 1e-8)).sum(dim=1).mean().item()

    print(f"MAP ECE: {map_ece:.4f} | Laplace ECE: {ece:.4f}")
    print(f"MAP Mean Entropy: {map_entropy:.4f} | Laplace Mean Entropy: {ent:.4f}")

    plot_dir = "./plots/laplace"
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Training loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("MAP Training Loss")
    ax.grid(True, alpha=0.3)
    fig.savefig(f"{plot_dir}/training_loss.png", dpi=150)
    print(f"Saved: {plot_dir}/training_loss.png")

    # 2. Uncertainty histogram (ID vs OOD)
    ood_probs, _, _ = collect_laplace_predictions(
        laplace, ood_loader, device, n_samples=args.n_samples, method=args.method
    )
    fig = mc.plot_uncertainty_histogram(id_probs, ood_probs, args.ood)
    fig.suptitle("Last-Layer Laplace: ID vs OOD Uncertainty", fontsize=12)
    fig.savefig(f"{plot_dir}/uncertainty_histogram.png", dpi=150)
    print(f"Saved: {plot_dir}/uncertainty_histogram.png")

    # 3. Reliability diagram
    fig = mc.plot_reliability_diagram(id_probs, id_targets)
    fig.suptitle("Last-Layer Laplace: Calibration", fontsize=12)
    fig.savefig(f"{plot_dir}/reliability_diagram.png", dpi=150)
    print(f"Saved: {plot_dir}/reliability_diagram.png")

    plt.show()
    print("\nDone!")


if __name__ == "__main__":
    main()
