"""
Test suite for custom optimizer implementations.

Compares custom implementations against PyTorch's built-in optimizers
across various hyperparameter configurations.
"""

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from optimizers import Adam, AdamW, Muon, RMSProp, SGDMomentum


class ToyMLP(nn.Module):
    """Small MLP for testing optimizers."""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, output_dim: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


@dataclass
class TestConfig:
    """Configuration for a single optimizer test."""

    name: str
    custom_cls: type
    reference_cls: type
    kwargs: dict[str, Any]
    steps: int = 50
    rtol: float = 1e-4
    atol: float = 1e-6


def create_toy_data(
    batch_size: int = 16,
    input_dim: int = 32,
    output_dim: int = 10,
    num_batches: int = 50,
    seed: int = 42,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create deterministic toy data."""
    torch.manual_seed(seed)
    data = []
    for _ in range(num_batches):
        x = torch.randn(batch_size, input_dim)
        y = torch.randint(0, output_dim, (batch_size,))
        data.append((x, y))
    return data


def run_optimization(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: list[tuple[torch.Tensor, torch.Tensor]],
    steps: int,
) -> dict[str, torch.Tensor]:
    """Run optimization steps and return final parameters."""
    criterion = nn.CrossEntropyLoss()

    for step in range(steps):
        x, y = data[step % len(data)]
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    return {name: param.clone() for name, param in model.named_parameters()}


def compare_parameters(
    params1: dict[str, torch.Tensor],
    params2: dict[str, torch.Tensor],
    rtol: float,
    atol: float,
) -> tuple[bool, dict[str, float]]:
    """Compare two sets of parameters, return success and max differences."""
    max_diffs = {}
    all_close = True

    for name in params1:
        p1, p2 = params1[name], params2[name]
        max_diff = (p1 - p2).abs().max().item()
        max_diffs[name] = max_diff

        if not torch.allclose(p1, p2, rtol=rtol, atol=atol):
            all_close = False

    return all_close, max_diffs


def run_test(config: TestConfig, data: list, seed: int = 123) -> bool:
    """Run a single optimizer comparison test."""
    # Create two identical models
    torch.manual_seed(seed)
    model_custom = ToyMLP()
    model_ref = copy.deepcopy(model_custom)

    # Create optimizers
    opt_custom = config.custom_cls(model_custom.parameters(), **config.kwargs)
    opt_ref = config.reference_cls(model_ref.parameters(), **config.kwargs)

    # Run optimization
    params_custom = run_optimization(model_custom, opt_custom, data, config.steps)
    params_ref = run_optimization(model_ref, opt_ref, data, config.steps)

    # Compare
    success, max_diffs = compare_parameters(params_custom, params_ref, config.rtol, config.atol)

    # Report
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"  {status}: {config.name}")

    if not success:
        worst_layer = max(max_diffs, key=max_diffs.get)
        print(f"         Max diff: {max_diffs[worst_layer]:.2e} in {worst_layer}")

    return success


def get_sgd_configs() -> list[TestConfig]:
    """Test configurations for SGD with Momentum."""
    return [
        TestConfig(
            name="SGD basic (lr=0.01, momentum=0.9)",
            custom_cls=SGDMomentum,
            reference_cls=torch.optim.SGD,
            kwargs=dict(lr=0.01, momentum=0.9),
        ),
        TestConfig(
            name="SGD high momentum (momentum=0.99)",
            custom_cls=SGDMomentum,
            reference_cls=torch.optim.SGD,
            kwargs=dict(lr=0.001, momentum=0.99),
        ),
        TestConfig(
            name="SGD with weight decay",
            custom_cls=SGDMomentum,
            reference_cls=torch.optim.SGD,
            kwargs=dict(lr=0.01, momentum=0.9, weight_decay=0.01),
        ),
        TestConfig(
            name="SGD with dampening",
            custom_cls=SGDMomentum,
            reference_cls=torch.optim.SGD,
            kwargs=dict(lr=0.01, momentum=0.9, dampening=0.1),
        ),
        TestConfig(
            name="SGD nesterov",
            custom_cls=SGDMomentum,
            reference_cls=torch.optim.SGD,
            kwargs=dict(lr=0.01, momentum=0.9, nesterov=True),
        ),
        TestConfig(
            name="SGD nesterov + weight decay",
            custom_cls=SGDMomentum,
            reference_cls=torch.optim.SGD,
            kwargs=dict(lr=0.01, momentum=0.9, nesterov=True, weight_decay=0.01),
        ),
        TestConfig(
            name="SGD low lr (lr=1e-5)",
            custom_cls=SGDMomentum,
            reference_cls=torch.optim.SGD,
            kwargs=dict(lr=1e-5, momentum=0.9),
        ),
    ]


def get_rmsprop_configs() -> list[TestConfig]:
    """Test configurations for RMSProp."""
    return [
        TestConfig(
            name="RMSProp basic",
            custom_cls=RMSProp,
            reference_cls=torch.optim.RMSprop,
            kwargs=dict(lr=0.01),
            rtol=1e-3,
            atol=1e-4,
        ),
        TestConfig(
            name="RMSProp with alpha=0.9",
            custom_cls=RMSProp,
            reference_cls=torch.optim.RMSprop,
            kwargs=dict(lr=0.01, alpha=0.9),
        ),
        TestConfig(
            name="RMSProp with momentum",
            custom_cls=RMSProp,
            reference_cls=torch.optim.RMSprop,
            kwargs=dict(lr=0.01, momentum=0.9),
        ),
        TestConfig(
            name="RMSProp centered",
            custom_cls=RMSProp,
            reference_cls=torch.optim.RMSprop,
            kwargs=dict(lr=0.01, centered=True),
        ),
        TestConfig(
            name="RMSProp centered + momentum",
            custom_cls=RMSProp,
            reference_cls=torch.optim.RMSprop,
            kwargs=dict(lr=0.01, centered=True, momentum=0.9),
            rtol=1e-3,
            atol=1e-4,
        ),
        TestConfig(
            name="RMSProp with weight decay",
            custom_cls=RMSProp,
            reference_cls=torch.optim.RMSprop,
            kwargs=dict(lr=0.01, weight_decay=0.01),
        ),
        TestConfig(
            name="RMSProp all options",
            custom_cls=RMSProp,
            reference_cls=torch.optim.RMSprop,
            kwargs=dict(lr=0.001, alpha=0.95, momentum=0.9, centered=True, weight_decay=0.01),
        ),
    ]


def get_adam_configs() -> list[TestConfig]:
    """Test configurations for Adam."""
    return [
        TestConfig(
            name="Adam basic",
            custom_cls=Adam,
            reference_cls=torch.optim.Adam,
            kwargs=dict(lr=0.001),
        ),
        TestConfig(
            name="Adam custom betas",
            custom_cls=Adam,
            reference_cls=torch.optim.Adam,
            kwargs=dict(lr=0.001, betas=(0.85, 0.995)),
        ),
        TestConfig(
            name="Adam with weight decay (L2)",
            custom_cls=Adam,
            reference_cls=torch.optim.Adam,
            kwargs=dict(lr=0.001, weight_decay=0.01),
        ),
        TestConfig(
            name="Adam amsgrad",
            custom_cls=Adam,
            reference_cls=torch.optim.Adam,
            kwargs=dict(lr=0.001, amsgrad=True),
        ),
        TestConfig(
            name="Adam amsgrad + weight decay",
            custom_cls=Adam,
            reference_cls=torch.optim.Adam,
            kwargs=dict(lr=0.001, amsgrad=True, weight_decay=0.01),
        ),
        TestConfig(
            name="Adam higher lr",
            custom_cls=Adam,
            reference_cls=torch.optim.Adam,
            kwargs=dict(lr=0.01),
        ),
        TestConfig(
            name="Adam low eps",
            custom_cls=Adam,
            reference_cls=torch.optim.Adam,
            kwargs=dict(lr=0.001, eps=1e-10),
        ),
    ]


def get_adamw_configs() -> list[TestConfig]:
    """Test configurations for AdamW."""
    return [
        TestConfig(
            name="AdamW basic",
            custom_cls=AdamW,
            reference_cls=torch.optim.AdamW,
            kwargs=dict(lr=0.001),
        ),
        TestConfig(
            name="AdamW with weight decay",
            custom_cls=AdamW,
            reference_cls=torch.optim.AdamW,
            kwargs=dict(lr=0.001, weight_decay=0.1),
        ),
        TestConfig(
            name="AdamW custom betas",
            custom_cls=AdamW,
            reference_cls=torch.optim.AdamW,
            kwargs=dict(lr=0.001, betas=(0.9, 0.99), weight_decay=0.01),
        ),
        TestConfig(
            name="AdamW amsgrad",
            custom_cls=AdamW,
            reference_cls=torch.optim.AdamW,
            kwargs=dict(lr=0.001, amsgrad=True),
        ),
        TestConfig(
            name="AdamW amsgrad + weight decay",
            custom_cls=AdamW,
            reference_cls=torch.optim.AdamW,
            kwargs=dict(lr=0.001, amsgrad=True, weight_decay=0.05),
        ),
        TestConfig(
            name="AdamW high weight decay",
            custom_cls=AdamW,
            reference_cls=torch.optim.AdamW,
            kwargs=dict(lr=0.001, weight_decay=0.3),
        ),
    ]


def run_muon_sanity_checks(data: list, seed: int = 123) -> bool:
    """
    Muon has no PyTorch reference, so we run sanity checks:
    1. Loss decreases over training
    2. Gradients flow correctly
    3. Different configs produce different results
    """
    print("  Running sanity checks (no PyTorch reference)...")

    all_passed = True

    # Test 1: Loss decreases
    torch.manual_seed(seed)
    model = ToyMLP()
    optimizer = Muon(model.parameters(), lr=0.02, momentum=0.95)
    criterion = nn.CrossEntropyLoss()

    initial_loss = None
    final_loss = None

    for step in range(50):
        x, y = data[step % len(data)]
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)

        if step == 0:
            initial_loss = loss.item()
        if step == 49:
            final_loss = loss.item()

        loss.backward()
        optimizer.step()

    if final_loss < initial_loss:
        print(f"  ✓ PASS: Loss decreased ({initial_loss:.4f} -> {final_loss:.4f})")
    else:
        print(f"  ✗ FAIL: Loss did not decrease ({initial_loss:.4f} -> {final_loss:.4f})")
        all_passed = False

    # Test 2: Nesterov vs non-Nesterov produce different results
    torch.manual_seed(seed)
    model1 = ToyMLP()
    opt1 = Muon(model1.parameters(), lr=0.02, momentum=0.95, nesterov=True)

    torch.manual_seed(seed)
    model2 = ToyMLP()
    opt2 = Muon(model2.parameters(), lr=0.02, momentum=0.95, nesterov=False)

    for step in range(20):
        x, y = data[step % len(data)]

        opt1.zero_grad()
        loss1 = criterion(model1(x), y)
        loss1.backward()
        opt1.step()

        opt2.zero_grad()
        loss2 = criterion(model2(x), y)
        loss2.backward()
        opt2.step()

    params_differ = False
    for (n1, p1), (n2, p2) in zip(
        model1.named_parameters(), model2.named_parameters(), strict=False
    ):
        if not torch.allclose(p1, p2, atol=1e-6):
            params_differ = True
            break

    if params_differ:
        print("  ✓ PASS: Nesterov vs non-Nesterov produce different results")
    else:
        print("  ✗ FAIL: Nesterov vs non-Nesterov produced same results")
        all_passed = False

    # Test 3: 1D params (biases) are updated correctly
    torch.manual_seed(seed)
    model = ToyMLP()
    initial_bias = model.fc1.bias.clone()
    optimizer = Muon(model.parameters(), lr=0.02, momentum=0.95)

    for step in range(10):
        x, y = data[step % len(data)]
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    bias_updated = not torch.allclose(model.fc1.bias, initial_bias, atol=1e-8)
    if bias_updated:
        print("  ✓ PASS: 1D parameters (biases) are updated")
    else:
        print("  ✗ FAIL: 1D parameters (biases) not updated")
        all_passed = False

    # Test 4: Different ns_steps produce different results
    torch.manual_seed(seed)
    model1 = ToyMLP()
    opt1 = Muon(model1.parameters(), lr=0.02, ns_steps=3)

    torch.manual_seed(seed)
    model2 = ToyMLP()
    opt2 = Muon(model2.parameters(), lr=0.02, ns_steps=7)

    for step in range(20):
        x, y = data[step % len(data)]

        opt1.zero_grad()
        loss1 = criterion(model1(x), y)
        loss1.backward()
        opt1.step()

        opt2.zero_grad()
        loss2 = criterion(model2(x), y)
        loss2.backward()
        opt2.step()

    params_differ = False
    for (n1, p1), (n2, p2) in zip(
        model1.named_parameters(), model2.named_parameters(), strict=False
    ):
        if not torch.allclose(p1, p2, atol=1e-6):
            params_differ = True
            break

    if params_differ:
        print("  ✓ PASS: Different ns_steps produce different results")
    else:
        print("  ✗ FAIL: Different ns_steps produced same results")
        all_passed = False

    return all_passed


def main():
    print("=" * 60)
    print("Custom Optimizer Test Suite")
    print("=" * 60)

    # Create shared toy data
    data = create_toy_data()

    results = {"passed": 0, "failed": 0}

    # Test SGD
    print("\n[SGDMomentum]")
    for config in get_sgd_configs():
        if run_test(config, data):
            results["passed"] += 1
        else:
            results["failed"] += 1

    # Test RMSProp
    print("\n[RMSProp]")
    for config in get_rmsprop_configs():
        if run_test(config, data):
            results["passed"] += 1
        else:
            results["failed"] += 1

    # Test Adam
    print("\n[Adam]")
    for config in get_adam_configs():
        if run_test(config, data):
            results["passed"] += 1
        else:
            results["failed"] += 1

    # Test AdamW
    print("\n[AdamW]")
    for config in get_adamw_configs():
        if run_test(config, data):
            results["passed"] += 1
        else:
            results["failed"] += 1

    # Test Muon (sanity checks only)
    print("\n[Muon]")
    if run_muon_sanity_checks(data):
        results["passed"] += 1
    else:
        results["failed"] += 1

    # Summary
    print("\n" + "=" * 60)
    total = results["passed"] + results["failed"]
    print(f"Results: {results['passed']}/{total} tests passed")

    if results["failed"] > 0:
        print(f"         {results['failed']} tests FAILED")
        return 1
    else:
        print("All tests passed!")
        return 0


if __name__ == "__main__":
    exit(main())
