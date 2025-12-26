"""
Benchmark mixed precision (bfloat16) vs full precision (fp32).

Compares forward pass, backward pass, and full training step performance
as well as memory usage.

Usage:
    python benchmark_amp.py
    python benchmark_amp.py --d_model 512 --num_layers 12 --batch_size 32
"""

import torch
import torch.nn.functional as F
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

from model import GPT, GPTConfig


@dataclass
class BenchmarkResult:
    name: str
    forward_ms: float
    backward_ms: float
    step_ms: float
    memory_mb: float
    throughput_tok_per_sec: float


def benchmark_forward(
    model: GPT,
    x: torch.Tensor,
    use_amp: bool,
    num_warmup: int = 5,
    num_runs: int = 20,
) -> Tuple[float, float]:
    """Benchmark forward pass. Returns (avg_time_ms, std_time_ms)."""
    model.eval()

    # Warmup
    for _ in range(num_warmup):
        with torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
        ):
            _ = model(x)

    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
        ):
            _ = model(x)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    return avg_time, std_time


def benchmark_backward(
    model: GPT,
    x: torch.Tensor,
    y: torch.Tensor,
    use_amp: bool,
    num_warmup: int = 5,
    num_runs: int = 20,
) -> Tuple[float, float]:
    """Benchmark forward + backward pass. Returns (avg_time_ms, std_time_ms)."""
    model.train()

    # Warmup
    for _ in range(num_warmup):
        model.zero_grad()
        with torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
        ):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(num_runs):
        model.zero_grad()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
        ):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    return avg_time, std_time


def benchmark_full_step(
    model: GPT,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    use_amp: bool,
    num_warmup: int = 5,
    num_runs: int = 20,
) -> Tuple[float, float]:
    """Benchmark full training step (forward + backward + optimizer). Returns (avg_time_ms, std_time_ms)."""
    model.train()

    # Warmup
    for _ in range(num_warmup):
        optimizer.zero_grad()
        with torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
        ):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        optimizer.zero_grad()
        with torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
        ):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    return avg_time, std_time


def measure_memory(
    model: GPT,
    x: torch.Tensor,
    y: torch.Tensor,
    use_amp: bool,
) -> float:
    """Measure peak memory usage during forward + backward. Returns MB."""
    model.train()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model.zero_grad()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()

    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2

    return peak_memory


def run_benchmark(
    d_model: int = 256,
    num_heads: int = 8,
    num_layers: int = 6,
    max_seq_len: int = 256,
    batch_size: int = 16,
    vocab_size: int = 50257,
) -> Dict[str, BenchmarkResult]:
    """Run full benchmark comparing fp32 vs bfloat16."""

    device = torch.device("cuda")

    # Check bfloat16 support
    if not torch.cuda.is_bf16_supported():
        print("ERROR: bfloat16 not supported on this GPU")
        return {}

    config = GPTConfig(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        dropout=0.0,
        rope=True,
        rmsnorm=True,
        attn_type="sdpa",
    )

    results = {}

    for name, use_amp in [("fp32", False), ("bfloat16", True)]:
        # print(f"\nBenchmarking {name}...")

        # Create fresh model for each test
        model = GPT(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Create test data
        x = torch.randint(0, vocab_size, (batch_size, max_seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, max_seq_len), device=device)

        # Benchmark each component
        forward_ms, _ = benchmark_forward(model, x, use_amp)
        backward_ms, _ = benchmark_backward(model, x, y, use_amp)
        step_ms, _ = benchmark_full_step(model, x, y, optimizer, use_amp)
        memory_mb = measure_memory(model, x, y, use_amp)

        # Calculate throughput
        total_tokens = batch_size * max_seq_len
        throughput = total_tokens / (step_ms / 1000)

        results[name] = BenchmarkResult(
            name=name,
            forward_ms=forward_ms,
            backward_ms=backward_ms,
            step_ms=step_ms,
            memory_mb=memory_mb,
            throughput_tok_per_sec=throughput,
        )

        # Cleanup
        del model, optimizer
        torch.cuda.empty_cache()

    return results


def print_results(results: Dict[str, BenchmarkResult], config_str: str):
    """Print benchmark results in a table."""
    print("\n" + "=" * 80)
    print("AMP BENCHMARK RESULTS")
    print(f"Config: {config_str}")
    print("=" * 80)

    fp32 = results.get("fp32")
    bf16 = results.get("bfloat16")

    if not fp32 or not bf16:
        print("Missing results!")
        return

    print(f"\n{'Metric':<25} {'fp32':<15} {'bfloat16':<15} {'Speedup':<10}")
    print("-" * 65)

    # Forward pass
    speedup = fp32.forward_ms / bf16.forward_ms
    print(
        f"{'Forward (ms)':<25} {fp32.forward_ms:<15.2f} {bf16.forward_ms:<15.2f} {speedup:.2f}x"
    )

    # Backward pass (includes forward)
    speedup = fp32.backward_ms / bf16.backward_ms
    print(
        f"{'Forward+Backward (ms)':<25} {fp32.backward_ms:<15.2f} {bf16.backward_ms:<15.2f} {speedup:.2f}x"
    )

    # Full step
    speedup = fp32.step_ms / bf16.step_ms
    print(
        f"{'Full Step (ms)':<25} {fp32.step_ms:<15.2f} {bf16.step_ms:<15.2f} {speedup:.2f}x"
    )

    # Memory
    reduction = (fp32.memory_mb - bf16.memory_mb) / fp32.memory_mb * 100
    print(
        f"{'Peak Memory (MB)':<25} {fp32.memory_mb:<15.1f} {bf16.memory_mb:<15.1f} {reduction:.1f}% less"
    )

    # Throughput
    speedup = bf16.throughput_tok_per_sec / fp32.throughput_tok_per_sec
    print(
        f"{'Throughput (tok/s)':<25} {fp32.throughput_tok_per_sec:<15.0f} {bf16.throughput_tok_per_sec:<15.0f} {speedup:.2f}x"
    )

    print("=" * 80)


def benchmark_scaling(batch_sizes: list = [8, 16, 32, 64]):
    """Benchmark how AMP benefits scale with batch size."""
    print("\n" + "=" * 80)
    print("SCALING BENCHMARK (varying batch size)")
    print("=" * 80)

    print(
        f"\n{'Batch':<10} {'fp32 (ms)':<15} {'bf16 (ms)':<15} {'Speedup':<10} {'fp32 Mem':<12} {'bf16 Mem':<12}"
    )
    print("-" * 75)

    for batch_size in batch_sizes:
        try:
            results = run_benchmark(
                d_model=256,
                num_heads=8,
                num_layers=6,
                max_seq_len=256,
                batch_size=batch_size,
            )

            fp32 = results["fp32"]
            bf16 = results["bfloat16"]
            speedup = fp32.step_ms / bf16.step_ms

            print(
                f"{batch_size:<10} "
                f"{fp32.step_ms:<15.2f} "
                f"{bf16.step_ms:<15.2f} "
                f"{speedup:<10.2f}x "
                f"{fp32.memory_mb:<12.0f} "
                f"{bf16.memory_mb:<12.0f}"
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{batch_size:<10} OOM")
                torch.cuda.empty_cache()
            else:
                raise

    print("=" * 80)


def benchmark_model_size():
    """Benchmark how AMP benefits scale with model size."""
    print("\n" + "=" * 80)
    print("MODEL SIZE BENCHMARK")
    print("=" * 80)

    configs = [
        (256, 8, 6, "Small (256d, 6L)"),
        (512, 8, 6, "Medium (512d, 6L)"),
        (512, 8, 12, "Large (512d, 12L)"),
        (768, 12, 12, "XL (768d, 12L)"),
    ]

    print(
        f"\n{'Config':<20} {'fp32 (ms)':<12} {'bf16 (ms)':<12} {'Speedup':<10} {'Mem Saved':<12}"
    )
    print("-" * 70)

    for d_model, num_heads, num_layers, name in configs:
        try:
            results = run_benchmark(
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                max_seq_len=256,
                batch_size=16,
            )

            fp32 = results["fp32"]
            bf16 = results["bfloat16"]
            speedup = fp32.step_ms / bf16.step_ms
            mem_saved = (fp32.memory_mb - bf16.memory_mb) / fp32.memory_mb * 100

            print(
                f"{name:<20} "
                f"{fp32.step_ms:<12.2f} "
                f"{bf16.step_ms:<12.2f} "
                f"{f'{speedup:.2f}x':<10} "
                f"{f'{mem_saved:.1f}%':<12}"
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{name:<20} OOM")
                torch.cuda.empty_cache()
            else:
                raise

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark AMP (bfloat16) vs fp32")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--scaling", action="store_true", help="Run batch size scaling benchmark"
    )
    parser.add_argument(
        "--model_size", action="store_true", help="Run model size scaling benchmark"
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"bfloat16 supported: {torch.cuda.is_bf16_supported()}")

    if not torch.cuda.is_bf16_supported():
        print("bfloat16 not supported on this GPU. Exiting.")
        return

    # Run main benchmark
    config_str = f"d_model={args.d_model}, layers={args.num_layers}, batch={args.batch_size}, seq_len={args.max_seq_len}"

    results = run_benchmark(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
    )

    print_results(results, config_str)

    # Optional scaling benchmarks
    if args.scaling:
        benchmark_scaling()

    if args.model_size:
        benchmark_model_size()


if __name__ == "__main__":
    main()
