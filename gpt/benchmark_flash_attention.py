"""
Benchmark different attention implementations.

Compares:
- Standard attention (manual implementation)
- SDPA (PyTorch's scaled_dot_product_attention, uses FlashAttention when available)
- Naive FlashAttention (our educational implementation)
- Naive FlashAttention-2 (our educational implementation)

Usage:
    python benchmark_flash_attention.py
    python benchmark_flash_attention.py --seq_len 512 --d_model 512 --num_heads 8
"""

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))

from gpt.model import (
    GPT,
    FlashAttention2Naive,
    FlashAttentionNaive,
    GPTConfig,
    attention_sdpa,
    attention_standard,
)


@dataclass
class BenchmarkResult:
    name: str
    avg_time_ms: float
    std_time_ms: float
    memory_mb: float
    correct: bool


def create_test_tensors(batch_size: int, num_heads: int, seq_len: int, d_k: int, device: str):
    """Create random Q, K, V tensors for benchmarking."""
    Q = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=torch.float32)
    K = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=torch.float32)
    V = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=torch.float32)
    return Q, K, V


def benchmark_fn(fn: Callable, Q, K, V, num_warmup: int = 5, num_runs: int = 20) -> BenchmarkResult:
    """Benchmark a single attention function."""
    device = Q.device

    # Warmup
    for _ in range(num_warmup):
        _ = fn(Q, K, V)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Timed runs
    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = fn(Q, K, V)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            start = time.perf_counter()
            _ = fn(Q, K, V)
            times.append((time.perf_counter() - start) * 1000)

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    if device.type == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        memory_mb = 0.0

    return avg_time, std_time, memory_mb


def check_correctness(
    fn: Callable, Q, K, V, reference_output: torch.Tensor, atol: float = 1e-4
) -> bool:
    """Check if function output matches reference."""
    output = fn(Q, K, V)
    if isinstance(output, tuple):
        output = output[0]
    return torch.allclose(output, reference_output, atol=atol, rtol=1e-3)


def run_benchmarks(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_len: int = 256,
    d_k: int = 64,
    device: str = "cuda",
) -> dict[str, BenchmarkResult]:
    """Run all attention benchmarks."""

    device = torch.device(device)
    Q, K, V = create_test_tensors(batch_size, num_heads, seq_len, d_k, device)

    # Create causal mask for standard attention
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)
    dropout = torch.nn.Dropout(0.0)

    # Reference implementation (standard attention)
    def standard_fn(Q, K, V):
        return attention_standard(Q, K, V, mask, dropout, training=False)

    reference_output = standard_fn(Q, K, V)

    # Implementations to benchmark
    implementations = {}

    # Standard attention
    implementations["standard"] = standard_fn

    # SDPA
    def sdpa_fn(Q, K, V):
        return attention_sdpa(Q, K, V, is_causal=True, dropout_p=0.0, training=False)

    implementations["sdpa"] = sdpa_fn

    # Naive FlashAttention
    flash1 = FlashAttentionNaive(d_k, causal=True).to(device)
    implementations["flash_naive"] = lambda Q, K, V: flash1(Q, K, V)

    # Naive FlashAttention-2
    flash2 = FlashAttention2Naive(d_k, causal=True).to(device)
    implementations["flash2_naive"] = lambda Q, K, V: flash2(Q, K, V)[0]

    # Run benchmarks
    results = {}

    for name, fn in implementations.items():
        # print(f"Benchmarking {name}...")

        # Check correctness
        correct = check_correctness(fn, Q, K, V, reference_output)

        # Benchmark
        avg_time, std_time, memory_mb = benchmark_fn(fn, Q, K, V)

        results[name] = BenchmarkResult(
            name=name,
            avg_time_ms=avg_time,
            std_time_ms=std_time,
            memory_mb=memory_mb,
            correct=correct,
        )

    return results


def print_results(results: dict[str, BenchmarkResult], seq_len: int, batch_size: int):
    """Print benchmark results in a table."""
    print("\n" + "=" * 80)
    print(f"ATTENTION BENCHMARK RESULTS (seq_len={seq_len}, batch_size={batch_size})")
    print("=" * 80)

    baseline = results["standard"].avg_time_ms

    print(
        f"{'Implementation':<20} {'Time (ms)':<15} {'Std (ms)':<12} {'Memory (MB)':<15} {'Speedup':<10} {'Correct'}"
    )
    print("-" * 80)

    for name, result in results.items():
        speedup = baseline / result.avg_time_ms
        correct_str = "✓" if result.correct else "✗"
        print(
            f"{result.name:<20} "
            f"{result.avg_time_ms:<15.3f} "
            f"{result.std_time_ms:<12.3f} "
            f"{result.memory_mb:<15.1f} "
            f"{speedup:.2f}x        "
            f"{correct_str}"
        )

    print("=" * 80)


def benchmark_scaling(device: str = "cuda"):
    """Benchmark how implementations scale with sequence length."""
    print("\n" + "=" * 80)
    print("SCALING BENCHMARK (varying sequence length)")
    print("=" * 80)

    seq_lens = [128, 256, 512, 1024]
    batch_size = 2
    num_heads = 8
    d_k = 64

    print(f"{'Seq Len':<10} {'Standard':<15} {'SDPA':<15} {'Flash Naive':<15} {'Flash2 Naive':<15}")
    print("-" * 70)

    for seq_len in seq_lens:
        results = run_benchmarks(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            d_k=d_k,
            device=device,
        )

        print(
            f"{seq_len:<10} "
            f"{results['standard'].avg_time_ms:<15.2f} "
            f"{results['sdpa'].avg_time_ms:<15.2f} "
            f"{results['flash_naive'].avg_time_ms:<15.2f} "
            f"{results['flash2_naive'].avg_time_ms:<15.2f}"
        )

    print("=" * 80)


def benchmark_full_model(device: str = "cuda"):
    """Benchmark full model forward pass with different attention types."""
    print("\n" + "=" * 80)
    print("FULL MODEL BENCHMARK (forward pass)")
    print("=" * 80)

    batch_size = 4
    seq_len = 256

    attn_types = ["standard", "sdpa", "flash", "flash2"]
    results = {}

    for attn_type in attn_types:
        print(f"Benchmarking model with {attn_type} attention...")

        config = GPTConfig(
            d_model=256,
            num_heads=8,
            num_layers=6,
            max_seq_len=seq_len,
            vocab_size=1000,
            dropout=0.0,
            rope=True,
            rmsnorm=True,
            attn_type=attn_type,
        )

        model = GPT(config).to(device)
        model.eval()

        x = torch.randint(0, 1000, (batch_size, seq_len), device=device)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(x)

        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # Timed runs
        times = []
        for _ in range(10):
            if device == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.no_grad():
                    _ = model(x)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                start = time.perf_counter()
                with torch.no_grad():
                    _ = model(x)
                times.append((time.perf_counter() - start) * 1000)

        avg_time = sum(times) / len(times)
        memory_mb = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0

        results[attn_type] = (avg_time, memory_mb)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Print results
    baseline = results["standard"][0]

    print(f"\n{'Attention Type':<20} {'Time (ms)':<15} {'Memory (MB)':<15} {'Speedup'}")
    print("-" * 60)

    for attn_type, (avg_time, memory_mb) in results.items():
        speedup = baseline / avg_time
        print(f"{attn_type:<20} {avg_time:<15.2f} {memory_mb:<15.1f} {speedup:.2f}x")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark attention implementations")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--d_k", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--scaling", action="store_true", help="Run scaling benchmark")
    parser.add_argument("--full_model", action="store_true", help="Run full model benchmark")

    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        print("CUDA not available, using CPU")
        args.device = "cpu"

    print(f"Device: {args.device}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Run attention-only benchmarks
    results = run_benchmarks(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        d_k=args.d_k,
        device=args.device,
    )
    print_results(results, args.seq_len, args.batch_size)

    # Optional benchmarks
    if args.scaling:
        benchmark_scaling(args.device)

    if args.full_model:
        benchmark_full_model(args.device)


if __name__ == "__main__":
    main()
