"""
Benchmark script to compare MHA vs GQA vs MQA.
Measures memory usage and generation latency.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.cuda

sys.path.append(str(Path(__file__).parent.parent))
from gpt.model import GPT, GPTConfig


def measure_memory_and_latency(
    model: GPT,
    prompt_len: int,
    gen_len: int,
    batch_size: int,
    device: torch.device,
    num_warmup: int = 3,
    num_runs: int = 10,
    use_cache: bool = True,
) -> dict[str, float]:
    """
    Measure peak memory and average latency for generation.

    Returns dict with:
        - peak_memory_mb: Peak GPU memory allocated
        - avg_latency_ms: Average generation time
        - tokens_per_sec: Generation throughput
    """
    model.eval()

    # Create dummy prompt
    prompt = torch.randint(0, model.config.vocab_size, (batch_size, prompt_len), device=device)

    # Warmup runs
    for _ in range(num_warmup):
        with torch.no_grad():
            if use_cache:
                _ = model.generate(prompt, max_new_tokens=gen_len)
            else:
                _ = model.generate_without_cache(prompt, max_new_tokens=gen_len)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    for i in range(num_runs):
        start_events[i].record()
        with torch.no_grad():
            if use_cache:
                _ = model.generate(prompt, max_new_tokens=gen_len)
            else:
                _ = model.generate_without_cache(prompt, max_new_tokens=gen_len)
        end_events[i].record()

    torch.cuda.synchronize()

    # Compute metrics
    latencies = [start_events[i].elapsed_time(end_events[i]) for i in range(num_runs)]
    avg_latency = sum(latencies) / len(latencies)
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2

    total_tokens = batch_size * gen_len
    tokens_per_sec = total_tokens / (avg_latency / 1000)

    return {
        "peak_memory_mb": peak_memory,
        "avg_latency_ms": avg_latency,
        "tokens_per_sec": tokens_per_sec,
        "latency_std_ms": (sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)) ** 0.5,
    }


def estimate_kv_cache_size(
    config: GPTConfig, seq_len: int, batch_size: int, dtype_bytes: int = 4
) -> float:
    """Estimate KV cache size in MB."""
    # Cache stores K and V for each layer
    # Shape per layer: (batch, num_kv_heads, seq_len, d_k) * 2 (K and V)
    d_k = config.d_model // config.num_heads
    cache_elements = batch_size * config.num_kv_heads * seq_len * d_k * 2 * config.num_layers
    return cache_elements * dtype_bytes / 1024**2


def run_comparison(
    d_model: int = 256,
    num_heads: int = 8,
    num_layers: int = 6,
    max_seq_len: int = 512,
    vocab_size: int = 50257,
    prompt_len: int = 64,
    gen_len: int = 128,
    batch_size: int = 1,
    device: str = "cuda",
) -> list[dict]:
    """Run comparison across MHA, GQA, and MQA configurations."""

    device = torch.device(device)

    # Configurations to test
    configs = [
        ("MHA", num_heads),  # Standard multi-head attention
        ("GQA-4", num_heads // 2),  # Grouped query attention (4 KV heads for 8 Q heads)
        ("GQA-2", num_heads // 4),  # More aggressive GQA (2 KV heads)
        ("MQA", 1),  # Multi-query attention (1 KV head)
    ]

    results = []

    for name, num_kv_heads in configs:
        print(f"\n{'=' * 60}")
        print(f"Testing {name} (num_kv_heads={num_kv_heads})")
        print(f"{'=' * 60}")

        # Create config
        config = GPTConfig(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            vocab_size=vocab_size,
            dropout=0.0,  # Disable for inference
            rope=True,
            rmsnorm=True,
        )

        # Create model
        model = GPT(config).to(device)
        model.eval()

        num_params = model.get_num_params()
        kv_cache_estimate = estimate_kv_cache_size(config, prompt_len + gen_len, batch_size)

        print(f"Parameters: {num_params:,}")
        print(f"Estimated KV cache size: {kv_cache_estimate:.2f} MB")

        # Benchmark with cache
        print("\nWith KV-cache:")
        metrics_cached = measure_memory_and_latency(
            model, prompt_len, gen_len, batch_size, device, use_cache=True
        )
        print(f"  Peak memory: {metrics_cached['peak_memory_mb']:.1f} MB")
        print(f"  Avg latency: {metrics_cached['avg_latency_ms']:.1f} ms")
        print(f"  Throughput: {metrics_cached['tokens_per_sec']:.1f} tok/s")

        # Benchmark without cache (for comparison)
        print("\nWithout KV-cache:")
        metrics_uncached = measure_memory_and_latency(
            model, prompt_len, gen_len, batch_size, device, use_cache=False
        )
        print(f"  Peak memory: {metrics_uncached['peak_memory_mb']:.1f} MB")
        print(f"  Avg latency: {metrics_uncached['avg_latency_ms']:.1f} ms")
        print(f"  Throughput: {metrics_uncached['tokens_per_sec']:.1f} tok/s")

        results.append(
            {
                "name": name,
                "num_kv_heads": num_kv_heads,
                "num_params": num_params,
                "kv_cache_estimate_mb": kv_cache_estimate,
                "cached": metrics_cached,
                "uncached": metrics_uncached,
            }
        )

        # Clean up
        del model
        torch.cuda.empty_cache()

    return results


def print_summary(results: list[dict]):
    """Print comparison summary table."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Header
    print(
        f"{'Config':<10} {'KV Heads':>10} {'Params':>12} {'KV Cache':>12} {'Memory':>12} {'Latency':>12} {'Tok/s':>10}"
    )
    print(f"{'':^10} {'':>10} {'':>12} {'(est. MB)':>12} {'(MB)':>12} {'(ms)':>12} {'':>10}")
    print("-" * 80)

    baseline_latency = results[0]["cached"]["avg_latency_ms"]
    baseline_memory = results[0]["cached"]["peak_memory_mb"]

    for r in results:
        latency_ratio = r["cached"]["avg_latency_ms"] / baseline_latency
        memory_ratio = r["cached"]["peak_memory_mb"] / baseline_memory

        print(
            f"{r['name']:<10} "
            f"{r['num_kv_heads']:>10} "
            f"{r['num_params']:>12,} "
            f"{r['kv_cache_estimate_mb']:>12.2f} "
            f"{r['cached']['peak_memory_mb']:>10.1f} ({memory_ratio:.2f}x) "
            f"{r['cached']['avg_latency_ms']:>10.1f} ({latency_ratio:.2f}x) "
            f"{r['cached']['tokens_per_sec']:>10.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark MHA vs GQA vs MQA")

    # Model config
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=50257)

    # Generation config
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--gen_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)

    # Output
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. This benchmark requires a GPU.")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"Config: d_model={args.d_model}, num_heads={args.num_heads}, num_layers={args.num_layers}"
    )
    print(
        f"Generation: prompt_len={args.prompt_len}, gen_len={args.gen_len}, batch_size={args.batch_size}"
    )

    results = run_comparison(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        vocab_size=args.vocab_size,
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        batch_size=args.batch_size,
    )

    print_summary(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
