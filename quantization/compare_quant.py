"""
Quantization Comparison Script

Run this after implementing quantization methods to compare their
impact on model quality.

Usage:
    # Basic comparison with default settings (RTN/Absmax only)
    python compare_quant.py --bits 8
    
    # 4-bit quantization
    python compare_quant.py --bits 4
    
    # Include calibration-based methods (GPTQ, AWQ)
    python compare_quant.py --bits 4 --calibration
    
    # Only quantize attention layers (to test sensitivity)
    python compare_quant.py --bits 4 --include attn
    
    # Only quantize MLP layers
    python compare_quant.py --bits 4 --include mlp
    
    # Quantize everything (including embed/lm_head - not recommended)
    python compare_quant.py --bits 8 --exclude none
    
    # Custom layer patterns
    python compare_quant.py --bits 4 --include q_proj,k_proj --exclude layers.0,layers.1
"""

import torch
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from datasets import load_dataset

from quant_eval import (
    load_model,
    load_hellaswag,
    load_wikitext,
    evaluate_hellaswag,
    evaluate_perplexity,
    get_device,
)
from quantization import (
    QuantConfig,
    quantize_model_rtn,
    quantize_model_absmax,
    quantize_model_gptq,
    quantize_model_awq,
    count_model_bits,
    LayerFilter,
)


# -----------------------------------------------------------------------------
# Layer Filtering for Quantization
# -----------------------------------------------------------------------------

# Predefined filter presets
INCLUDE_PRESETS = {
    'all': None,  # None means include all
    'attn': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    'mlp': ['gate_proj', 'up_proj', 'down_proj'],
    'qkv': ['q_proj', 'k_proj', 'v_proj'],
}

EXCLUDE_PRESETS = {
    'none': None,  # None means exclude nothing
    'default': ['embed_tokens', 'lm_head', 'norm'],
    'embed_only': ['embed_tokens'],
    'head_only': ['lm_head'],
}


def make_filter(patterns: Optional[list[str]], match_any: bool = True) -> Optional[LayerFilter]:
    """
    Create a filter function from a list of patterns.
    
    Args:
        patterns: List of substrings to match against layer names
        match_any: If True, match if ANY pattern is in name. If False, match if ALL are.
    
    Returns:
        Filter function or None if patterns is None/empty
    """
    if patterns is None or len(patterns) == 0:
        return None
    
    if match_any:
        return lambda name: any(p in name for p in patterns)
    else:
        return lambda name: all(p in name for p in patterns)


def parse_layer_arg(arg: str, presets: dict) -> Optional[list[str]]:
    """
    Parse a layer filter argument.
    
    Can be:
        - A preset name (e.g., 'default', 'attn', 'mlp')
        - A comma-separated list of patterns (e.g., 'q_proj,k_proj')
        - 'none' or 'all' for no filtering
    
    Returns:
        List of patterns or None
    """
    if arg is None:
        return None
    
    arg = arg.strip().lower()
    
    # Check for preset
    if arg in presets:
        return presets[arg]
    
    # Check for 'none' or 'all'
    if arg in ('none', 'all', ''):
        return None
    
    # Otherwise, split by comma
    return [p.strip() for p in arg.split(',') if p.strip()]


def get_quantized_layers(model, config: QuantConfig) -> list[str]:
    """Get list of layer names that will be quantized."""
    quantized = []
    for name, module in model.named_modules():
        if config.should_quantize(name, module):
            quantized.append(name)
    return quantized


def get_skipped_layers(model, config: QuantConfig) -> list[str]:
    """Get list of Linear layer names that will NOT be quantized."""
    import torch.nn as nn
    skipped = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not config.should_quantize(name, module):
            skipped.append(name)
    return skipped


# -----------------------------------------------------------------------------
# Calibration Data for GPTQ/AWQ
# -----------------------------------------------------------------------------

def generate_calibration_data(
    tokenizer,
    n_samples: int = 128,
    seq_len: int = 512,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train",
) -> list[torch.Tensor]:
    """
    Generate calibration data for GPTQ/AWQ from a text dataset.
    
    Args:
        tokenizer: HuggingFace tokenizer
        n_samples: Number of calibration samples
        seq_len: Sequence length for each sample
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration
        split: Dataset split to use
    
    Returns:
        List of input_ids tensors, each of shape [1, seq_len]
    """
    print(f"Generating {n_samples} calibration samples from {dataset_name}...")
    
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    
    # Concatenate all text
    text = "\n\n".join([item["text"] for item in dataset if item["text"].strip()])
    
    # Tokenize the entire text
    all_tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Create samples of seq_len tokens each
    calibration_data = []
    for i in range(n_samples):
        start_idx = i * seq_len
        if start_idx + seq_len > len(all_tokens):
            # Wrap around if we run out of tokens
            start_idx = start_idx % (len(all_tokens) - seq_len)
        
        tokens = all_tokens[start_idx:start_idx + seq_len]
        input_ids = torch.tensor([tokens], dtype=torch.long)
        calibration_data.append(input_ids)
    
    print(f"  Generated {len(calibration_data)} samples of length {seq_len}")
    return calibration_data


def move_calibration_to_device(calibration_data: list[torch.Tensor], device) -> list[torch.Tensor]:
    """Move calibration data to the specified device."""
    return [x.to(device) for x in calibration_data]


# -----------------------------------------------------------------------------
# bitsandbytes Quantization
# -----------------------------------------------------------------------------

def load_model_bitsandbytes(
    model_name: str,
    bits: int = 4,
) -> tuple:
    """
    Load a model with bitsandbytes quantization.
    
    Args:
        model_name: HuggingFace model identifier
        bits: 4 or 8 bit quantization
    
    Returns:
        (model, tokenizer) tuple
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    if bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",  # normalized float 4-bit (better than int4)
            bnb_4bit_use_double_quant=True,  # quantize the quantization constants too
        )
    elif bits == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        raise ValueError(f"bitsandbytes only supports 4 or 8 bits, got {bits}")
    
    print(f"Loading {model_name} with bitsandbytes {bits}-bit quantization...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    return model, tokenizer


# -----------------------------------------------------------------------------
# Main Comparison Function
# -----------------------------------------------------------------------------

def run_comparison(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    bits: int = 8,
    hellaswag_n: int = 500,
    wikitext_tokens: int = 4096,
    include_patterns: Optional[list[str]] = None,
    exclude_patterns: Optional[list[str]] = ['embed_tokens', 'lm_head', 'norm'],
    run_calibration_methods: bool = False,
    calibration_samples: int = 128,
    calibration_seq_len: int = 512,
    run_bitsandbytes: bool = False,
    save_results: bool = True,
):
    """
    Compare baseline vs quantized model performance.
    
    Args:
        model_name: HuggingFace model identifier
        bits: Quantization bit width (8 or 4)
        hellaswag_n: Number of HellaSwag samples for evaluation
        wikitext_tokens: Number of WikiText tokens for perplexity
        include_patterns: Only quantize layers matching these patterns (None = all)
        exclude_patterns: Skip layers matching these patterns (None = none)
        run_calibration_methods: If True, also run GPTQ and AWQ (slower but better)
        calibration_samples: Number of samples for GPTQ/AWQ calibration
        calibration_seq_len: Sequence length for calibration samples
        run_bitsandbytes: If True, also evaluate bitsandbytes quantization
        save_results: Whether to save results to JSON
    """
    print("=" * 70)
    print(f"QUANTIZATION COMPARISON: {model_name}")
    print(f"Target precision: {bits}-bit")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Build filter functions
    include_fn = make_filter(include_patterns)
    exclude_fn = make_filter(exclude_patterns)
    
    # Print filter configuration
    print("\nLayer filtering:")
    print(f"  Include: {include_patterns if include_patterns else 'all layers'}")
    print(f"  Exclude: {exclude_patterns if exclude_patterns else 'none'}")
    
    # Load model and tokenizer
    model, tokenizer = load_model(model_name)
    device = get_device(model)
    
    # Create config and show which layers will be quantized
    config = QuantConfig(
        bits=bits, 
        symmetric=True, 
        per_channel=True,
        include=include_fn,
        exclude=exclude_fn,
    )
    
    quantized_layers = get_quantized_layers(model, config)
    skipped_layers = get_skipped_layers(model, config)
    
    print(f"\nLayers to quantize: {len(quantized_layers)}")
    if len(quantized_layers) <= 20:
        for layer in quantized_layers:
            print(f"    ✓ {layer}")
    else:
        for layer in quantized_layers[:5]:
            print(f"    ✓ {layer}")
        print(f"    ... ({len(quantized_layers) - 10} more)")
        for layer in quantized_layers[-5:]:
            print(f"    ✓ {layer}")
    
    print(f"\nLayers to skip: {len(skipped_layers)}")
    for layer in skipped_layers[:10]:
        print(f"    ✗ {layer}")
    if len(skipped_layers) > 10:
        print(f"    ... ({len(skipped_layers) - 10} more)")
    
    # Memory estimation
    mem_stats = count_model_bits(model, config)
    print("\nMemory estimation:")
    print(f"  FP16: {mem_stats['fp16_mb']:.1f} MB")
    print(f"  Quantized: {mem_stats['quantized_mb']:.1f} MB")
    print(f"  Compression: {mem_stats['compression_ratio']:.2f}x")
    print(f"  Params quantized: {mem_stats['quantized_params']:,} / {mem_stats['total_params']:,}")
    
    # Load datasets once (reuse across all experiments)
    print("\n" + "-" * 70)
    print("Loading Evaluation Datasets")
    print("-" * 70)
    hellaswag = load_hellaswag(tokenizer, n_samples=hellaswag_n)
    wikitext = load_wikitext(tokenizer, max_tokens=wikitext_tokens)
    
    # Results storage
    all_results = {
        "model": model_name,
        "bits": bits,
        "timestamp": datetime.now().isoformat(),
        "layer_config": {
            "include_patterns": include_patterns,
            "exclude_patterns": exclude_patterns,
            "quantized_layers": quantized_layers,
            "skipped_layers": skipped_layers,
            "num_quantized": len(quantized_layers),
            "num_skipped": len(skipped_layers),
        },
        "memory": mem_stats,
        "experiments": {},
    }
    
    # ---------------------------------------------------------------------
    # Experiment 1: Baseline (FP16)
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Experiment 1: BASELINE (FP16)")
    print("=" * 70)
    
    baseline_hella = evaluate_hellaswag(model, tokenizer, hellaswag)
    baseline_ppl = evaluate_perplexity(model, tokenizer, wikitext)
    
    all_results["experiments"]["baseline"] = {
        "precision": "fp16",
        "hellaswag_accuracy": baseline_hella["accuracy"],
        "wikitext_perplexity": baseline_ppl["perplexity"],
    }
    
    # ---------------------------------------------------------------------
    # Experiment 2: RTN Per-Tensor
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Experiment 2: RTN {bits}-bit (per-tensor)")
    print("=" * 70)
    
    config_tensor = QuantConfig(
        bits=bits, 
        symmetric=True, 
        per_channel=False, 
        include=include_fn,
        exclude=exclude_fn,
    )
    model_rtn_tensor = quantize_model_rtn(model, config_tensor)
    
    rtn_tensor_hella = evaluate_hellaswag(model_rtn_tensor, tokenizer, hellaswag)
    rtn_tensor_ppl = evaluate_perplexity(model_rtn_tensor, tokenizer, wikitext)
    
    all_results["experiments"]["rtn_per_tensor"] = {
        "precision": f"int{bits} (per-tensor)",
        "hellaswag_accuracy": rtn_tensor_hella["accuracy"],
        "wikitext_perplexity": rtn_tensor_ppl["perplexity"],
    }
    
    del model_rtn_tensor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ---------------------------------------------------------------------
    # Experiment 3: RTN Per-Channel
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Experiment 3: RTN {bits}-bit (per-channel)")
    print("=" * 70)
    
    config_channel = QuantConfig(
        bits=bits, 
        symmetric=True, 
        per_channel=True, 
        include=include_fn,
        exclude=exclude_fn,
    )
    model_rtn_channel = quantize_model_rtn(model, config_channel)
    
    rtn_channel_hella = evaluate_hellaswag(model_rtn_channel, tokenizer, hellaswag)
    rtn_channel_ppl = evaluate_perplexity(model_rtn_channel, tokenizer, wikitext)
    
    all_results["experiments"]["rtn_per_channel"] = {
        "precision": f"int{bits} (per-channel)",
        "hellaswag_accuracy": rtn_channel_hella["accuracy"],
        "wikitext_perplexity": rtn_channel_ppl["perplexity"],
    }
    
    del model_rtn_channel
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ---------------------------------------------------------------------
    # Experiment 4: Absmax Per-Channel
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Experiment 4: Absmax {bits}-bit (per-channel)")
    print("=" * 70)
    
    config_absmax = QuantConfig(
        bits=bits, 
        symmetric=True, 
        per_channel=True, 
        include=include_fn,
        exclude=exclude_fn,
    )
    model_absmax = quantize_model_absmax(model, config_absmax)
    
    absmax_hella = evaluate_hellaswag(model_absmax, tokenizer, hellaswag)
    absmax_ppl = evaluate_perplexity(model_absmax, tokenizer, wikitext)
    
    all_results["experiments"]["absmax_per_channel"] = {
        "precision": f"int{bits} absmax (per-channel)",
        "hellaswag_accuracy": absmax_hella["accuracy"],
        "wikitext_perplexity": absmax_ppl["perplexity"],
    }
    
    del model_absmax
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ---------------------------------------------------------------------
    # Experiment 5 & 6: GPTQ and AWQ (calibration-based methods)
    # ---------------------------------------------------------------------
    if run_calibration_methods:
        # Generate calibration data
        print("\n" + "-" * 70)
        print("Generating Calibration Data for GPTQ/AWQ")
        print("-" * 70)
        
        calibration_data = generate_calibration_data(
            tokenizer,
            n_samples=calibration_samples,
            seq_len=calibration_seq_len,
        )
        calibration_data = move_calibration_to_device(calibration_data, device)
        
        # -----------------------------------------------------------------
        # Experiment 5: GPTQ
        # -----------------------------------------------------------------
        print("\n" + "=" * 70)
        print(f"Experiment 5: GPTQ {bits}-bit")
        print("=" * 70)
        
        config_gptq = QuantConfig(
            bits=bits, 
            symmetric=True, 
            per_channel=True,
            include=include_fn,
            exclude=exclude_fn,
        )
        
        try:
            # Use fast mode (sequential=False) by default
            model_gptq = quantize_model_gptq(model, calibration_data, config_gptq, sequential=False)
            
            gptq_hella = evaluate_hellaswag(model_gptq, tokenizer, hellaswag)
            gptq_ppl = evaluate_perplexity(model_gptq, tokenizer, wikitext)
            
            all_results["experiments"]["gptq"] = {
                "precision": f"int{bits} GPTQ",
                "hellaswag_accuracy": gptq_hella["accuracy"],
                "wikitext_perplexity": gptq_ppl["perplexity"],
                "calibration_samples": calibration_samples,
                "calibration_seq_len": calibration_seq_len,
            }
            
            del model_gptq
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"  ⚠ GPTQ failed: {e}")
            all_results["experiments"]["gptq"] = {
                "precision": f"int{bits} GPTQ",
                "error": str(e),
            }
        
        # -----------------------------------------------------------------
        # Experiment 6: AWQ
        # -----------------------------------------------------------------
        print("\n" + "=" * 70)
        print(f"Experiment 6: AWQ {bits}-bit")
        print("=" * 70)
        
        config_awq = QuantConfig(
            bits=bits, 
            symmetric=True, 
            per_channel=True,
            include=include_fn,
            exclude=exclude_fn,
        )
        
        try:
            # Use fast mode (sequential=False) by default
            model_awq = quantize_model_awq(model, calibration_data, config_awq, sequential=False)
            
            awq_hella = evaluate_hellaswag(model_awq, tokenizer, hellaswag)
            awq_ppl = evaluate_perplexity(model_awq, tokenizer, wikitext)
            
            all_results["experiments"]["awq"] = {
                "precision": f"int{bits} AWQ",
                "hellaswag_accuracy": awq_hella["accuracy"],
                "wikitext_perplexity": awq_ppl["perplexity"],
                "calibration_samples": calibration_samples,
                "calibration_seq_len": calibration_seq_len,
            }
            
            del model_awq
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"  ⚠ AWQ failed: {e}")
            all_results["experiments"]["awq"] = {
                "precision": f"int{bits} AWQ",
                "error": str(e),
            }
        
        # Clean up calibration data
        del calibration_data
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ---------------------------------------------------------------------
    # Experiment 7: bitsandbytes (if requested)
    # ---------------------------------------------------------------------
    if run_bitsandbytes:
        print("\n" + "=" * 70)
        print(f"Experiment: bitsandbytes {bits}-bit")
        print("=" * 70)
        
        try:
            # Load fresh model with bitsandbytes quantization
            model_bnb, _ = load_model_bitsandbytes(model_name, bits=bits)
            
            bnb_hella = evaluate_hellaswag(model_bnb, tokenizer, hellaswag)
            bnb_ppl = evaluate_perplexity(model_bnb, tokenizer, wikitext)
            
            quant_type = "NF4" if bits == 4 else "int8"
            all_results["experiments"]["bitsandbytes"] = {
                "precision": f"{bits}-bit bnb ({quant_type})",
                "hellaswag_accuracy": bnb_hella["accuracy"],
                "wikitext_perplexity": bnb_ppl["perplexity"],
            }
            
            del model_bnb
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except ImportError as e:
            print(f"  ⚠ bitsandbytes not installed: {e}")
            print("  Install with: pip install bitsandbytes")
            all_results["experiments"]["bitsandbytes"] = {
                "precision": f"{bits}-bit bnb",
                "error": "bitsandbytes not installed",
            }
        except Exception as e:
            print(f"  ⚠ bitsandbytes failed: {e}")
            all_results["experiments"]["bitsandbytes"] = {
                "precision": f"{bits}-bit bnb",
                "error": str(e),
            }
    
    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    # Print layer config summary
    print("\nLayer configuration:")
    print(f"  Quantized: {len(quantized_layers)} layers")
    print(f"  Skipped: {len(skipped_layers)} layers")
    print(f"  Compression: {mem_stats['compression_ratio']:.2f}x")
    if run_calibration_methods:
        print(f"  Calibration: {calibration_samples} samples × {calibration_seq_len} tokens")
    
    print(f"\n{'Method':<30} {'HellaSwag':>12} {'PPL':>10} {'Δ Acc':>10} {'Δ PPL':>10}")
    print("-" * 70)
    
    baseline_acc = baseline_hella["accuracy"]
    baseline_ppl_val = baseline_ppl["perplexity"]
    
    for name, exp in all_results["experiments"].items():
        # Check if experiment failed
        if "error" in exp:
            print(f"{exp['precision']:<30} {'FAILED':>12} {'':>10} {'':>10} {'':>10}")
            continue
            
        acc = exp["hellaswag_accuracy"]
        ppl = exp["wikitext_perplexity"]
        delta_acc = acc - baseline_acc
        delta_ppl = ppl - baseline_ppl_val
        
        print(f"{exp['precision']:<30} {acc:>12.4f} {ppl:>10.2f} {delta_acc:>+10.4f} {delta_ppl:>+10.2f}")
    
    # Save results
    if save_results:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Create descriptive filename
        include_str = "_".join(include_patterns) if include_patterns else "all"
        exclude_str = "_".join(exclude_patterns) if exclude_patterns else "none"
        filename = f"quant_{bits}bit_inc-{include_str}_exc-{exclude_str}_{model_name.replace('/', '_')}.json"
        # Truncate if too long
        if len(filename) > 200:
            filename = f"quant_{bits}bit_{model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = results_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {filepath}")
    
    return all_results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare quantization methods with configurable layer filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: quantize all layers except embed/lm_head/norm (RTN/Absmax only)
  python compare_quant.py --bits 8
  
  # Include calibration-based methods (GPTQ, AWQ) - slower but better quality
  python compare_quant.py --bits 4 --calibration
  
  # Compare with bitsandbytes (production-quality quantization)
  python compare_quant.py --bits 4 --bitsandbytes
  
  # Full comparison: RTN, Absmax, GPTQ, AWQ, and bitsandbytes
  python compare_quant.py --bits 4 --calibration --bitsandbytes
  
  # Only quantize attention layers
  python compare_quant.py --bits 4 --include attn
  
  # Only quantize MLP layers  
  python compare_quant.py --bits 4 --include mlp
  
  # Custom patterns: only q and k projections
  python compare_quant.py --bits 4 --include q_proj,k_proj
  
  # Quantize everything (including embed/lm_head)
  python compare_quant.py --bits 8 --exclude none

Include presets: all, attn, mlp, qkv
Exclude presets: none, default, embed_only, head_only
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        choices=[8, 4, 2],
        help="Quantization bit width"
    )
    parser.add_argument(
        "--include",
        type=str,
        default=None,
        help="Layer patterns to include (preset or comma-separated). Presets: all, attn, mlp, qkv"
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="default",
        help="Layer patterns to exclude (preset or comma-separated). Presets: none, default, embed_only, head_only"
    )
    parser.add_argument(
        "--hellaswag-n",
        type=int,
        default=500,
        help="Number of HellaSwag samples"
    )
    parser.add_argument(
        "--wikitext-tokens",
        type=int,
        default=4096,
        help="Number of WikiText tokens"
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Run calibration-based methods (GPTQ, AWQ) - slower but better quality"
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=64,
        help="Number of calibration samples for GPTQ/AWQ"
    )
    parser.add_argument(
        "--calibration-seq-len",
        type=int,
        default=256,
        help="Sequence length for calibration samples"
    )
    parser.add_argument(
        "--bitsandbytes",
        action="store_true",
        help="Run bitsandbytes quantization (4-bit NF4 or 8-bit int8)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    
    args = parser.parse_args()
    
    # Parse layer filter arguments
    include_patterns = parse_layer_arg(args.include, INCLUDE_PRESETS)
    exclude_patterns = parse_layer_arg(args.exclude, EXCLUDE_PRESETS)
    
    run_comparison(
        model_name=args.model,
        bits=args.bits,
        hellaswag_n=args.hellaswag_n,
        wikitext_tokens=args.wikitext_tokens,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        run_calibration_methods=args.calibration,
        calibration_samples=args.calibration_samples,
        calibration_seq_len=args.calibration_seq_len,
        run_bitsandbytes=args.bitsandbytes,
        save_results=not args.no_save,
    )