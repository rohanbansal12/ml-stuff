#!/usr/bin/env python
"""Run ablation studies with different configurations.

Usage:
    # Run a single preset config
    python run_ablations.py --config baseline

    # Run multiple configs
    python run_ablations.py --config baseline resnet34 cosine_schedule

    # Run all preset configs
    python run_ablations.py --all

    # List available configs
    python run_ablations.py --list

    # Run from a JSON config file
    python run_ablations.py --from-file my_config.json
"""

import argparse
import subprocess
import sys
from config import CONFIGS, TrainConfig, get_config


def run_config(config: TrainConfig, dry_run: bool = False):
    """Run training with the given config."""
    args = ["python", "train_cifar.py"] + config.to_args()

    print(f"\n{'='*60}")
    print(f"Running: {config.run_name or config.model}")
    print(f"Command: {' '.join(args)}")
    print(f"{'='*60}\n")

    if dry_run:
        print("[DRY RUN] Skipping execution")
        return 0

    result = subprocess.run(args)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--config", nargs="+", help="Config name(s) to run")
    parser.add_argument("--all", action="store_true", help="Run all preset configs")
    parser.add_argument("--list", action="store_true", help="List available configs")
    parser.add_argument("--from-file", type=str, help="Load config from JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--seed", type=int, help="Override seed for all runs")
    args = parser.parse_args()

    if args.list:
        print("Available configs:")
        for name, config in CONFIGS.items():
            print(f"  {name}: {config.model}, epochs={config.epochs}, "
                  f"lr={config.lr}, schedule={config.lr_schedule}")
        return

    configs_to_run = []

    if args.from_file:
        config = TrainConfig.load(args.from_file)
        configs_to_run.append(config)

    elif args.all:
        for name in CONFIGS:
            config = get_config(name)
            config.run_name = name
            configs_to_run.append(config)

    elif args.config:
        for name in args.config:
            config = get_config(name)
            config.run_name = name
            configs_to_run.append(config)

    else:
        parser.print_help()
        return

    # Override seed if specified
    if args.seed is not None:
        for config in configs_to_run:
            config.seed = args.seed

    # Run all configs
    results = []
    for config in configs_to_run:
        returncode = run_config(config, dry_run=args.dry_run)
        results.append((config.run_name or config.model, returncode))

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for name, code in results:
        status = "OK" if code == 0 else f"FAILED ({code})"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
