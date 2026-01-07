import argparse
import glob
import os
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sns.set_theme(
    style="whitegrid",
    context="talk",
    palette="deep",
)


# -------------------------
# Helpers
# -------------------------

RUN_RE = re.compile(r"bert_(head|full|lora)_lr([0-9.eE+-]+)(?:_r=([0-9]+))?")


def find_event_file(run_dir: str) -> str | None:
    # Common TB event file glob
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not files:
        # fall back: any file containing tfevents
        files = sorted(glob.glob(os.path.join(run_dir, "*tfevents*")))
    return files[0] if files else None


def parse_run_name(run_dir_name: str):
    """
    Expects your naming convention like:
      bert_head_lr0.001
      bert_full_lr2e-05
      bert_lora_lr0.0002_r=8
    """
    m = RUN_RE.search(run_dir_name)
    if not m:
        return None
    run_type = m.group(1)
    lr = float(m.group(2))
    r = m.group(3)
    rank = int(r) if r is not None else None
    return run_type, lr, rank


def load_scalars(event_path: str) -> dict[str, list]:
    """
    Returns dict tag -> list of scalar events (each has .step, .value)
    """
    ea = EventAccumulator(
        event_path,
        size_guidance={
            "scalars": 0,  # load all
        },
    )
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    out = {}
    for t in tags:
        out[t] = ea.Scalars(t)
    return out


@dataclass
class RunSummary:
    run_dir: str
    run_name: str
    run_type: str
    lr: float
    rank: int | None
    trainable_params: float | None
    total_params: float | None
    best_val_acc: float | None
    best_train_acc: float | None
    best_val_loss: float | None
    best_val_step: int | None


def scalar_last_value(events) -> float | None:
    if not events:
        return None
    return float(events[-1].value)


def scalar_first_value(events) -> float | None:
    if not events:
        return None
    return float(events[0].value)


def scalar_best_max(events) -> (float | None, int | None):
    if not events:
        return None, None
    best = max(events, key=lambda e: e.value)
    return float(best.value), int(best.step)


def scalar_best_min(events) -> (float | None, int | None):
    if not events:
        return None, None
    best = min(events, key=lambda e: e.value)
    return float(best.value), int(best.step)


# -------------------------
# Main extraction
# -------------------------


def collect_runs(root_dir: str) -> list[RunSummary]:
    run_dirs = sorted([d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)])
    summaries: list[RunSummary] = []

    for rd in run_dirs:
        name = os.path.basename(rd)
        parsed = parse_run_name(name)
        if parsed is None:
            # skip dirs that don't match naming convention
            continue
        run_type, lr, rank = parsed

        event_path = find_event_file(rd)
        if event_path is None:
            print(f"[WARN] No event file found in {rd}")
            continue

        scalars = load_scalars(event_path)

        # Pull expected tags
        trainable_params = scalar_first_value(scalars.get("model/trainable_params", []))
        total_params = scalar_first_value(scalars.get("model/total_params", []))

        best_val_acc, best_val_step = scalar_best_max(scalars.get("Acc/val", []))
        best_train_acc, _ = scalar_best_max(scalars.get("Acc/train", []))

        # Optional: epoch val loss
        best_val_loss, _ = scalar_best_min(scalars.get("Loss/val_epoch", []))

        summaries.append(
            RunSummary(
                run_dir=rd,
                run_name=name,
                run_type=run_type,
                lr=lr,
                rank=rank,
                trainable_params=trainable_params,
                total_params=total_params,
                best_val_acc=best_val_acc,
                best_train_acc=best_train_acc,
                best_val_loss=best_val_loss,
                best_val_step=best_val_step,
            )
        )

    return summaries


# -------------------------
# Plotting
# -------------------------


def ensure_outdir(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)


def plot_best_val_acc_vs_rank(runs, out_dir):
    lora = [
        r
        for r in runs
        if r.run_type == "lora" and r.rank is not None and r.best_val_acc is not None
    ]
    lora.sort(key=lambda x: x.rank)

    if not lora:
        print("[WARN] No LoRA runs found for rank plot")
        return

    data = {
        "rank": [r.rank for r in lora],
        "best_val_acc": [r.best_val_acc for r in lora],
    }

    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(
        x=data["rank"],
        y=data["best_val_acc"],
        marker="o",
        linewidth=2,
    )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("Best validation accuracy")
    ax.set_title("Validation accuracy vs LoRA rank")

    plt.tight_layout()
    path = os.path.join(out_dir, "best_val_acc_vs_rank.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[OK] Wrote {path}")


def plot_best_val_acc_vs_trainable_params(runs, out_dir):
    rows = [r for r in runs if r.best_val_acc is not None and r.trainable_params is not None]

    if not rows:
        print("[WARN] No runs found for param plot")
        return

    data = {
        "trainable_params": [r.trainable_params for r in rows],
        "best_val_acc": [r.best_val_acc for r in rows],
        "run_type": [r.run_type for r in rows],
        "rank": [r.rank for r in rows],
    }

    plt.figure(figsize=(9, 6))
    ax = sns.scatterplot(
        x=data["trainable_params"],
        y=data["best_val_acc"],
        hue=data["run_type"],
        style=data["run_type"],
        s=120,
        alpha=0.9,
    )

    # annotate LoRA ranks
    for r in rows:
        if r.run_type == "lora" and r.rank is not None:
            ax.annotate(
                f"r={r.rank}",
                (r.trainable_params, r.best_val_acc),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=10,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Best validation accuracy")
    ax.set_title("Accuracy vs trainable parameters")

    plt.legend(title="Run type")
    plt.tight_layout()
    path = os.path.join(out_dir, "best_val_acc_vs_trainable_params.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[OK] Wrote {path}")


def plot_training_curves_selected(
    runs,
    root_dir,
    out_dir,
    include_head=True,
    include_full=True,
    lora_ranks=(1, 2, 4, 8, 16, 32),
):
    lookup = {}
    for r in runs:
        key = (r.run_type, r.rank if r.run_type == "lora" else None)
        lookup[key] = r

    selected = []
    if include_head and ("head", None) in lookup:
        selected.append(lookup[("head", None)])
    if include_full and ("full", None) in lookup:
        selected.append(lookup[("full", None)])

    for rr in lora_ranks:
        key = ("lora", rr)
        if key in lookup:
            selected.append(lookup[key])

    if not selected:
        print("[WARN] No runs selected for curve plot")
        return

    curve_rows = []

    for s in selected:
        event_path = find_event_file(s.run_dir)
        if event_path is None:
            continue

        scalars = load_scalars(event_path)
        ev = scalars.get("Acc/val", [])
        for e in ev:
            curve_rows.append(
                {
                    "step": e.step,
                    "val_acc": e.value,
                    "label": (s.run_type if s.run_type != "lora" else f"LoRA r={s.rank}"),
                }
            )

    if not curve_rows:
        print("[WARN] No scalar data for curve plot")
        return

    curve_rows = pd.DataFrame(curve_rows)

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=curve_rows,
        x="step",
        y="val_acc",
        hue="label",
        linewidth=2,
    )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Validation accuracy during training")
    ax.legend(title="Run")

    plt.tight_layout()
    path = os.path.join(out_dir, "val_acc_curves_selected.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[OK] Wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/workspace/ml-stuff/runs/lora",
        help="Parent directory containing run subdirectories.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/workspace/ml-stuff/LoRA/plots",
        help="Directory to write summary plots.",
    )
    parser.add_argument(
        "--no_curves", action="store_true", help="Disable curve plot and only write summary plots."
    )
    args = parser.parse_args()

    ensure_outdir(args.out_dir)

    runs = collect_runs(args.root_dir)
    if not runs:
        raise RuntimeError(
            f"No runnable TB runs found under {args.root_dir}. "
            f"Check naming convention and event files."
        )

    # Print a quick table
    print("\n=== Run summaries ===")
    for r in sorted(runs, key=lambda x: (x.run_type, x.rank if x.rank is not None else -1)):
        print(
            f"{r.run_name:35s}  "
            f"type={r.run_type:4s}  "
            f"rank={str(r.rank):>4s}  "
            f"trainable={int(r.trainable_params) if r.trainable_params is not None else None:>8}  "
            f"best_val_acc={r.best_val_acc if r.best_val_acc is not None else None}"
        )

    plot_best_val_acc_vs_rank(runs, args.out_dir)
    plot_best_val_acc_vs_trainable_params(runs, args.out_dir)

    if not args.no_curves:
        plot_training_curves_selected(runs, args.root_dir, args.out_dir)


if __name__ == "__main__":
    main()
