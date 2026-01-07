# Setup Guide

## Prerequisites

Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Workflow

### Initial Setup (on any machine)

```bash
git clone <your-repo-url> ml-stuff
cd ml-stuff
uv sync
```

That's it. uv automatically:
- Creates a `.venv` in the project directory
- Installs CUDA-enabled PyTorch on Linux
- Installs MPS-enabled PyTorch on Mac
- Resolves all dependencies from `uv.lock`

### Activating the Environment

```bash
# Option 1: Use uv run (recommended - no activation needed)
uv run python train.py
uv run jupyter lab

# Option 2: Activate manually
source .venv/bin/activate
python train.py
```

### Adding New Dependencies

```bash
# Add a package
uv add <package-name>

# Add a dev dependency
uv add --dev <package-name>

# Then commit pyproject.toml and uv.lock
git add pyproject.toml uv.lock
git commit -m "Add <package-name>"
```

### Syncing After Git Pull

After pulling changes that modified `pyproject.toml` or `uv.lock`:
```bash
uv sync
```

### Updating Dependencies

```bash
# Update all packages to latest compatible versions
uv lock --upgrade
uv sync

# Update a specific package
uv lock --upgrade-package torch
uv sync
```

## Verify Installation

```bash
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

Expected output:
- **Mac M1**: CUDA: False, MPS: True
- **Linux GPU**: CUDA: True, MPS: False

## Troubleshooting

### Different CUDA Version Needed

If your GPU instance has a different CUDA version, edit `pyproject.toml`:
```toml
[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu118"  # Change to cu118, cu121, etc.
```

Then regenerate the lock:
```bash
uv lock --upgrade-package torch --upgrade-package torchvision
```

### Fresh Start

```bash
rm -rf .venv uv.lock
uv sync
```
