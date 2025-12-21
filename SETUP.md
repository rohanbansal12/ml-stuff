# Quick Setup for Cloud GPU Instances

## Installation with uv (Recommended)

### First-time setup on a new instance:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo
git clone <your-repo-url> ml-stuff
cd ml-stuff

# Install all dependencies
uv pip install -e .
```

### For GPU instances with specific CUDA versions:

```bash
# CUDA 12.1 (most common for modern GPUs)
uv pip install -e . --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (for older setups)
uv pip install -e . --index-url https://download.pytorch.org/whl/cu118
```

### Install dev dependencies (optional):

```bash
uv pip install -e ".[dev]"
```

## Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Alternative: Using uv sync (for locked dependencies)

If you want reproducible builds across instances:

```bash
# First time: create a lock file
uv lock

# On new instances: install from lock
uv sync
```

This ensures identical package versions across all your GPU instances.
