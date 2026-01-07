#!/bin/bash
set -e

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv version: $(uv --version)"

# Sync dependencies
echo "Syncing dependencies..."
uv sync

echo "Done! Run 'uv run python <script.py>' or 'source .venv/bin/activate'"
