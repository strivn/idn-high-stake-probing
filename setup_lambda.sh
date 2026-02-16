#!/usr/bin/env bash
# setup_lambda.sh — Quick setup for bluedot-project on Lambda Labs GPU instance.
#
# Assumes bluedot-project is already on disk (e.g., /home/ubuntu/default-filesystem/bluedot-project).
# Usage:
#   cd /home/ubuntu/default-filesystem/bluedot-project
#   bash setup_lambda.sh
#
# Or with explicit path:
#   bash setup_lambda.sh /home/ubuntu/default-filesystem/bluedot-project

set -euo pipefail

# ---------------------------------------------------------------
# Find project directory
# ---------------------------------------------------------------
PROJECT_DIR="${1:-.}"
if [ "$PROJECT_DIR" = "." ]; then
    PROJECT_DIR="$(pwd)"
fi

# If not in bluedot-project, try fallback path
if [ ! -f "$PROJECT_DIR/pyproject.toml" ]; then
    if [ -f "/home/ubuntu/default-filesystem/bluedot-project/pyproject.toml" ]; then
        PROJECT_DIR="/home/ubuntu/default-filesystem/bluedot-project"
    else
        echo "ERROR: bluedot-project not found. Usage: bash setup_lambda.sh [/path/to/bluedot-project]"
        exit 1
    fi
fi

cd "$PROJECT_DIR"
echo "=== Lambda Labs Setup ==="
echo "Project dir: $PROJECT_DIR"
echo ""

# ---------------------------------------------------------------
# 1. Install uv (via pip, faster than curl)
# ---------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    pip install uv --quiet
else
    echo "uv already installed"
fi

# ---------------------------------------------------------------
# 2. Install gh (GitHub CLI)
# ---------------------------------------------------------------
if ! command -v gh &> /dev/null; then
    echo "Installing GitHub CLI (gh)..."
    (type -p wget >/dev/null || (sudo apt-get update && sudo apt-get install wget -y)) && \
    sudo mkdir -p -m 755 /etc/apt/keyrings && \
    wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.sources.list > /dev/null && \
    sudo apt-get update && sudo apt-get install gh -y --quiet
else
    echo "gh already installed"
fi

# ---------------------------------------------------------------
# 3. Install all dependencies (creates .venv automatically)
# ---------------------------------------------------------------
# Avoid failed hardlink attempts on cross-filesystem setups (NFS home dir)
export UV_LINK_MODE=copy

echo "Pre-installing Python 3.11 interpreter..."
uv python install 3.11

echo "Running uv sync..."
uv sync

# ---------------------------------------------------------------
# 4. Register Jupyter kernel
# ---------------------------------------------------------------
echo ""
echo "Registering Jupyter kernel..."
uv run python -m ipykernel install --user --name bluedot --display-name "Bluedot Project"

# ---------------------------------------------------------------
# 5. Create .env if it doesn't exist
# ---------------------------------------------------------------
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env template..."
    cat > .env << 'EOF'
# HuggingFace token (required for Llama-3.1-8B-Instruct)
HF_TOKEN=

# Anthropic API key (required for translation in notebook 04)
ANTHROPIC_API_KEY=
EOF
    echo "⚠️  IMPORTANT: Edit .env and add your HF_TOKEN and ANTHROPIC_API_KEY"
else
    echo ".env already exists"
fi

# ---------------------------------------------------------------
# 6. Verify GPU access
# ---------------------------------------------------------------
echo ""
echo "=== Environment Check ==="
uv run python -c "
import torch
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    vram  = props.total_memory / (1024**3)
    print(f'GPU:      {props.name}')
    print(f'VRAM:     {vram:.1f} GB')
    print(f'Quantize: {\"yes (8-bit)\" if vram < 20 else \"no (fp16)\"}')"

# ---------------------------------------------------------------
# 7. Install VS Code extensions (for VS Code Server / Remote SSH)
# ---------------------------------------------------------------
if command -v code &> /dev/null; then
    echo ""
    echo "Installing VS Code extensions..."
    code --install-extension ms-python.python          --force
    code --install-extension ms-toolsai.jupyter        --force
    code --install-extension ms-toolsai.vscode-jupyter-cell-tags --force
else
    echo ""
    echo "VS Code CLI not found, skipping extension install"
fi

echo ""
echo "=== Setup Complete ==="
echo "Next: source .venv/bin/activate && jupyter lab --no-browser --port=8888"
