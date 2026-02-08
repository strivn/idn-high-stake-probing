#!/usr/bin/env bash
# setup_lambda.sh — Set up the bluedot-project on a Lambda Labs GPU instance.
#
# Usage:
#   git clone git@github.com:<your-user>/bluedot-project.git
#   cd bluedot-project
#   bash setup_lambda.sh
#
# After setup:
#   source .venv/bin/activate
#   jupyter lab --no-browser --port=8888
#   (then SSH tunnel: ssh -L 8888:localhost:8888 ubuntu@<IP>)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "=== Lambda Labs Setup ==="
echo "Project dir: $PROJECT_DIR"
echo ""

# ---------------------------------------------------------------
# 1. Install uv if not present
# ---------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

# ---------------------------------------------------------------
# 2. Install all dependencies (creates .venv automatically)
# ---------------------------------------------------------------
echo "Running uv sync..."
uv sync

# ---------------------------------------------------------------
# 3. Register Jupyter kernel
# ---------------------------------------------------------------
echo ""
echo "Registering Jupyter kernel..."
uv run python -m ipykernel install --user --name bluedot --display-name "Bluedot Project"

# ---------------------------------------------------------------
# 4. Create .env if it doesn't exist
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
    echo "IMPORTANT: Edit .env and add your HF_TOKEN and ANTHROPIC_API_KEY"
else
    echo ".env already exists."
fi

# ---------------------------------------------------------------
# 5. Verify GPU access
# ---------------------------------------------------------------
echo ""
echo "=== Environment Check ==="
uv run python -c "
import torch
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    vram  = props.total_mem / (1024**3)
    print(f'GPU:      {props.name}')
    print(f'VRAM:     {vram:.1f} GB')
    print(f'Quantize: {\"yes (8-bit)\" if vram < 20 else \"no (fp16)\"}')
"

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Edit .env with your tokens"
echo "  2. source .venv/bin/activate"
echo "  3. jupyter lab --no-browser --port=8888"
echo "  4. SSH tunnel: ssh -L 8888:localhost:8888 ubuntu@<INSTANCE_IP>"
echo "  5. Open http://localhost:8888 and run notebook 03"
