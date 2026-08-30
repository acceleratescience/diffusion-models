#!/usr/bin/env bash
set -euxo pipefail

# Install torch into venv (CPU build for codespaces)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu