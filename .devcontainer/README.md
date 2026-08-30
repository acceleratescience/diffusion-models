# Dev Container Configuration

This directory contains the development container configuration for the diffusion model workshop.

## What gets installed

When a user opens this repository in GitHub Codespaces, the following will be automatically set up:

### Base Environment
- Python 3.11 (Debian Bullseye)
- Git and GitHub CLI

### Python Dependencies
- **Workshop dependencies**: Installed from `pyproject.toml`
  - Diffusers, transformers, scikit-learn and other ML libraries
  - PyTorch CPU version (for lighter resource usage in Codespaces)
- **Documentation dependencies**: Also installed from `pyproject.toml`
  - MkDocs and related tools for building documentation

### VS Code Extensions
- Python development: Python, Pylance, Black formatter, Ruff linter
- Jupyter notebooks: Full Jupyter support with renderers and keybindings
- General development: JSON, YAML, and Markdown support

### Port Forwarding
- **Port 8888**: Jupyter notebook server

## Usage

Users simply need to:
1. Fork this repository
2. Open in GitHub Codespaces
3. Wait for the automatic setup to complete
4. Start working with the workshop materials

The environment will be ready to use with all dependencies installed and VS Code properly configured for Python and Jupyter development. 