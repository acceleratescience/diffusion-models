# Running Locally

You can run the notebooks locally on your own machine by following the steps below.

We recommend using `uv` for dependencies as it is much faster and automatically manages a `.venv` for you.

However, you can also install dependencies with plain `pip` if you prefer.

>
> PyTorch (`torch`) is **not included** in the installation instructions below. PyTorch wheels differ depending on your system and can be very large if installed incorrectly. 
> 
> Please install the version suitable for your setup by following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## 1. Cloning the repository
Clone the repository to your machine:
```bash
git clone https://github.com/acceleratescience/diffusion-models
cd diffusion-models
```
Alternatively, if you're using an IDE such as VS Code or PyCharm, you can paste the repository URL directly into the IDE's respective clone repository UI.

## 2. Installing dependencies with `uv` (preferred)
### Windows (Powershell)
Install uv if needed:
```bash
irm https://astral.sh/uv/install.ps1 | iex
```
Create the virtual environment and install dependencies:
```bash
uv sync
```
Finally, activate the environment:
```bash
.\.venv\Scripts\activate
```
### macOS / Linux
Install uv if needed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Create the virtual environment and install dependencies:
```bash
uv sync
```
Finally, activate the environment:
```bash
source .venv/bin/activate
```

## 3. Installing dependencies with `pip` (alternative)
### Windows (Powershell)
Create and activate an empty virtual environment:
```bash
python3 -m diffusion-models .venv
.\.venv\Scripts\activate
```
Finally, install dependencies to virtual environment:
```bash
pip install .
```
### macOS / Linux
Create and activate an empty virtual environment:
```bash
python3 -m diffusion-models .venv
source .venv\bin\activate
```
Finally, install dependencies to virtual environment:
```bash
pip install .
```