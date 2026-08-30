# Codespaces

For this workshop, you can work entirely within Github Codespaces.

!!! warning

    GitHub Codespaces only provides CPU resources, which means training and inference of the models will run much slower. If you have access to a GPU on your local machine, we recommend running the workshop locally instead and installing PyTorch manually by following the instructions in [Running Locally](./running-locally.md).

First create a new repository on Github, and then create a codespace from that repository. 

Navigate to the LLM workshop repo (click the GitHub symbol in the top right of this page). Switch to the `Handson` branch, and download the content as a zip file. Upload this to your new repo. Open the repo as a Codespace. There is no need to create a virtual environment, since you're already in a containerized environment anyway with a version of Python 3.12.

You don’t need to create or manage a virtual environment yourself. The devcontainer will automatically install `uv`, create and activate a `.venv`, and install all required dependencies when your Codespace is built.