#!/bin/sh

ln -sfn /workspaces/gt4py/.devcontainer/.vscode /workspaces/gt4py/.vscode
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements-dev.txt
uv pip install -e .
uv pip install -i https://test.pypi.org/simple/ atlas4py
pre-commit install --install-hooks
deactivate
