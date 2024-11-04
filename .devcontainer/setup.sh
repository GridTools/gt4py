#!/bin/sh

export PRE_COMMIT_HOME=/workspaces/.caches/pre-commit
ln -sfn /workspaces/gt4py/.devcontainer/.vscode /workspaces/gt4py/.vscode
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
pip install -r requirements-dev.txt
pip install -i https://test.pypi.org/simple/ atlas4py
pre-commit install --install-hooks
deactivate
