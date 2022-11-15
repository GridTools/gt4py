# Gt4Py Workshop

## Install Tox
```bash
pip install tox
```

## Install GT4Py
```bash
git clone -b functional-workshop https://github.com/gridtools/gt4py.git

# Create the development environment
tox --devenv .venv -e gt4py-py310-base

# Activate the environment
source .venv/bin/activate
```

## Installation Instructions
```bash
pip install notebook

jupytext Workshop.md --to .ipynb
jupyter notebook
```

Open the notebook


