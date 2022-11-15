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
# Install jupyter notebook into the .venv
pip install notebook

#Convert markup file to.ipynb notebook
jupytext Workshop.md --to .ipynb

#open jupyter notebook application in your browser
jupyter notebook
```

Open the notebook


