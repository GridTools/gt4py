# Hands on Session for GT4Py Workshop

## Install Tox into your current Python enviornment
```bash
pip install tox
```

## Install GT4Py
```bash

# Clone the repository
git clone -b functional-workshop https://github.com/gridtools/gt4py.git
cd gt4py

# Create a development environment
tox --devenv .venv -e gt4py-py310-base

# Activate that environment
source .venv/bin/activate
```

## Installation Instructions
```bash
# Install jupyter notebook into the .venv environment
pip install notebook

# Open the jupyter notebook application in your browser
jupyter notebook 
```

Open Workshop.md using the GUI


