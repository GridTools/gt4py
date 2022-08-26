![tox-badge](https://github.com/GridTools/gt4py/actions/workflows/tox.yml/badge.svg?branch=functional) ![qa-badge](https://github.com/GridTools/gt4py/actions/workflows/qa.yml/badge.svg?branch=functional)![license-badge](https://img.shields.io/github/license/GridTools/gt4py)


# GT4Py: GridTools for Python

## Description

GT4Py is a Python library for generating high performance implementations
of stencil kernels from a high-level definition using regular Python
functions. GT4Py is part of the GridTools framework, a set of libraries
and utilities to develop performance portable applications in the area
of weather and climate.

**NOTE:** this is a development branch for a new and experimental version
of GT4Py working only with unstructured meshes and Python 3.10. The more
stable version of GT4Py for Cartesian meshes lives in the `master` branch.


## Installation Instructions

GT4Py can be installed as a regular Python package using `pip` (or any
other PEP-517 frontend). As usual, we strongly recommended to create a
new virtual environment to work on this project.

### Recommended Installation using `tox`

If [tox](https://tox.wiki/en/latest/) is already installed in your system (``tox`` is available in PyPI and many other package managers), the easiest way to create a virtual environment ready for development is:

```bash
# Clone the repository
git clone -b functional https://github.com/gridtools/gt4py.git
cd gt4py

# Create the development environment in any location (usually `.venv`)
# selecting one of the following templates:
#     py310-base  -> base environment
#     py310-atlas -> base environment + atlas4py bindings
tox --devenv .venv -e py310-base

# Finally, activate the environment and check that everything works
source .venv/bin/activate
pytest -v
```

### Manual Installation

Alternatively, a development environment can be created from scratch:

```bash
# Clone the repository
git clone -b functional https://github.com/gridtools/gt4py.git
cd gt4py

# Create a (Python 3.10) virtual environment (usually at `.venv`)
python3.10 -m venv .venv

# Activate the virtual environment and update basic packages
source .venv/bin/activate
pip install --upgrade wheel setuptools pip

# Install the required development tools
pip install -r requirements-dev.txt
# Install GT4Py project in editable mode
pip install -e .

# Optionally, install atlas4py bindings directly from the repo
# pip install git+https://github.com/GridTools/atlas4py#egg=atlas4py

# Finally, check that everything works
pytest -v
```

## Development Instructions

After following the installation instructions above, an _editable_  installation
of the GT4Py package will be active in the virtual environment and thus you will be ready to start your development contributions to GT4Py. Make sure you read the [CONTRIBUTING.md](CONTRIBUTING.md) and [CODING_GUIDELINES.md](CODING_GUIDELINES.md) documents before you start.
