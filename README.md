[![logo](https://raw.githubusercontent.com/GridTools/gt4py/gh-pages/v1.0/_static/logo.svg)](https://GridTools.github.io/gt4py)

![license](https://img.shields.io/github/license/GridTools/gt4py)
[![slack](https://img.shields.io/badge/slack-join-orange?logo=slack&labelColor=3a3a3a)](https://join.slack.com/t/gridtools/shared_invite/zt-1mceuj747-59swuowC3MKAuCFyNAnc1g)

[![Daily CI](https://github.com/GridTools/gt4py/actions/workflows/daily-ci.yml/badge.svg)](https://github.com/GridTools/gt4py/actions/workflows/daily-ci.yml)
![test-cartesian](https://github.com/GridTools/gt4py/actions/workflows/test-cartesian.yml/badge.svg?branch=main)
![test-next](https://github.com/GridTools/gt4py/actions/workflows/test-next.yml/badge.svg?branch=main)
![test-storage](https://github.com/GridTools/gt4py/actions/workflows/test-storage.yml/badge.svg?branch=main)
![test-eve](https://github.com/GridTools/gt4py/actions/workflows/test-eve.yml/badge.svg?branch=main)
![qa](https://github.com/GridTools/gt4py/actions/workflows/code-quality.yml/badge.svg?branch=main)

# GT4Py: GridTools for Python

GT4Py is a Python library for generating high performance implementations of stencil kernels from a high-level definition using regular Python functions. GT4Py is part of the GridTools framework, a set of libraries and utilities to develop performance portable applications in the area of weather and climate modeling.

**NOTE:** The `gt4py.next` subpackage contains a new version of GT4Py which is not compatible with the current _stable_ version defined in `gt4py.cartesian`. The new version is still experimental.

## 📃 Description

GT4Py is a Python library for expressing computational motifs as found in weather and climate applications. These computations are expressed in a domain specific language (GTScript) which is translated to high-performance implementations for CPUs and GPUs.

The DSL expresses computations on a 3-dimensional Cartesian grid. The horizontal axes (`I`, `J`) are always computed in parallel, while the vertical (`K`) can be iterated in sequential, forward or backward, order. Cartesian offsets are expressed relative to a center index.

In addition, GT4Py provides functions to allocate arrays with memory layout suited for a particular backend.

The following backends are supported:

- `numpy`: Pure-Python backend
- `gt:cpu_ifirst`: GridTools C++ CPU backend using `I`-first data ordering
- `gt:cpu_kfirst`: GridTools C++ CPU backend using `K`-first data ordering
- `gt:gpu`: GridTools backend for CUDA
- `cuda`: CUDA backend minimally using utilities from GridTools
- `dace:cpu`: Dace code-generated CPU backend
- `dace:gpu`: Dace code-generated GPU backend

## 🚜 Installation

GT4Py can be installed as a regular Python package using `pip` (or any other PEP-517 frontend). As usual, we strongly recommended to create a new virtual environment to work on this project.

The performance backends also require the [Boost](https://www.boost.org) library, a dependency of [GridTools C++](https://github.com/GridTools/gridtools), which needs to be installed by the user.

## ⚙ Configuration

If GridTools or Boost are not found in the compiler's standard include path, or a custom version is desired, then a couple configuration environment variables will allow the compiler to use them:

- `GT_INCLUDE_PATH`: Path to the GridTools installation.
- `BOOST_ROOT`: Path to a boost installation.

Other commonly used environment variables are:

- `CUDA_ARCH`: Set the compute capability of the NVIDIA GPU if it is not detected automatically by `cupy`.
- `CXX`: Set the C++ compiler.
- `GT_CACHE_DIR_NAME`: Name of the compiler's cache directory (defaults to `.gt_cache`)
- `GT_CACHE_ROOT`: Path to the compiler cache (defaults to `./`)

More options and details are available in [`config.py`](https://github.com/GridTools/gt4py/blob/main/src/gt4py/cartesian/config.py).

## 📖 Documentation

GT4Py uses Sphinx documentation. To build the documentation install the dependencies in `requirements-dev.txt`

```bash
pip install -r ./gt4py/requirements-dev.txt
```

and then build the docs with

```bash
cd gt4py/docs/user/cartesian
make html  # run 'make help' for a list of targets
```

## 🛠 Development Instructions

Follow the installation instructions below to initialize a development virtual environment containing an _editable_ installation of the GT4Py package. Make sure you read the [CONTRIBUTING.md](CONTRIBUTING.md) and [CODING_GUIDELINES.md](CODING_GUIDELINES.md) documents before you start working on the project.

### Recommended Installation using `tox`

If [tox](https://tox.wiki/en/latest/) is already installed in your system (`tox` is available in PyPI and many other package managers), the easiest way to create a virtual environment ready for development is:

```bash
# Clone the repository
git clone https://github.com/gridtools/gt4py.git
cd gt4py

# Create the development environment in any location (usually `.venv`)
# selecting one of the following templates:
#     dev-py310       -> base environment
#     dev-py310-atlas -> base environment + atlas4py bindings
tox devenv -e dev-py310 .venv

# Finally, activate the environment
source .venv/bin/activate
```

### Manual Installation

Alternatively, a development environment can be created from scratch installing the frozen dependencies packages :

```bash
# Clone the repository
git clone https://github.com/gridtools/gt4py.git
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
```

## ⚖️ License

GT4Py is licensed under the terms of the [BSD-3-Clause](https://github.com/GridTools/gt4py/blob/main/LICENSE.txt).
