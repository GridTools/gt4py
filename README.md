[![logo](https://raw.githubusercontent.com/GridTools/gt4py/gh-pages/v1.0/_static/logo.svg)](https://GridTools.github.io/gt4py)

![license](https://img.shields.io/github/license/GridTools/gt4py)
[![slack](https://img.shields.io/badge/slack-join-orange?logo=slack&labelColor=3a3a3a)](https://join.slack.com/t/gridtools/shared_invite/zt-1mceuj747-59swuowC3MKAuCFyNAnc1g)

[![Daily CI](https://github.com/GridTools/gt4py/actions/workflows/daily-ci.yml/badge.svg)](https://github.com/GridTools/gt4py/actions/workflows/daily-ci.yml)
![test-cartesian](https://github.com/GridTools/gt4py/actions/workflows/test-cartesian.yml/badge.svg?branch=main)
![test-next](https://github.com/GridTools/gt4py/actions/workflows/test-next.yml/badge.svg?branch=main)
![test-storage](https://github.com/GridTools/gt4py/actions/workflows/test-storage.yml/badge.svg?branch=main)
![test-eve](https://github.com/GridTools/gt4py/actions/workflows/test-eve.yml/badge.svg?branch=main)
![qa](https://github.com/GridTools/gt4py/actions/workflows/code-quality.yml/badge.svg?branch=main)

[![uv](https://img.shields.io/badge/-uv-261230.svg?logo=uv)](https://github.com/astral-sh/uv)
[![Nox](https://img.shields.io/badge/%F0%9F%A6%8A-Nox-D85E00.svg)](https://github.com/wntrblm/nox)

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

GT4Py can be installed as a regular Python package using [uv](https://docs.astral.sh/uv/), [pip](https://pip.pypa.io/en/stable/) or any other PEP-517 compatible frontend. We strongly recommended to use`uv` to create and manage virtual environments for your own projects.

## ⚙ Configuration

Other useful available environment variables are:

- `CUDA_ARCH`: Set the compute capability of the NVIDIA GPU if it is not detected automatically by `cupy`.
- `CXX`: Set the C++ compiler.
- `GT_CACHE_DIR_NAME`: Name of the compiler's cache directory (defaults to `.gt_cache`)
- `GT_CACHE_ROOT`: Path to the compiler cache (defaults to `./`)

More options and details are available in [`config.py`](https://github.com/GridTools/gt4py/blob/main/src/gt4py/cartesian/config.py).

## 🛠 Development Instructions

Follow the installation instructions below to initialize a development virtual environment containing an _editable_ installation of the GT4Py package. Make sure you read the [CONTRIBUTING.md](CONTRIBUTING.md) and [CODING_GUIDELINES.md](CODING_GUIDELINES.md) documents before you start working on the project.

### Development Environment Installation using `uv`

GT4Py uses the [`uv`](https://docs.astral.sh/uv/) project manager for the development workflow. `uv` is a versatile tool that consolidates functionality usually distributed across different applications into subcommands.

- The `uv pip` subcommand provides a _fast_ Python package manager, emulating [`pip`](https://pip.pypa.io/en/stable/).
- The `uv export | lock | sync` subcommands manage dependency versions in a manner similar to the [`pip-tools`](https://pip-tools.readthedocs.io/en/stable/) command suite.
- The `uv init | add | remove | build | publish | ...` subcommands facilitate project development workflows, akin to [`hatch`](https://hatch.pypa.io/latest/).
- The `uv tool` subcommand serves as a runner for Python applications in isolation, similar to [`pipx`](https://pipx.pypa.io/stable/).
- The `uv python` subcommands manage different Python installations and versions, much like [`pyenv`](https://github.com/pyenv/pyenv).

We require a reasonably recent version of `uv`, which can be installed in various ways (see its [installation instructions](https://docs.astral.sh/uv/getting-started/installation/)), with the recommended method being the standalone installer:

```bash
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

Once `uv` is installed in your system, it is enough to clone this repository and let `uv` handling the installation of the development environment.

```bash
# Clone the repository
git clone https://github.com/gridtools/gt4py.git
cd gt4py

# Let uv create the development environment at `.venv`.
# The `--extra all` option tells uv to install all the optional
# dependencies of gt4py, and thus it is not strictly necessary.
# Note that if no dependency groups are provided as an option,
# uv uses `--group dev` by default so the development dependencies
# are installed.
uv sync --extra all

# Finally, activate the virtual environment and start writing code!
source .venv/bin/activate
```

The newly created _venv_ is a standard Python virtual environment preconfigured with all necessary runtime and development dependencies. Additionally, the `gt4py` package is installed in editable mode, allowing for seamless development and testing. To install new packages in this environment, use the `uv pip` subcommand which emulates the `pip` interface and is generally much faster than the original `pip` tool (which is also available within the venv although its use is discouraged).

The `pyproject.toml` file contains both the definition of the `gt4py` Python distribution package and the settings of the development tools used in this project, most notably `uv`, `ruff`, and `mypy`. It also contains _dependency groups_ (see [PEP 735](https://peps.python.org/pep-0735/) for further reference) with the development requirements listed in different groups (`build`, `docs`, `lint`, `test`, `typing`, ...) and collected together in the general `dev` group, which gets installed by default by `uv` as mentioned above.

### Development Tasks (`dev-tasks.py`)

Recurrent development tasks like bumping versions of used development tools or required third party dependencies have been collected as different subcommands in the [`dev-tasks.py`](./dev-tasks.py) script. Read the tool help for a brief description of every task and always use this tool to update the versions and sync the version configuration accross different files (e.g. `pyproject.toml` and `.pre-commit-config.yaml`).

## 📖 Documentation

GT4Py uses the Sphinx tool for the documentation. To build browseable HTML documentation, install the required tools provided in the `docs` dependency group:

```bash
uv sync --group docs --extra all  # or --group dev
```

(Note that most likely these tools are already installed in your development environment, since the `docs` group is included in the `dev` group, which installed by default by `uv sync` if no dependency groups are specified.)

Once the requirements are already installed, then build the docs using:

```bash
cd gt4py/docs/user/cartesian
make html  # run 'make help' for a list of targets
```

## ⚖️ License

GT4Py is licensed under the terms of the [BSD-3-Clause](https://github.com/GridTools/gt4py/blob/main/LICENSE.txt).
