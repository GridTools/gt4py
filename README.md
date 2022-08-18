![tox-badge](https://github.com/GridTools/gt4py/actions/workflows/tox.yml/badge.svg?branch=functional) ![qa-badge](https://github.com/GridTools/gt4py/actions/workflows/qa.yml/badge.svg?branch=functional)


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

### Installation from Scratch

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
of the GT4Py package will be active in the virtual environment. In this mode, code changes are immediately visible since source files are imported directly by the Python interpreter, which is very convenient for development. Simple instructions to run development tools are given below, but make sure you read the [CONTRIBUTING.md](CONTRIBUTING.md) and [CODING_GUIDELINES.md](CODING_GUIDELINES.md) documents before you start with the actual development.

### Code Quality Checks

[pre-commit](https://pre-commit.com/) is used to run several linting and checking tools. It should always be executed locally before opening a pull request. `pre-commit` can be installed as a _git hook_ to automatically check the staged changes before commiting:

```bash
# Install pre-commit as a git hook and set up all the tools
pre-commit install --install-hooks
```

Or it can be executed on demand from the command line:

```bash
# Check only the staged changes
pre-commit run

# Check all the files in the repository
pre-commit run -a

# Run only some of the tools (e.g. flake8)
pre-commit run -a flake8
```

### Testing

GT4Py testing uses the [pytest](https://pytest.org/) framework which comes with an integrated ``pytest`` CLI tool to run tests in a very convenient way. For example:

```bash
# Run tests inside `path/to/test/folder`
pytest path/to/test/folder

# Run tests in parallel: -n NUM_OF_PROCS (or `auto`)
pytest -n auto tests/

# Run only tests that failed last time: --lf / --last-failed
pytest --lf tests/

# Run all the tests starting with the tests that failed last time:
# --ff / --failed-first
pytest --ff tests/

# Run tests with more informative output:
#   -v / --verbose          - increase verbosity
#   -l / --showlocalsflag   - show locals in tracebacks
#   -s                      - show tests outputs to stdout
pytest -v -l -s tests/
```

Check `pytest` documentation (`pytest --help`) for all the options to select and execute tests.

To run the complete test suite we recommended to also use `tox`:

```bash
# List all the available test environments
tox -a

# Run test suite in a specific environment
tox -e py310-base
```

`tox` is configured to generate test coverage reports by default. An `HTML`
copy will be written in `tests/_reports/coverage_html/` at the end of the run.
