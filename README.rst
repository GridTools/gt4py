|tox| |format|

.. |tox| image:: https://github.com/GridTools/gt4py/actions/workflows/tox.yml/badge.svg?branch=functional
   :alt:
.. |format| image:: https://github.com/GridTools/gt4py/actions/workflows/qa.yml/badge.svg?branch=functional
   :alt:


GT4Py: GridTools for Python
===========================


Description
-----------

GT4Py is a Python library for generating high performance implementations
of stencil kernels from a high-level definition using regular Python
functions. GT4Py is part of the GridTools framework, a set of libraries
and utilities to develop performance portable applications in the area
of weather and climate.

**NOTE:** this is a development branch for a new and experimental version
of GT4Py working only with unstructured meshes and Python 3.10. The more
stable version of GT4Py for Cartesian meshes lives in the ``master`` branch.


Installation instructions
-------------------------

GT4Py can be installed as a regular Python package using *pip* (or any
other PEP517 frontend). As usual, we strongly recommended to create a
new virtual environment to work on this project.

Recommended installation using ``tox``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If `tox <https://tox.wiki/en/latest/#>`_ is already installed in your system (``tox`` is available in PyPI
and many other package managers), the easiest way to create
a virtual environment ready for development is::

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

Installation from scratch
~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, a development environment can be created from scratch::

    # Clone the repository
    git clone -b functional https://github.com/gridtools/gt4py.git
    cd gt4py

    # Create a (Python 3.10) virtual environment (usually at `.venv`)
    python3.10 -m venv .venv

    # Activate the virtual environment and make sure that 'wheel' is installed
    source .venv/bin/activate
    pip install --upgrade wheel

    # Install the required development tools and GT4Py (in editable mode)
    pip install -r requirements-dev.txt
    pip install -e .

    # Optionally, install atlas4py bindings directly from the repo
    # pip install git+https://github.com/GridTools/atlas4py#egg=atlas4py

    # Finally, check that everything works
    pytest -v


Development instructions
------------------------

After following the installation instructions above, an *editable*  installation
of the GT4Py package will be active in the virtual environment. In this mode,
code changes are directly visible since source files are imported directly in
the environment.

Code quality checks
~~~~~~~~~~~~~~~~~~~

The `pre-commit <https://pre-commit.com/>`_ framework is used to run several formatting and linting tools.
It should always be executed locally before opening a PR in the public repository.
``pre-commit`` can be installed as a *git hook* to automatically check the staged
changes before commiting::

    # Install pre-commit as a git hook and set up all the tools
    pre-commit install --install-hooks

Or it can be executed on demand from the command line::

    # Check only the staged changes
    pre-commit run

    # Check all the files in the repository
    pre-commit run -a

Testing
~~~~~~~

GT4Py testing uses the `pytest <https://pytest.org/>`_ framework which comes with an integrated ``pytest``
command tool to run tests easily::

    # Run all tests
    pytest -v

    # Run only tests in `path/to/test/folder`
    pytest -v path/to/test/folder

However, the recommended way to run the complete test suite is to use ``tox``
and select the appropriate test configuration::

    # List all the available test environments
    tox -a

    # Run test suite in a specific environment
    tox -e py310-base

``tox`` is configured to generate test coverage reports by default. An `HTML`
copy will be written in ``tests/_reports/coverage_html/`` at the end of the run.

