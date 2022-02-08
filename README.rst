|tox| |format|

.. |tox| image:: https://github.com/GridTools/gt4py/workflows/Tox/badge.svg?branch=functional
   :alt:
.. |format| image:: https://github.com/GridTools/gt4py/workflows/Formatting%20&%20compliance/badge.svg?branch=functional
   :alt:


GT4Py: GridTools for Python
===========================


Description
-----------

GT4Py is a Python library for generating high performance
implementations of stencil kernels from a high-level definition using
regular Python functions. GT4Py is part of the GridTools framework, a
set of libraries and utilities to develop performance portable
applications in the area of weather and climate.

**NOTE:** this is a development branch for a new and experimental version
of GT4Py working only with unstructured meshes and Python 3.10. The more stable
version of GT4Py for cartesian meshes lives in the ``master`` branch.


Installation instructions
-------------------------

GT4Py can be installed as a regular Python package using *pip* (or any
other PEP517 frontend). As usual, we strongly recommended to create a
new virtual environment to work on this project.


Recommended installation using ``tox``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``tox`` is already installed in your system (``tox`` is available in PyPI
and many other package managers), the easiest way to create
a virtual environment ready for development is::

    # Clone the repository
    git clone https://github.com/gridtools/gt4py.git
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, a development environment can be created from scratch::

    # Clone the repository
    git clone https://github.com/gridtools/gt4py.git
    cd gt4py

    # Create a (Python 3.10) virtual environment (usually at `.venv`)
    python3.10 -m venv .venv

    # Activate the virtual environment and make sure that 'wheel' is installed
    source .venv/bin/activate
    pip install --upgrade wheel

    # Install the required development tools and GT4Py (in editable mode)
    pip install -r requirements-dev.txt
    pip install -e .

    # Optionally, install atlas4py bindigns directly from the repo
    # pip install git+https://github.com/GridTools/atlas4py#egg=atlas4py  

    # Finally, check that everything works
    pytest -v
