|tox| |format|

.. |tox| image:: https://github.com/GridTools/gt4py/workflows/Tox%20(CPU%20only)/badge.svg?branch=master
   :alt:
.. |format| image:: https://github.com/GridTools/gt4py/workflows/Formatting%20&%20compliance/badge.svg?branch=master
   :alt:


GT4Py: GridTools for Python
===========================

WARNING!
--------

GT4Py is still under development and is in an incomplete state. While
GT4Py has been proven to work for some basic applications already, we
are working actively on making GT4Py suitable for more use cases by
adding features, improving performance as well as reliability and adding
a thorough documentation. GT4Py will be changing for the time being. New
features will be implemented, APIs may change and dependencies will be
added.


Description
-----------

GT4Py is a Python library for generating high performance
implementations of stencil kernels from a high-level definition using
regular Python functions. GT4Py uses the `GridTools
Framework <https://github.com/GridTools/gridtools>`__ for a native
implementation of the kernel, but other code-generating backends are
also available.

The GridTools framework is a set of libraries and utilities to develop
performance portable applications in the area of weather and climate. To
achieve the goal of performance portability, the user-code is written in
a generic form which is then optimized for a given architecture at
compile-time. The core of GridTools is the stencil composition module
which implements a DSL embedded in C++ for stencils and stencil-like
patterns. Further, GridTools provides modules for halo exchanges,
boundary conditions, data management and bindings to C and Fortran.


Installation instructions
-------------------------

GT4Py contains a standard ``setup.py`` installation script that might be
installed as usual with *pip*. Additional commands are provided to
install and remove the GridTools C++ sources, which are not contained in
the package.

**IMPORTANT:** if the user provides a custom installation of GridTools
C++ sources, it should be compatible with the latest stable release of
GridTools, which is the version targeted by GT4Py. Note that a
compilation problem may also appear if there is a different GridTools
C++ version installed in a standard prefix (e.g. ``/usr/local``) which
could be included in user-provided or standard ``setuptools`` include
paths (for example if *Boost* is installed in the same prefix).

As usual in Python, we strongly recommended to create a new virtual
environment for any project:

::

    # Create a virtual environment using the 'venv' module
    python -m venv path_for_the_new_venv

    # Activate the virtual environment and make sure that 'wheel' is installed
    source path_for_the_new_venv/bin/activate
    pip install --upgrade wheel

Recommended installation for regular users
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are not planning to modify GT4Py sources, the easiest way to
complete the installation would be:

::

    # First, clone the repository
    git clone https://github.com/gridtools/gt4py.git

    # Then install the Python package directly from the local repository
    # For the CUDA backends add the '[cudaXX]' optional dependency
    # (XX = 90, 91, 92, 100 or 101 depending on CUDA version 9.0, 9.1, ...)
    pip install -e ./gt4py     # pip install -e ./gt4py[cudaXX]

Alternatively, if you do not need to build the documentation or look at
the examples, you could install GT4Py directly from the GitHub
repository:

::

    # Install the package directly from GitHub:
    # For the CUDA backends add the '[cudaXX]' optional dependency
    # (XX = 90, 91, 92, 100 or 101 depending on CUDA version 9.0, 9.1, ...)
    pip install git+https://github.com/gridtools/gt4py.git
    # pip install git+https://github.com/gridtools/gt4py.git#egg=gt4py[cudaXX]

In either case, you need to run a post-installation script to install
GridTools C++ sources.The new ``gtc:`` backends require GridTools v2,
while the old GT4Py ``gt:`` backends require GridTools v1:

::

    # Run the command to install GridTools v1.x C++ sources
    python -m gt4py.gt_src_manager install -m 1

    # Run the command to install GridTools v2.x C++ sources
    python -m gt4py.gt_src_manager install -m 2

Note that ``pip`` will not delete GridTools C++ sources when
uninstalling the package, so make sure you run the remove command in
advance:

::

    python -m gt4py.gt_src_manager remove  # -m 1 and/or -m 2
    pip uninstall gt4py

Recommended installation for developers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For GT4Py developers and advanced users, it is recommended to clone the
repository and use an *editable* installation of GT4Py:

::

    # First, clone the repository
    git clone https://github.com/gridtools/gt4py.git

    # Then install the Python package directly from the local repository
    # adding the '-e' flag to get an editable installation
    # For the CUDA backends add the '[cudaXX]' optional dependency
    # (XX = 90, 91, 92, 100 or 101 depending on CUDA version 9.0, 9.1, ...)
    pip install -e ./gt4py     # pip install -e ./gt4py[cudaXX]

    # Run the command to install GridTools C++ sources
    python -m gt4py.gt_src_manager install

    # Install the pre-commit checks
    pip install pre-commit
    # You need to have a python3.6 interpreter in your PATH for the following:
    pre-commit install-hooks  # in the repo directory
    # But you can develop using any version >= 3.6


Documentation
-------------

A proper documentation is in the works. Please refer to the jupyter
notebooks in the examples folder of this repository for examples of how
GT4Py can be used, or the *Quickstart* page of the documentation. To
build it, you need to clone the repository first (follow the
instructions in `Recommended installation for
developers <#recommended-installation-for-developers>`__) and then
install the additional development requirements with:

::

    pip install -r ./gt4py/requirements-dev.txt

and then build the docs with:

::

    cd gt4py/docs
    make html  # run 'make help' for a list of targets

Development roadmap
-------------------

A short overview of the new features and changes planned for the coming
weeks & months.

-  Integration with `Dawn <https://github.com/MeteoSwiss-APN/dawn>`__
   compiler
-  Update documentation (API reference, tutorial, notebooks and
   examples)
-  Missing features:

   +  Support for unstructured grids (GTScript extensions)
   +  Support for run-time values in interval definitions (run-time
      splitters)
   +  Support for different field layouts (storages masks)
   +  Support for OOP-based stencil definitions
   +  Support for boundary condition functions
   +  Support for proper function & stencil calls
