GT4Py: GridTools for Python
===========================

WARNING!
--------

GT4Py is still under development and is in an incomplete state. While 
GT4Py has been proven to work for some basic applications already, we are
working actively on making GT4Py suitable for more use cases by adding 
Features, improving performance as well as reliability, adding a thorough
documentation. GT4Py will be changing for the time being. New features will
be implemented, APIs may change and dependencies will be added.


Description
-----------

GT4Py is a Python library for generating high performance
implementations of stencil kernels from a high-level definition using
regular Python functions. GT4Py uses the [GridTools
Framework](https://github.com/GridTools/gridtools) for a native
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

GT4Py contains a standard `setup.py` installation script that might be
installed as usual with *pip*. Two additional `setup.py` commands are
provided to install and remove the GridTools C++ sources, which are not
contained in the package. The complete installation procedure is thus:

    # First, clone the repository
    git clone https://github.com/gridtools/gt4py.git

    # Then install the Python package
    # add the '-e' flag if you want an editable installation
    # add the '[cudaXX]' extra if you want to install the GPU backends
    # (XX = 90, 91, 92, 100 or 101 depending on CUDA version 9.0, 9.1, ...)
    pip install gt4py/  # pip install gt4py/[cudaXX]

    # Finally run the command to install GridTools C++ sources
    python gt4py/setup.py install_gt_sources

Note that pip will not delete the GridTools C++ sources when
uninstalling the package, so make sure you run the remove command
before:

    python gt4py/setup.py remove_gt_sources
    pip uninstall gt4py

Documentation
-------------

A proper documentation is in the works. Please refer to the jupyter notebooks in the examples folder of this
repository for examples how GT4Py can be used.

Development roadmap
-------------------

A short overview of the new features and changes planned for the coming
weeks & months.


*  Integration with [Dawn](https://github.com/MeteoSwiss-APN/dawn)
    compiler
* Update documentation (API reference, tutorial, notebooks and examples)
* Missing features
    - Support for unstructured grids (GTScript extensions)
    - Support for run-time values in interval definitions (run-time splitters)
    - Support for different field layouts (storages masks)
    - Support for OOP-based stencil definitions
    - Support for boundary condition functions
    - Support for proper function & stencil calls


