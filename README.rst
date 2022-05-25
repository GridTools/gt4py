GT4Py: GridTools for Python
===========================

Python library for generating high-performance implementations of
stencil kernels from a domain-specific language (DSL).

|tox| |format|

.. |tox| image:: https://github.com/GridTools/gt4py/workflows/Tox%20(CPU%20only)/badge.svg?event=schedule
   :alt:
.. |format| image:: https://github.com/GridTools/gt4py/workflows/Formatting%20&%20compliance/badge.svg?branch=master
   :alt:

⚡️ Quick Start
--------------

GT4Py uses the standard Python packaging method, and can be installed
using `pip`.
However, it is not yet released on PyPI, so users have to point to the
git repository to install it.

As always, it is recommended to install the package in a virtual
environment.

.. code-block:: bash

    $ git clone https://github.com/GridTools/gt4py.git && cd gt4py
    $ python -m venv venv
    $ source venv/bin/activate
    $ pip install --upgrade setuptools wheel pip
    # For the CUDA backends add the '[cudaXX]' optional dependency
    $ pip install -e ./[cuda11]

There are notebooks in the ``examples/`` directory that can be run using IPython
notebooks on Jupyter.

.. code-block:: bash

   $ pip install jupyterlab
   $ jupyter-lab

In order to run the unit and integration tests in ``tests/``, there are
a couple options:

1. Use tox: ``pip install tox && tox -r -e py38-all-cpu``.
2. Install the development requirements: ``pip install -r requirements-dev.txt``,
   then ``pytest`` can execute the tests directly.


📖 Description
--------------

GT4Py is a Python library for expressing finite-difference stencils
related to weather and climate modeling using a high-level
DSL using Python functions.
These functions are compiled by the framework into high-performance
implementations for CPUs and GPUs.

The DSL expresses the stencils using the parallel model from the
`GridTools C++ Framework <https://github.com/GridTools/gridtools>`__,
and uses it, as well as other backends, for optimized code generation.
In this Cartesian parallel model there are always three dimensions,
and the vertical (``K``) is treated separately from the horizontal axes
(``I``, ``J``), which are iterated over in parallel.

Stencil expressions are Cartesian offsets from a center index, as it
would be written algorithmically.
List of these stencil *statements* form *computations*, and can be specialized
in the vertical index to account for boundaries, or accumulate fields.


🚜 Installation
---------------

The base version of GT4Py does not have dependencies other than the
Python packages included in the ``setup.cfg`` which are automatically
installed.
The GridTools backends however require

1. `GridTools <https://github.com/GridTools/gridtools>`__ C++ sources.
2. `Boost <https://www.boost.org/>`__ a dependency of GridTools,
   which needs to be installed by the user.

The correct version of GridTools is downloaded automatically when needed.

Options
~~~~~~~

If GridTools or Boost are not found in the compiler's standard include
path, or a custom version is desired, then a couple configuration
environment variables will allow the compiler to use them:

- ``GT2_INCLUDE_PATH``: Path to the GridTools v2 (default) installation.
- ``BOOST_ROOT``: Path to the boost headers.

Other commonly used environment variables are:

- ``CUDA_ARCH``: Set the compute capability of the NVIDIA GPU if it is not
  detected automatically by ``cupy``.
- ``CXX``: Set the C++ compiler.
- ``GT_CACHE_DIR_NAME``: Name of the compiler's cache directory
  (defaults to ``.gt_cache``)
- ``GT_CACHE_ROOT``: Path to the compiler cache (defaults to ``./``)

More options and details in
`config.py <https://github.com/GridTools/gt4py/blob/master/src/gt4py/config.py>`__.

Documentation
~~~~~~~~~~~~~

GT4Py uses Sphinx documentation.
However, it is not published yet, so it needs to be built by the user.
To do that, first install the dependencies in ``requirements-dev.txt``:

.. code-block:: bash

    $ pip install -r ./gt4py/requirements-dev.txt

and then build the docs with:

.. code-block:: bash

    $ cd gt4py/docs
    $ make html  # run 'make help' for a list of targets

Development
~~~~~~~~~~~

For GT4Py developers and advanced users, it is recommended to clone the
repository and use an *editable* installation of GT4Py:

.. code-block:: bash

   $ git clone https://github.com/gridtools/gt4py.git
   $ pip install -e ./     # pip install -e ./[cudaXX] for GPU support
   $ pip install -r requirements-dev.txt
   $ pre-commit install-hooks

Dependencies for running tests locally and for linting and formatting
code are listed in `requirements.dev.txt`, so these should be installedj


⚠️ License
---------

GT4Py is licensed under the terms of the
`GPLv3 <https://github.com/GridTools/gt4py/blob/master/LICENSE.txt>`__.
Of particular note is that this requires any code that imports `gt4py` to
carry a GPL license.

