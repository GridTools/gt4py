GT4Py: GridTools for Python
===========================

Python library for generating high-performance implementations of
stencil kernels for weather and climate modeling from a
domain-specific language (DSL).

|tox| |format|

.. |tox| image:: https://github.com/GridTools/gt4py/workflows/Tox%20(CPU%20only)/badge.svg?event=schedule
   :alt:
.. |format| image:: https://github.com/GridTools/gt4py/workflows/Formatting%20&%20compliance/badge.svg?branch=master
   :alt:

⚡️ Quick Start
--------------

GT4Py requires Python 3.8+ and uses the standard Python packaging method,
so can be installed using `pip`.
It is not yet released on PyPI, so users have to point to the
git repository to install it.

It is recommended to install the package in a virtual environment.
For example:

.. code-block:: bash

    $ git clone https://github.com/GridTools/gt4py.git && cd gt4py
    $ python -m venv venv
    $ source venv/bin/activate
    $ pip install --upgrade setuptools wheel pip
    # For the CUDA backends add the '[cudaXXX]' optional dependency
    $ pip install -e ./[cuda117]

There are notebooks in the ``examples/`` directory that can be run using
IPython notebooks on Jupyter.

.. code-block:: bash

   $ pip install jupyterlab matplotlib
   $ jupyter-lab

There are two options to run the unit and integration tests in ``tests/``:

1. Use tox: ``pip install tox && tox -r -e py39-all-cpu``.
2. Install the development requirements: ``pip install -r requirements-dev.txt``,
   then ``pytest`` can execute the tests directly.


📖 Description
--------------

GT4Py is a Python library for expressing computational motifs as found in weather and climate applications.
These computations are expressed in a domain specific language (GTScript) which is translated to high-performance implementations for CPUs and GPUs.

The DSL expresses computations on a 3-dimensional Cartesian grid.
The horizontal axes (``I``, ``J``) are always computed in parallel, while the vertical (``K``) can be iterated in sequential, forward or backward, order. Cartesian offsets are expressed relative to a center index.

In addition, GT4Py provides functions to allocate arrays with memory layout suited for a particular backend.

The following backends are supported:

- ``numpy``: Pure-Python backend
- ``gt:cpu_ifirst``: GridTools C++ CPU backend using ``I``-first data ordering
- ``gt:cpu_kfirst``: GridTools C++ CPU backend using ``K``-first data ordering
- ``gt:gpu``: GridTools backend for CUDA
- ``cuda``: CUDA backend minimally using utilities from GridTools
- ``dace:cpu``: Dace code-generated CPU backend
- ``dace:gpu``: Dace code-generated GPU backend

🚜 Installation
---------------

For testing GT4Py with the ``numpy`` backend, all dependencies are included in the ``setup.cfg`` and are automatically
installed.
The performance backends require

1. `GridTools <https://github.com/GridTools/gridtools>`__ which is automatically downloaded on first use,
2. `Boost <https://www.boost.org/>`__ a dependency of GridTools,
   which needs to be installed by the user.

GridTools will automatically be downloaded when needed by the ``gt_src_manager.py`` module.
To manually install or uninstall, run:

::

    $ python -m gt4py.gt_src_manager {un}install

Options
~~~~~~~

If GridTools or Boost are not found in the compiler's standard include
path, or a custom version is desired, then a couple configuration
environment variables will allow the compiler to use them:

- ``GT_INCLUDE_PATH``: Path to the GridTools installation.
- ``BOOST_ROOT``: Path to a boost installation.

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
To build the documentation install the dependencies in ``requirements-dev.txt``

.. code-block:: bash

    $ pip install -r ./gt4py/requirements-dev.txt

and then build the docs with

.. code-block:: bash

    $ cd gt4py/docs
    $ make html  # run 'make help' for a list of targets


Development
~~~~~~~~~~~

For developing GT4Py we recommend to clone the repository
and use an *editable* installation of GT4Py:

.. code-block:: bash

   $ git clone https://github.com/gridtools/gt4py.git
   $ pip install -e ./     # pip install -e ./[cudaXX] for GPU support
   $ pip install -r requirements-dev.txt
   $ pre-commit install-hooks

Dependencies for running tests locally and for linting and formatting
source are listed in `requirements-dev.txt`.


⚠️ License
---------

GT4Py is licensed under the terms of the
`GPLv3 <https://github.com/GridTools/gt4py/blob/master/LICENSE.txt>`__.
