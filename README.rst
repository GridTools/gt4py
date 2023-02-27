|logo|

.. |logo| image:: https://raw.githubusercontent.com/GridTools/gt4py/gh-pages/v1.0/_static/logo.svg
   :alt:
.. _logo: https://GridTools.github.io/gt4py

|license| |slack|_

|test-cartesian| |test-next| |test-storage| |test-eve| |qa|

.. |license| image:: https://img.shields.io/github/license/GridTools/gt4py
   :alt:
.. |slack| image:: https://img.shields.io/badge/slack-join-orange?logo=slack&labelColor=3a3a3a
   :alt:
.. _slack: https://join.slack.com/t/gridtools/shared_invite/zt-1mceuj747-59swuowC3MKAuCFyNAnc1g

.. |test-cartesian| image:: https://github.com/GridTools/gt4py/actions/workflows/test-cartesian.yml/badge.svg?branch=main
   :alt:
.. |test-next| image:: https://github.com/GridTools/gt4py/actions/workflows/test-next.yml/badge.svg?branch=main
   :alt:
.. |test-storage| image:: https://github.com/GridTools/gt4py/actions/workflows/test-storage.yml/badge.svg?branch=main
   :alt:
.. |test-eve| image:: https://github.com/GridTools/gt4py/actions/workflows/test-eve.yml/badge.svg?branch=main
   :alt:
.. |qa| image:: https://github.com/GridTools/gt4py/actions/workflows/code-quality.yml/badge.svg?branch=main
   :alt:


Python library for generating high-performance implementations of stencil kernels for weather and climate modeling from a domain-specific language (DSL).


⚡️ Quick Start
---------------

GT4Py requires Python 3.8+ and uses the standard Python packaging method, so can be installed using `pip`.

It is recommended to install the package in a virtual environment. For example:

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate
    # For the CUDA backends add the '[cudaXXX]' optional dependency
    pip install gt4py[cuda11x]


📖 Description
--------------

GT4Py is a Python library for expressing computational motifs as found in weather and climate applications. These computations are expressed in a domain specific language (GTScript) which is translated to high-performance implementations for CPUs and GPUs.

The DSL expresses computations on a 3-dimensional Cartesian grid. The horizontal axes (``I``, ``J``) are always computed in parallel, while the vertical (``K``) can be iterated in sequential, forward or backward, order. Cartesian offsets are expressed relative to a center index.

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

For testing GT4Py with the ``numpy`` backend, all dependencies are included in the ``setup.cfg`` and are automatically installed. The performance backends also require

1. the `Boost <https://www.boost.org/>`__ library, a dependency of GridTools C++, which needs to be installed by the user.

Options
~~~~~~~

If GridTools or Boost are not found in the compiler's standard include path, or a custom version is desired, then a couple configuration environment variables will allow the compiler to use them:

- ``GT_INCLUDE_PATH``: Path to the GridTools installation.
- ``BOOST_ROOT``: Path to a boost installation.

Other commonly used environment variables are:

- ``CUDA_ARCH``: Set the compute capability of the NVIDIA GPU if it is not detected automatically by ``cupy``.
- ``CXX``: Set the C++ compiler.
- ``GT_CACHE_DIR_NAME``: Name of the compiler's cache directory (defaults to ``.gt_cache``)
- ``GT_CACHE_ROOT``: Path to the compiler cache (defaults to ``./``)

More options and details in `config.py <https://github.com/GridTools/gt4py/blob/main/src/gt4py/cartesian/config.py>`__.


Documentation
~~~~~~~~~~~~~

GT4Py uses Sphinx documentation. To build the documentation install the dependencies in ``requirements-dev.txt``

.. code-block:: bash

   pip install -r ./gt4py/requirements-dev.txt

and then build the docs with

.. code-block:: bash

   cd gt4py/docs
   make html  # run 'make help' for a list of targets


Development
~~~~~~~~~~~

For developing GT4Py we recommend to clone the repository and use an *editable* installation of GT4Py:

.. code-block:: bash

   git clone https://github.com/gridtools/gt4py.git
   pip install -e ./     # pip install -e ./[cudaXX] for GPU support
   pip install -r requirements-dev.txt
   pre-commit install-hooks

Dependencies for running tests locally and for linting and formatting source are listed in `requirements-dev.txt`.

There are notebooks in the `examples/ <https://github.com/GridTools/gt4py/tree/main/examples/cartesian>`__ folder that can be run using IPython notebooks on Jupyter.

.. code-block:: bash

   pip install jupyterlab matplotlib
   jupyter-lab

There are two options to run the unit and integration tests in ``tests/``:

1. Use tox: ``pip install tox && tox -r -e py39-all-cpu``.
2. Install the development requirements: ``pip install -r requirements-dev.txt``, then ``pytest`` can execute the tests directly.


⚖️ License
----------

GT4Py is licensed under the terms of the `GPLv3 <https://github.com/GridTools/gt4py/blob/main/LICENSE.txt>`__.
