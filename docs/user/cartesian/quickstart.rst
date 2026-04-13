=================
Quick Start Guide
=================

This document will guide you through the basic steps to get started with GT4Py.

------------
Installation
------------

GT4Py contains a ``setup.py`` installation script and can be installed as usual with `pip`.
Additional commands are provided to install and remove the GridTools C++ sources, which are not contained in the package.

We strongly recommended to create a virtual environment for any new project:

.. code:: bash

    python -m venv path_for_the_new_venv
    source path_for_the_new_venv/bin/activate
    pip install --upgrade pip wheel setuptools


Then clone the GT4Py repository and install the local copy or install it directly from PyPI: `pip install gt4py`
For use with NVIDIA GPUs, add the `[cudaXX]` optional dependency, where `XX` takes the values
`11`, `12`, ... depending on the CUDA version installed in your system (CUDA version 11.x, 12.x, ...).

.. code:: bash

    git clone https://github.com/GridTools/gt4py.git
    pip install ./gt4py
    # pip install ./gt4py[cuda12]

Or

.. code:: bash

    pip install gt4py
    # pip install gt4py[cuda12]


------------
Introduction
------------

In GT4Py, grid computations such as stencils are defined in a domain-specific language (DSL) called GTScript.
GTScript is syntactically a subset of Python, but has different semantics.
Computations defined in this DSL are translated by the GT4Py toolchain into code in Python based on
`NumPy <http://www.numpy.org/>`_ or C++/CUDA based on the `GridTools <http://gridtools.github.io/>`_ library.
To be able to achieve full performance with GridTools, data has to adhere to certain layout requirements, which
are taken care of by storing the data in GT4Py storage containers.

A Simple Example
----------------
Suppose we want to write a stencil computing a parameterized linear combination of multiple 3D fields.
The stencil has two parameters to change the relative weights (``alpha`` and ``weight``) of the input
fields (``field_a``, ``field_b``, and ``field_c``) to be combined:

.. code:: python

    import numpy as np

    # assuming fields of size nx, ny, nz
    result = np.zeros((nx, ny, nz))

    def stencil_example(field_a, field_b, field_c, result, alpha, weight):
        nx, ny, nz = result.shape
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    result[i, j, k] = field_a[i, j, k] - (1 - alpha) * (
                            field_b[i, j, k] - weight * field_c[i, j, k]
                    )

To express this calculation using GTScript, we create a function and use the DSL syntax to represent the loop
over 3 dimensions:

.. code:: python

    import numpy as np

    import gt4py.cartesian.gtscript as gtscript

    backend = "numpy"

    @gtscript.stencil(backend=backend)
    def stencil_example(
        field_a: gtscript.Field[np.float64],
        field_b: gtscript.Field[np.float64],
        field_c: gtscript.Field[np.float64],
        result: gtscript.Field[np.float64],
        *,
        alpha: np.float64,
        weight: np.float64 = 2.0,
    ):
        with computation(PARALLEL), interval(...):
            result = field_a[0, 0, 0] - (1 - alpha) * (
                field_b[0, 0, 0] - weight * field_c[0, 0, 0]
            )

    assert callable(stencil_example) is True
    print(type(stencil_example), "\n", stencil_example)

    # --> <class '_GT_.__main__.stencil_example.m_stencil_example__numpy'...
    # --> <StencilObject: __main__.stencil_example> [backend="numpy"] ...


This definition basically expresses the operations (or *kernel*) performed at every point of the computation domain to generate the output values.
The indices inside the brackets are interpreted as offsets relative to the current point in the iteration, and not as absolute positions in the data fields.
For an explanation of the line ``with computation(PARALLEL), interval(...):``, please refer to the section :ref:`Computations and Intervals`.

.. note::
    While not required, it is recommended to specify *fields* as standard arguments and run-time *parameters* as
    *keyword-only* arguments.

Once the stencil kernel has been defined, we use GT4Py to generate an *implementation* of this high-level definition as
a callable object that we can use to apply the operations to data. This can be done by just decorating the definition
function with the ``stencil`` decorator provided by GT4Py.

The ``stencil`` decorator generates code in Python or C++ depending on the ``backend`` specified by name.
Currently, the following backends are available:

* ``"numpy"``: a vectorized Python backend
* ``"gt:cpu_kfirst"``: a backend based on GridTools code performance-optimized for x86 architecture
* ``"gt:cpu_ifirst"``: a GridTools backend targeting many core architectures
* ``"gt:gpu"``: a GridTools backend targeting GPUs
* ``"dace:cpu"``: Dace code-generated CPU backend
* ``"dace:cpu_kfirst"``: Dace code-generated CPU backend using `K`-first data ordering
* ``"dace:gpu"``: Dace code-generated GPU backend
* ``"debug"``: A pure python backend used for prototyping new features

The decorator further replaces the stencil definition function (here ``stencil_example``) by a callable object that
can be used as a function to call the generated code which modifies the passed data in place.

Instead of using the ``stencil`` decorator, it is also possible to compile the stencil using a
regular function call receiving the definition function:

.. code:: python

    import gt4py.cartesian.gtscript as gtscript

    def stencil_example(
        field_a: gtscript.Field[np.float64],
        field_b: gtscript.Field[np.float64],
        field_c: gtscript.Field[np.float64],
        result: gtscript.Field[np.float64],
        *,
        alpha: np.float64,
        weight: np.float64 = 2.0,
    ):
        with computation(PARALLEL), interval(...):
            result = field_a[0, 0, 0] - (1. - alpha) * (
                field_b[0, 0, 0] - weight * field_c[0, 0, 0]
            )

    stencil_example_numpy = gtscript.stencil(backend="numpy", definition=stencil_example)

    another_example_gt = gtscript.stencil(backend="gt:cpu_ifirst", definition=stencil_example)

The generated code is written to and compiled in a local '.gt_cache' folder. Subsequent
invocations will check whether a recent version of the stencil already exists in the cache.

----------
Allocation
----------

Since some backends require data to be in a certain layout in memory, GT4Py provides special `NumPy`-like
allocators. They work like the familiar set of routines used in `NumPy` for allocation: ``ones``, ``zeros``,
``full`` and ``empty``. There is also ``from_array`` that initializes the array to a provided array value.
The result of these routines is either a ``numpy.ndarray`` (for CPU backends) or a ``cupy.ndarray``
(for GPU backends).

.. code:: python

    import gt4py.storage as gt_storage

    backend= "numpy"

    field_a = gt_storage.from_array(
        np.random.randn(10, 10, 10),
        np.float64,
        backend=backend,
        aligned_index=(0, 0, 0),
    )
    field_b = gt_storage.ones(
        (10, 10, 10), np.float64, backend=backend, aligned_index=(0, 0, 0)
    )
    field_c = gt_storage.zeros(
        (10, 10, 10), np.float64, backend=backend, aligned_index=(0, 0, 0)
    )
    result = gt_storage.empty(
        (10, 10, 10), np.float64, backend=backend, aligned_index=(0, 0, 0)
    )

    stencil_example(field_a, field_b, field_c, result, alpha=0.5)


The ``aligned_index`` specifies that the array is to be allocated such that memory address of the point specified in
``aligned_index`` is `aligned` to a backend-dependent value. For optimal performance, you set the ``algined_index`` to
a point which is the lower-left corner of the iteration domain most frequently used for this field.

----------------
Array Interfaces
----------------

When passing buffers to stencils, they can be in any form that is compatible with ``np.asarray`` or ``cp.asarray``,
respectively. Some meta information can be provided to describe the correspondence between array dimensions and
their semantic meaning (e.g. IJK) as well as to specify the correspondence. Also, an index can be designated as the
`origin` of the array to denote the start of the index range considered to be the `iteration domain`. Specifically, the
behavior is as follows:

#. Dimensions can be denoted by adding a ``__gt_dims__`` attribute to the buffer object. It should be a tuple of strings
   where currently valid dimensions are ``"I", "J", "K"`` as well as string representations of integers to represent
   data dimensions, i.e. the dimensions of vector, matrices or higher tensors per grid point. If ``__gt_dims__`` is not
   present, the dimensions specified in the ``Field`` annotation of functions serves as a default.
#. The origin can be specified with the ``__gt_origin__`` attribute, which is a tuple of ``int`` s. If when calling the
   stencil, no other `origin` is specified, this value is where the `iteration domain` begins, i.e. the grid point with
   the lowest index where a value is written. The explicit ``origin`` keyword when calling a stencil takes priority over
   this.


.. _Computations and Intervals:

--------------------------
Computations and Intervals
--------------------------

`Computations` and `interval` determine the iteration space and schedule in the vertical direction.
The `computation` context determines in which order the vertical
dimension is iterated over. ``FORWARD`` specifies an iteration from low to high index, while ``BACKWARD`` is an
iteration from high to low index. For contexts declared ``PARALLEL``, no order is assumed and only computations
for which the result is the same irrespective of iteration order are allowed.

`Intervals` declare the range of indices for which the statements
of the respective context are applied. For example, ``interval(0,1)`` declares that the following context is executed for indices
in [0,1), i.e. only for `K=0`. The ``interval(1, None)`` represents indices in [1,∞), ``interval(0, -1)`` all indices
except the last.

For example the Thomas algorithm to solve a linear, tridiagonal system of equations can be implemented using a
forward and a backward loop with specialized computations at the beginning of each iteration:

.. code:: python

    import gt4py.cartesian.gtscript as gtscript

    @gtscript.stencil
    def tridiagonal_solver(
        inf: gtscript.Field[np.float64],
        diag: gtscript.Field[np.float64],
        sup: gtscript.Field[np.float64],
        rhs: gtscript.Field[np.float64],
        out: gtscript.Field[np.float64],
    ):
        with computation(FORWARD):
            with interval(0, 1):
                sup = sup / diag
                rhs = rhs / diag
            with interval(1, None):
                sup = sup / (diag - sup[0, 0, -1] * inf)
                rhs = (rhs - inf * rhs[0, 0, -1]) / (diag - sup[0, 0, -1] * inf)

        with computation(BACKWARD):
            with interval(0, -1):
                out = rhs - sup * out[0, 0, 1]
            with interval(-1, None):
                out = rhs


However, ``PARALLEL`` also differs from ``FORWARD`` and ``BACKWARD`` in another way:
For ``PARALLEL``, we can assume that each statement
(i.e. each assignment within the context) is applied to the full vertical domain, before the next one starts.
If ``FORWARD`` or ``BACKWARD`` is specified
however, all statements are applied to each slice with the same ``K``, one after each other, before moving to ``K+1``
or ``K-1``, respectively.


---------
Functions
---------

Functions allow to reuse code elements and to structure your code. They are decorated with ``@gtscript.function``.

.. code:: python

    @gtscript.function
    def ddx(v, h = 0.1):
        v2 = v[-1, 0, 0] + v[1, 0, 0] - 2. * v[0, 0, 0]
        return v2 / (h * h)

    @gtscript.function
    def ddy(v, h = 0.1):
        v2 = v[0, -1, 0] + v[0, 1, 0] - 2. * v[0, 0, 0]
        return v2 / (h * h)

    @gtscript.function
    def ddz(v, h = 0.1):
        v2 = v[0, 0, -1] + v[0, 0, 1] - 2. * v[0, 0, 0]
        return v2 / (h * h)

    @gtscript.stencil(backend=backend)
    def laplacian(
        v: gtscript.Field[np.float64], lap: gtscript.Field[np.float64], *, h: np.float64 = 0.1
    ):
        with computation(PARALLEL), interval(1, -1):
            lap = ddx(v, h) + ddy(v, h) + ddz(v, h)


Functions are pure, that is, none of the passed fields are modified and the results must be passed back using
the ``return`` statement. In the above example, ``v`` is not modified. However, multiple return values are allowed:

.. code:: python

    @gtscript.function
    def ddxyz(v, h = 0.1):
        x = v[-1, 0, 0] + v[1, 0, 0] - 2. * v[0, 0, 0]
        y = v[0, -1, 0] + v[0, 1, 0] - 2. * v[0, 0, 0]
        z = v[0, 0, -1] + v[0, 0, 1] - 2. * v[0, 0, 0]
        return x / (h * h), y / (h * h), z / (h * h)

    @gtscript.stencil(backend=backend)
    def laplace(
        v: gtscript.Field[np.float64], lap: gtscript.Field[np.float64], *, h: np.float64 = 0.1
    ):
        with computation(PARALLEL), interval(1, -1):
            x, y, z = ddxyz(v, h)
            lap = x + y + z

Functions can be used only for code inside of computation/interval blocks. There is no overhead attached
to function calls since they are inlined in the generated code.

-------------
Offset syntax
-------------

Offsets can be specified either as a list of offsets on all spatial axes, e.g. ``field[0, 0, 1]``, or as offsets on the
axes present by specifying the axis ``field[K+1]``.

------------
System Setup
------------

Compilation settings for GT4Py backends generating C++ or CUDA code can be modified by updating
the default values in the `gt4py.cartesian.config <https://github.com/GridTools/gt4py/blob/main/src/gt4py/cartesian/config.py>`_ module.
Note that most of the system dependent settings can also be modified using the following environment variables:

* ``CUDA_ROOT`` or ``CUDA_HOME``: installation prefix of the CUDA toolkit.
* ``GT_INCLUDE_PATH``: path prefix to an alternative installation of GridTools header files.
* ``OPENMP_CPPFLAGS``: preprocessor arguments for OpenMP support.
* ``OPENMP_LDFLAGS``: arguments when linking executables with OpenMP support.


MacOS
-----

The clang compiler supplied with the MacOS Command Line Tools does not support the ``-fopenmp`` flag, but it does have
support for OpenMP in the C preprocessor and can link with OpenMP support if the libomp package is installed using
``homebrew`` (https://brew.sh/). Then set the following environment variables:

.. code:: bash

    export OPENMP_CPPFLAGS="-Xpreprocessor -fopenmp"
    export OPENMP_LDFLAGS="$(brew --prefix libomp)/lib/libomp.a"
