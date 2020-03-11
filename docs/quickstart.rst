=================
Quick Start Guide
=================

This document will guide you through the basic steps to get started with GT4Py.


------------
Installation
------------

GT4Py contains a standard ``setup.py`` installation script and thus it might be
installed as usual with *pip*. Additional commands are provided to install
and remove the GridTools C++ sources, which are not contained in the package.

We strongly recommended to create a virtual environment for any new project:

.. code:: bash

    python -m venv path_for_the_new_venv
    source path_for_the_new_venv/bin/activate
    pip install --upgrade wheel


Then clone the GT4Py repository and install the local copy or install it
directly from GitHub with ``pip``. For the CUDA backends add the
`[cudaXX]` optional dependency, where `XX` takes the values `90`, `91`,
`92`, `100` or `101` depending on the CUDA version installed in your system
(CUDA version 9.0, 9.1, ..., 10.1).

.. code:: bash

    git clone https://github.com/GridTools/gt4py.git
    pip install ./gt4py     # pip install ./gt4py[cuda101]

Or

.. code:: bash

    pip install git+https://github.com/gridtools/gt4py.git
    # pip install git+https://github.com/gridtools/gt4py.git#egg=gt4py[cuda101]

Finally, the last step to install the required GridTools C++ sources:

.. code:: bash

    python -m gt4py.gt_src_manager install


------------
Introduction
------------
In GT4Py, grid computations such as stencils are defined in a Domain Specific Language (DSL) which is called GTScript.
GTScript is syntactically a subset of python, but has different semantics. Computations defined in this DSL are then
translated by the GT4Py tool chain into code in Python based on `NumPy <http://www.numpy.org/>`_ or C++/CUDA based on
the `GridTools <http://gridtools.github.io/>`_ library. To be able to achieve full performance with GridTools, data has
to adhere to certain layout requirements, which are taken care of by storing the data in GT4Py storage containers.

A Simple Example
----------------
Suppose we want to write a simple stencil computing a parametrized linear combination of multiple 3D fields. The stencil
would have some parameters to change the relative weights of the input fields in the linear combination. As an example,
we could define this stencil with three input fields (``field_a``, ``field_b`` and ``field_c``) and two parameters
(``alpha`` and ``weight``) in the following way:

.. code:: python

    import numpy as np

    import gt4py.gtscript as gtscript

    backend = "numpy"

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


This definition basically expresses the operations (or *kernel*) performed at every point of the computation domain to
generate the output values. The indices inside the brackets are interpreted as offsets relative to the
current point in the iteration, and not as absolute positions in the data fields. For an explanation of the line
``with computation(PARALLEL), interval(...):``, please refer to the section *Computations and Intervals*.

.. note::
    While not required, we recommend to specify *fields* as standard arguments and run-time *parameters* as
    *keyword-only* arguments.

Once the stencil kernel has been defined, we use GT4Py to generate an *implementation* of this high-level definition as
a callable object that we can use to apply the operations to data. This can be done by just decorating the definition
function with the ``stencil`` decorator provided by GT4Py. In this case, for example:

.. code:: python

    import numpy as np

    import gt4py.gtscript as gtscript

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

The ``stencil`` decorator generates code in Python or C++ depending on the ``backend`` specified by name.
Currently, the following backends are available:

* ``"debug"``: a slow, yet human-readable python backend
* ``"numpy"``: a vectorized python backend
* ``"gtx86"``: a backend based on GridTools code performance-optimized for x86 architecture
* ``"gtmc"``: a GridTools backend targeting many core architectures
* ``"gtcuda"``: a GridTools backend targeting GPUs

The decorator further replaces the stencil definition function (here ``stencil_example``) by a callable object that
can be used as a function to call the generated code which modifies the passed data in place.

If for any reason we cannot (or we do not want to) use the ``stencil`` decorator, it is also possible to call it as a
regular function call receiving the definition function:

.. code:: python

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

    stencil_example_implementation = gtscript.stencil(backend="numpy", definition=stencil_example)

    another_example_implementation = gtscript.stencil(backend="gtmc", definition=stencil_example)


--------
Storages
--------
Since some of the backends require your data to be in a certain layout in memory, GT4Py provides a special `NumPy`-like
multidimensional array implementation called ``storage``. Storage containers can be allocated through the same familiar
set of routines used in `NumPy` for allocation: ``from_array``, ``ones``, ``zeros`` and ``empty``.

The ``default_origin`` parameter plays two roles:

#. If when calling the stencil, no other `origin` is specified, this value is where the `iteration domain` begins, i.e.
   the grid point with the lowest index where a value is written.

#. The data is allocated such that memory address of the point specified in ``default_origin`` is `aligned` to  a
   backend-dependent value. This is a performance concern. Ideally, you set this to the value to a point which is the
   corner of the iteration domain with the lowest coordinates for most of your stencils.

.. code:: python

    import gt4py.storage as gt_storage

    field_a = gt_storage.from_array(
        data=np.random.randn(10, 10, 10),
        backend=backend,
        dtype=np.float64,
        default_origin=(0, 0, 0),
    )
    field_b = gt_storage.ones(
        backend=backend, shape=(10, 10, 10), dtype=np.float64, default_origin=(0, 0, 0)
    )
    field_c = gt_storage.zeros(
        backend=backend, shape=(10, 10, 10), dtype=np.float64, default_origin=(0, 0, 0)
    )
    result = gt_storage.empty(
        backend=backend, shape=(10, 10, 10), dtype=np.float64, default_origin=(0, 0, 0)
    )

    stencil_example(field_a, field_b, field_c, result, alpha=0.5)


--------------------------
Computations and Intervals
--------------------------
We have already seen the stencil interface for fields and parameters, as well as externals and compile-time conditions.
Let's now look at `computations` and `intervals`. The `computation` context determines in which order the vertical
dimension is iterated over. ``FORWARD`` stands for an iteration from low to high index, while ``BACKWARD`` is an
iteration from high to low index. For contexts declared ``PARALLEL``, no order can be assumed and only statement are
allowed for which the result is the same irrespective of iteration order.

`Intervals` are the second information given to a context. they declare the range of indices for which the statements
of the respective context are applied. E.g. ``interval(0,1)`` declares that the following context is applied for indices
in [0,1), i.e. only to `K=0`. The ``interval(1, None)`` represents indices in [1,∞), ``interval(0, -1)`` all indices
except the last.

.. code:: python

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



However, the ``PARALLEL`` and other orders differ in more ways. For parallel regions, we can assume that each statement
(i.e. each assign) is applied to the full domain, before the next one starts. If an iteration order is specified
however, all statements are applied to each slice with the same ``K``, one after each other, before moving to ``K+1``.


-----------
Subroutines
-----------
To reuse code elements and to structure your code, subroutines are a useful tool. They need to be decorated with
``@gtscript.function``.

.. code:: python

    @gtscript.function
    def ddx(v, h = 0.1):
        v2 = v[-1, 0, 0] + v[1, 0, 0] - 2 * v[0, 0, 0]
        return v2 / (h * h)

    @gtscript.function
    def ddy(v, h = 0.1):
        v2 = v[0, -1, 0] + v[0, 1, 0] - 2 * v[0, 0, 0]
        return v2 / (h * h)

    @gtscript.function
    def ddz(v, h = 0.1):
        v2 = v[0, 0, -1] + v[0, 0, 1] - 2 * v[0, 0, 0]
        return v2 / (h * h)

    @gtscript.stencil(backend=backend)
    def laplace(
        v: gtscript.Field[np.float64], lap: gtscript.Field[np.float64], *, h: np.float64 = 0.1
    ):
        with computation(PARALLEL), interval(1, -1):
            lap = ddx(v, h) + ddy(v, h) + ddz(v, h)


They are pure functions, that is, the none of the passed fields are modified and the results are passed only through
the ``return`` statement. That is, in the above example, ``v`` is not modified. However, multiple return values are allowed:

.. code:: python

    @gtscript.function
    def ddxyz(v, h=0.1):
        x = v[-1, 0, 0] + v[1, 0, 0] - 2 * v[0, 0, 0]
        y = v[0, -1, 0] + v[0, 1, 0] - 2 * v[0, 0, 0]
        z = v[0, 0, -1] + v[0, 0, 1] - 2 * v[0, 0, 0]
        return x / (h * h), y / (h * h), z / (h * h)

    @gtscript.stencil(backend=backend)
    def laplace(
        v: gtscript.Field[np.float64], lap: gtscript.Field[np.float64], *, h: np.float64 = 0.1
    ):
        with computation(PARALLEL), interval(1, -1):
            x, y, z = ddxyz(v, h)
            lap = x + y + z


-------------------------
Compile-time conditionals
-------------------------
Run-time parameters are a powerful way to customize the computation with scalar values that may be different for every
call. However, sometimes a structural modification of the kernel definition is required depending on the context. For
example, when we are testing an extension to a existing model, we might want to perform some additional computations
when running the extended versions and compare the results against the regular one. For this purpose we can force a
*compile-time* evaluation of a conditional ``if`` statement whose test condition depends only on **constant symbol**
definitions. The condition will be thus evaluated at the generation step and only the statements in the
selected branch will be actually compiled, allowing more drastic changes in the kernel definition.

For example, the previous definition could be modified in the following way:

.. code:: python

    USE_ALPHA = True

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
        if __INLINED(USE_ALPHA):
            with computation(PARALLEL), interval(...):
                result = field_a[0, 0, 0] - (1 - alpha) * (
                    field_b[0, 0, 0] - weight * field_c[0, 0, 0]
                )
        else:
            with computation(PARALLEL), interval(...):
                result = field_a[0, 0, 0] - (field_b[0, 0, 0] - weight * field_c[0, 0, 0])


The ``__INLINED()`` call is used to force the compile-time evaluation of ``USE_ALPHA``, which is an external symbol
that must be defined explicitly before the ``gtscript.stencil()`` decorator processes the definition function.
For `C` programmers, compile-time evaluation of conditional statements could be considered a bit like preprocessor
``#IF`` definitions.

Alternatively, the actual values of *constant* symbols might be defined in the ``gtscript.stencil()`` call as a
dictionary passed to the ``externals`` keyword. This allows an even more flexible way to parametrize kernel definitions.
In this case, the symbol must further be imported from ``__externals__`` in the body of the function definition.

.. code:: python

    @gtscript.stencil(backend=backend, externals={"USE_ALPHA": True})
    def stencil_example(
        field_a: gtscript.Field[np.float64],
        field_b: gtscript.Field[np.float64],
        field_c: gtscript.Field[np.float64],
        result: gtscript.Field[np.float64],
        *,
        alpha: np.float64,
        weight: np.float64 = 2.0,
    ):
        from __externals__ import USE_ALPHA

        if __INLINED(USE_ALPHA):
            with computation(PARALLEL), interval(...):
                result = field_a[0, 0, 0] - (1 - alpha) * (
                    field_b[0, 0, 0] - weight * field_c[0, 0, 0]
                )
        else:
            with computation(PARALLEL), interval(...):
                result = field_a[0, 0, 0] - (field_b[0, 0, 0] - weight * field_c[0, 0, 0])

