.. include:: ../defs.hrst


.. _getting_started:

===============
Getting Started
===============

This chapter describes how to use |GT| to solve a (simple) PDE.
We will use a fourth-order horizontal smoothing filter
to explain the necessary steps to assemble a
stencil from scratch. We will not go into details in this chapter but
refer to later chapters for more details.

Our example PDE is given by

.. math::
   \frac{\partial \phi}{\partial t} =\begin{cases}
   - \alpha \nabla^4 \phi & z \leq z_\text{0}\\
   0 & z > z_0
   \end{cases}

where :math:`\nabla^4` is the squared two dimensional horizontal
Laplacian and we apply the filter only up to some maximal :math:`z_0` (to make
the example more interesting). The filter is calculated in two steps:
first we calculate the Laplacian of :math:`\phi`

.. math::
   L = \Delta \phi = \left( \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}\right) \phi

then we calculate the Laplacian of :math:`L` as :math:`-\alpha \nabla^4 \phi = -\alpha  \Delta L`.

In the following we will walk through the following steps:

-   The |GT| coordinate system and its notation.
-   Storages: how does |GT| manage the input and output fields.
-   The first stencil: calculating :math:`L`, the second order Laplacian of :math:`\phi`.
-   The final stencil: function calls, apply-method overloads and temporaries

-----------------
Coordinate System
-----------------

For a finite difference discretization we restrict the field
:math:`\phi \in \mathbb{R}^3` to a discrete grid. We use the notation
:math:`i = x_i` and :math:`j = y_j` for the horizontal dimension and :math:`k = z_k` for
the vertical dimension, where :math:`x_i, y_j, z_k` are the :math:`x,y,z`
coordinates restricted on the grid. The *computation domain* is defined
by all grid points in our domain of interest

.. math::

   \Lambda = (i,j,k) \quad \text{with}\quad i \in \{ 0\dots N_i-1\}, j \in \{0\dots N_j-1\}, k\in\{0 \dots N_k-1\}

|GT| supports any number of dimension, however the iteration is always restricted to three dimensions, and |GT| will
treat one dimension, here the :math:`k` dimension, differently: the :math:`ij`-plane is executed in parallel while the
computation in :math:`k` can be sequential. The consequence is that there must not be a dependency in :math:`ij` within
a stencil while there can be a dependency in :math:`k`. For now (this chapter) it is sufficient to just remember that
the :math:`ij`-plane and the :math:`k` dimension are treated differently by |GT|.

The calculation domain is surrounded by a *boundary region* as depicted
in :numref:`fig_getting_started_coordinates`. Computation happens
only within the calculation domain but values may be read from grid
points in the boundary region.

.. _fig_getting_started_coordinates:
.. figure:: figures/coordinates.png
   :scale: 60 %

   Coordinate system

--------
Storages
--------

In this section we will set up the fields for our example: we need a
storage for the :math:`\phi`-field (``phi_in``) and a storage for the output
(``phi_out``).

Storages in |GT| are n-dimensional array-like objects with the
following capabilities:

-   access an element with :math:`(i,j,k)` syntax

-   synchronization between CPU memory and a device (e.g. a CUDA capable GPU)

.. _getting_started_storage_traits:

^^^^^^^^^^^^^^
Storage Traits
^^^^^^^^^^^^^^

Since the storages depend on the architecture (e.g. CPU or GPU) our first step
is to define the *storage traits* type which typically looks like

.. literalinclude:: code/test_gt_storage.cpp
   :language: gridtools
   :start-after: #ifdef GT_CUDACC
   :end-before: #else

for the CUDA :term:`Storage Traits` or

.. literalinclude:: code/test_gt_storage.cpp
   :language: gridtools
   :start-after: #else
   :end-before: #endif

for the CPU :term:`Storage Traits`.

^^^^^^^^^^^^^^^^^
Building Storages
^^^^^^^^^^^^^^^^^

For efficient memory accesses the index ordering might depend on the target architecture, therefore the
memory layout will be implicitly decided by storage traits.

|GT| storage classes don't have user facing constructors. The builder design pattern is used instead.
The library provides the ``storage::builder`` object template that is instantiated by storage traits.
To create a storage we need to supply the builder with the desired storage properties by chaining
the setter methods and finally call the ``build()`` method which returns a ``std::shared_ptr``
of a newly created storage. The builder also provides overloaded call operator which is a synonym of ``build()``.
There are two required properties that need to be set:
the type of element, eg ``.type<double>()``, and the sizes for each dimension, eg ``.dimensions(10, 12, 20)``.
Other properties are optional.

If we need several storages that share some properties, we can construct a partially specified builder by
setting the shared properties and reuse it while building concrete storages.

.. literalinclude:: code/test_gt_storage.cpp
   :language: gridtools
   :start-after: int main() {
   :end-at: auto lap
   :dedent: 4

.. note::

   It is recommended to use the ``id`` property each time the ``dimensions`` property is set.
   ``id`` should identify the unique set of dimension sizes.
   This is because the stencil computation engine assumes that the storages that have the same ``id`` have the same
   sizes. However, if only one set of dimensions is used, the ``id`` property can be skipped.

We can now

-   retrieve the name of the field,
-   create a view and read and write values in the field using the parenthesis syntax,
-   query the lengths of each dimension.


.. literalinclude:: code/test_gt_storage.cpp
   :language: gridtools
   :start-after: auto lap =
   :end-before: }
   :dedent: 4

--------
Stencils
--------

A *stencil* is a kernel that updates array elements according to a fixed
access pattern.

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example: Naive 2D Laplacian
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest discretization of the 2D Laplacian is the finite difference
five-point stencil as depicted in :numref:`fig_getting_started_2dlap`.

.. _fig_getting_started_2dlap:
.. figure:: figures/Laplacian2D.png
   :scale: 60 %

   Access pattern of a 2D Laplacian

For the calculation of
the Laplacian at a given grid point we need the value at the grid point
itself and its four direct neighbors along the Cartesian axis.

A naive C++ implementation of the 2D Laplacian stencil looks as follows:

.. literalinclude:: code/test_naive_implementation.cpp
   :language: cpp
   :start-after: // lap-begin
   :end-before: // lap-end

Apart from the initialization the stencil implementation
consists of 2 main components:

- Loop-logic: defines the stencil application domain and loop order
- Update-logic: defines the update formula (here: the 2D Laplacian)

Special care has to be taken at the boundary of the domain. Since the
Laplacian needs the neighboring points, we cannot calculate the Laplacian
on the boundary layer and have to exclude it from the loop.

-----------------------
First GridTools Stencil
-----------------------

In |GT| the loop logic and the storage order are implemented (and optimized) by the library while the update function, to
be applied to each gridpoint, is implemented by the user. The loop logic (for a given architecture) is combined with the
user-defined update function at compile-time by template meta-programming.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Update-logic: GridTools 2D Laplacian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The update-logic is implemented with state-less functors. A
|GT| functor is a ``struct`` or ``class`` providing a *static* method
called ``apply``. The update-logic is implemented in these :term:`Apply-Methods<Apply-Method>`.
As the functors are state-less (no member variables, static methods
only) they can be passed by type, i.e. at compile-time, and therefore
allow for compile-time optimizations.

.. literalinclude:: code/test_gt_laplacian.cpp
   :language: gridtools
   :start-after: #endif
   :end-before: int main() {

In addition to the ``apply``-method, the functor contains ``accessor`` s. These
two ``accessor`` s are parameters of the functor, i.e. they are mapped to
fields passed to the functor. They contain compile-time information if
they are only used as input parameters, e.g. the ``in`` accessor in the
example, or if we want to write into the associated field (``inout``). Additionally,
the ``extent`` defines which grid points are needed by the stencil relative
to the current point. The format for the extent is

.. code-block:: gridtools

   extent<i_minus, i_plus, j_minus, j_plus, k_minus, k_plus>

where ``i_minus`` and ``i_plus`` define an interval on the :math:`i`-axis relative to
the current position; ``i_minus`` is the negative offset, i.e. zero or a
negative number, while ``i_plus`` is the positive offset. Analogously for
:math:`j` and :math:`k`. In the Laplacian example, the first two numbers
in the extent of the ``in`` accessor define that we want to access the
field at :math:`i-1`, :math:`i` and  :math:`i+1`. The accessor type and the extent is needed for a
dependency analysis in the compile-time optimizations for more complex
stencils. (For example, the computation
domain needs to be extended when we calculate the Laplacian of the Laplacian later. This is done automatically by the
library.)

The first template argument is an index defining the order of the
parameters, i.e. the order in which the fields are passed to the
functor. The ``param_list`` is a |GT| keyword which has to be defined for each stencil,
and should contain the list of accessors.

A ``apply``-method needs as first parameter a context
object, usually called ``eval``, which is created and passed to the method by the library on
invocation. This object contains, among other things, the index of the
active grid point (:term:`Iteration Point`) and the mapping of data-pointers to the ``accessor`` s. The
second argument is optional and specifies the interval on the :math:`k`-axis where this implementation
of the :term:`Apply-Method` should be executed. This allows to apply a different update-logic on
:term:`Vertical Intervals<Vertical Interval>` by overloading the :term:`Apply-Method`. We will define :term:`Vertical Intervals<Vertical Interval>`
later. If the second parameter is not specified, a default interval is assumed.

The body of the ``apply``-method looks quite similar to the one in the
naive implementation, except that each
field access has to be wrapped by a call to the context object ``eval``.
This is necessary to map the compile-time parameter, the ``accessor``, to
the run-time data in the ``data_store``.

^^^^^^^^^^^^^^^^^^^
Calling the Stencil
^^^^^^^^^^^^^^^^^^^

In the naive implementation, the call to the ``laplacian`` is as simple as

.. code-block:: gridtools

   int boundary_size = 1;
   laplacian( lap, phi, boundary_size );

since it contains already all the information: the update-logic *and* the loop-logic.

The |GT| stencil, does not contain any
information about the loop-logic, i.e. about the domain where we want to apply the stencil operation,
since we need to specify it in a platform-independent syntax, a *domain specific embedded language*
(DSEL), such that the :term:`Backend` can decide on the specific implementation.

For our example this looks as follows

.. literalinclude:: code/test_gt_laplacian.cpp
   :language: gridtools
   :start-after: int main() {
   :end-before: }
   :dedent: 4
   :linenos:

In lines 14-16 we setup the physical dimension of the problem.
First we define which points on the :math:`i` and the :math:`j`-axis belong
to the computational domain and which points belong to the boundary (or
a padding region). For now it is enough to know that these lines define
a region with a boundary of size 1 surrounding the :math:`ij`-plane. In the
next lines the layers in :math:`k` are defined. In this case we have only one
interval. We will discuss the details later.

In line 18 we execute the stencil computation. In our example only one stencil participates.
Hence, we can use the simple ``run_single_stage`` API. We pass the stencil the backend object, the grid
(the information about the loop bounds) and the storages on which the computation needs to be executed.
The number and the order of storage arguments should match the number of stencil accessors.

^^^^^^^^^^^^^^^^^^^^^^^^
Full GridTools Laplacian
^^^^^^^^^^^^^^^^^^^^^^^^

The full working example looks as follows:

.. literalinclude:: code/test_gt_laplacian.cpp
   :language: gridtools
   :linenos:

There are some points which we did not discuss so far. For a first look at |GT| these can be considered fixed patterns and
we won't discuss them now in detail.

A common pattern is to use the preprocessor flag ``__CUDACC__`` to distinguish between CPU and GPU code.
Here we use the |GT| internal ``GT_CUDA`` macro to do this. The macro is used to set the correct :term:`Backend`
and :term:`Storage Traits` for the CUDA or CPU architecture, respectively.

The code example can be compiled using the following simple CMake script (requires an installation of |GT|, see :ref:`installation`).

.. literalinclude:: code/CMakeLists.txt

-------------------------------------
Assembling Stencils: Smoothing Filter
-------------------------------------

In the preceding section we saw how a first simple |GT| stencil
is defined and executed. In this section we will use this stencil to
compute our example PDE. A naive implementation could look as in

.. literalinclude:: code/test_naive_implementation.cpp
   :language: gridtools
   :start-after: // smoothing-begin
   :end-before: // smoothing-end

For the |GT| implementation we will learn three things in this
section: how to define special regions in the :math:`k`-direction; how to use
|GT| temporaries and how to call functors from functors.

^^^^^^^^^^^^^^^^^^^^^^^
`apply`-method overload
^^^^^^^^^^^^^^^^^^^^^^^

Our first |GT| implementation will be very close to the naive
implementation: we will call two times the Laplacian functor from the
previous section and store the result in two extra fields. Then we will
call a third functor to compute the final result. This functor shows how
we can specialize the computation in the :math:`k`-direction:

.. literalinclude:: code/gt_smoothing_variant1_operator.hpp
   :language: gridtools

We use two different
:term:`Vertical Intervals<Vertical Interval>`, the ``lower_domain`` and the ``upper_domain``, and provide an overload of the
:term:`Apply-Method` for each interval.

The :term:`Vertical Intervals<Vertical Interval>` are defined as

.. literalinclude:: code/gt_smoothing.hpp
   :language: gridtools
   :start-after: constexpr unsigned halo = 2;
   :end-before: struct lap_function {

The first line defines an axis with 2 :term:`Vertical Intervals<Vertical Interval>`. From this axis retrieve the :term:`Vertical Intervals<Vertical Interval>`
and give them a name.

Then we can assemble the computation

.. literalinclude:: code/gt_smoothing_variant1_computation.hpp
   :language: gridtools

We cannot use the ``run_single_stage`` API because now we need to compose a :term:`Computation` from three stencil calls. To achieve that
we use the full featured ``run`` API. It requires a stencil composition specification as a first parameter. That specification
is provided in form of a generic lambda. Its arguments represent the storages that are used in the computation.
The expression that is returned describes how the stencils should be composed. It is where the |GT| DSL is used.
``execute_parallel()`` means that each :math:`k`-level can be executed in parallel. The ``stage`` clause represents
a call to the stencil with the given arguments.

In this version we needed to explicitly allocate the temporary fields
``lap`` and ``laplap``. In the next section we will learn about
|GT| temporaries.

^^^^^^^^^^^^^^^^
|GT| Temporaries
^^^^^^^^^^^^^^^^

|GT| *temporary storages* are storages with the lifetime of the
``computation``. This is exactly what we need for the ``lap`` and ``laplap``
fields.

.. note::

   Note that temporaries are not allocated explicitly and we cannot
   access them from outside of the computation. Therefore, sometimes it might be
   necessary to replace a temporary by a normal storage for debugging.

To use temporary storages we exclude the correspondent fields from the arguments of our ``spec`` and declare them as
temporaries, within the `spec` lambda, using the ``GT_DECLARE_TMP`` macro. We don't need the explicit
instantiations any more and don't have to pass them to the ``run``. The new code looks as follows

.. literalinclude:: code/gt_smoothing_variant2_computation.hpp
   :language: gridtools

Besides the simplifications in the code (no explicit storage needed), the
concept of temporaries allows |GT| to apply optimization. While normal storages
have a fixed size, temporaries can have block-private :term:`Halos<Halo>` which are used for redundant computation.

.. note::

   It might be semantically incorrect to replace a temporary with a normal storage, as normal storages don't have the :term:`Halo`
   region for redundant computation. In such case several threads (OpenMP or CUDA) will write the same location multiple
   times. As long as all threads write the same data (which is a requirement for correctness of |GT|), this should be
   no problem for correctness on current hardware (might change in the future) but might have side-effects on performance.

.. note::

   This change from normal storages to temporaries did not require any code changes to the functor.

^^^^^^^^^^^^^
Functor Calls
^^^^^^^^^^^^^

The next feature we want to use is the *stencil function call*. In the first example we computed the Laplacian
and the Laplacian of the Laplacian explicitly and stored the intermediate values in the temporaries. Stencil function
calls will allow us do the computation on the fly and will allow us to get rid of the temporaries.

.. note::

   Note that this is
   not necessarily a performance optimization. It might well be that the version with temporaries is actually the
   faster one.

In the following we will remove only one of the temporaries. Instead of calling the Laplacian twice from the
``spec``, we will move one of the calls into the smoothing functor. The new smoothing functor looks as follows

.. literalinclude:: code/gt_smoothing_variant3_operator.hpp
   :language: gridtools

In ``call`` we specify the functor which we want to apply.  In ``with`` the ``eval`` is forwarded, followed by all the
input arguments for the functor. The functor in the call is required to have exactly one ``inout_accessor`` which will
be the return value of the call. Note that ``smoothing_function_3`` still needs to specify the extents explicitly;
for functor calls they cannot be inferred automatically.

One of the ``stage(lap_function(), ...)`` was now moved inside of the functor, therefore the new spec is just:

.. literalinclude:: code/gt_smoothing_variant3_computation.hpp
   :language: gridtools

The attentive reader may have noticed that our first versions did more
work than needed: we calculated the Laplacian of the Laplacian of phi
(:math:`\Delta \Delta \phi`) for all :math:`k`-levels, however we used it only for
:math:`k<k_\text{max}`. In this version we do a bit better: we still calculate
the Laplacian (:math:`L = \Delta \phi`) for all levels but we only calculate
:math:`\Delta L` for the levels where we need it.
