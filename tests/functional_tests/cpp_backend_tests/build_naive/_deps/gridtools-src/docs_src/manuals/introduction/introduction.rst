.. include:: ../defs.hrst

.. _introduction:

============
Introduction
============

-----------------
What Is GridTools
-----------------

The |GT| (GT) framework is a set of libraries and utilities to develop performance portable applications in which
stencil operations on grids are central. The focus of the project is on regular and block-structured grids as are
commonly found in the weather and climate application field. In this context, GT provides a useful level of abstraction
to enhance productivity and obtain excellent performance on a wide range of computer architectures. Additionally, it
addresses the challenges that arise from integration into production code, such as the expression of boundary
conditions, or conditional execution. The framework is structured such that it can be called from different weather
models (numerical weather and climate codes) or programming interfaces, and can target various computer architectures.
This is achieved by separating the GT core library in a user facing part (frontend) and architecture specific (backend)
parts. The core library also abstracts various possible data layouts and applies optimizations on stages with multiple
stencils. The core library is complemented by facilities to interoperate with other languages (such as C and Fortran),
to aid code development and a communication layer.

|GT| provides optimized backends for GPUs and manycore architectures. Stencils can be run efficiently on different
architectures without any code change needed. Stencils can be built up by small composeable units called stages, using
|GT| domain-specific language. Such a functor can be as simple as being just a copy stencil, copying data from one field
to another:

.. code-block:: gridtools

  struct copy_functor {
      using in = in_accessor<0>;
      using out = inout_accessor<1>;

      using param_list = make_param_list<in, out>;

      template <typename Evaluation>
      GT_FUNCTION static void apply(Evaluation eval) {
          eval(out()) = eval(in());
      }
  };

Several such stages can be composed into a :term:`Computation` and be applied on each grid-point of a grid. Requiring this
abstract descriptions of a stencils, the DSL allows |GT| can apply architecture-specific optimizations to the stencil
computations in order to be optimal on the target hardware.


^^^^^^^^^^^^
Dependencies
^^^^^^^^^^^^

|GT| requires a header-only installation of Boost_, a recent version of CMake_ and a modern compiler.
The exact version requirements can be found on `GitHub <https://github.com/GridTools/gridtools>`_.

.. _Boost: https://www.boost.org/
.. _CMake: https://www.cmake.org/

Addtionally |GT| requires the following optional dependencies.
For the communication module (GCL) *MPI* is required. For the GPU backends a *CUDA* or *HIP* compiler is required.
For the CPU backends *OpenMP* is required.


---------------
Using GridTools
---------------

|GT| uses CMake as its build system.  CMake offers two ways of using a CMake-managed project: from an installation
or using FetchContent to pull in a dependency on the fly. |GT| supports both ways.

.. _installation:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Installing and Validating GridTools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install |GT| (in ``/usr/local``)
and run the tests, use the following commands.

.. code-block:: shell

 git clone http://github.com/GridTools/gridtools.git
 cd gridtools
 mkdir build && cd build
 cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
 cmake --build --parallel 4
 ctest

CMake will detect the optional dependencies and enable the available backends accordingly.
During configure the available |GT| targets will be listed.

The following CMake variables are available to customize the installation of |GT|.

  * Set ``GT_INSTALL_EXAMPLES`` to ``ON`` and
  * select a directory for installation with ``GT_INSTALL_EXAMPLES_PATH``.
    The examples come with a standalone CMake project and can be built separately.

Additionally, use the following CMake variables to customize building of tests.

  * Set ``BUILD_TESTING`` to ``OFF`` to disable building any tests (fast installation without validation).
  * Set ``GT_GCL_GPU`` to ``OFF`` to disable the ``gcl_gpu`` target (and to disable building of GPU GCL tests).
    This is useful, if you have CUDA in your environment, but the MPI implementation is not CUDA-aware.
  * Set ``GT_CUDA_ARCH`` to the compute capability of the GPU device on which you want to run the tests.
  * If your compiler is a CUDA-capable *Clang*, you can switch how CUDA code will be compiled, by setting
    ``GT_CLANG_CUDA_MODE`` to one of ``AUTO`` (default, prefer Clang-CUDA if available), ``Clang-CUDA`` (compile with
    Clang), ``NVCC-CUDA`` (compile with NVCC and Clang as the host compiler).

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using a GridTools Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using |GT| follows standard CMake practices. If |GT| was installed to `<prefix>`,
provide ``-DCMAKE_PREFIX_PATH=<prefix>`` or ``-DGridTools_ROOT=<prefix>`` to indicate where |GT| can be found.
The ``CMakeLists.txt`` file should then contain the following line:

.. code-block:: cmake

 find_package(GridTools VERSION ... REQUIRED)

.. note::

  If |GT| should use CUDA with NVCC, you must call ``enable_language(CUDA)`` before the call to ``find_package``.

.. note::

  If you are compiling with *Clang*, set the variable ``GT_CLANG_CUDA_MODE`` before calling ``find_package``, see
  :ref:`installation`. It is recommended to make this variable a CMake cached variable to allow users of your code
  to change the mode.

.. _fetch_content:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using GridTools with CMake’s FetchContent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively to a |GT| installation you can use |GT| with *FetchContent*. To use FetchContent add the following
lines to your CMake project

.. code-block:: cmake

  include(FetchContent)
  FetchContent_Declare(GridTools
      URL https://github.com/GridTools/gridtools/archive/<release_tag>.tar.gz
  )
  FetchContent_GetProperties(GridTools)
  if(NOT GridTools_POPULATED)
      FetchContent_Populate(GridTools)
      add_subdirectory(${gridtools_SOURCE_DIR} ${gridtools_BINARY_DIR})
  endif()

where *<release_tag>* is the git tag of the |GT| release, e.g. ``v2.0.0``.

The following CMake options are available (see also :ref:`installation`).

  * By default, all |GT| tests are disabled. To enable building of the tests, set ``GT_BUILD_TESTING`` to ``ON``.
    If tests are enabled, their behavior can be changed as described in :ref:`installation`.
  * Use ``GT_CLANG_CUDA_MODE`` to select how CUDA code is compiled, see :ref:`installation`.

^^^^^^^^^^^^^^^
Using GridTools
^^^^^^^^^^^^^^^

After |GT| was made available by either ``find_package`` or ``FetchContent`` the following targets for the
different |GT| modules are available

  * ``stencil_naive``

If *OpenMP* is available

  * ``stencil_cpu_ifirst``
  * ``stencil_cpu_kfirst``
  * ``storage_cpu_ifirst``
  * ``storage_cpu_kfirst``
  * ``layout_transformation_cpu``
  * ``boundaries_cpu``

If *OpenMP* **and** *MPI* is available

  * ``gcl_cpu``

If a *CUDA runtime* (or a *HIP compiler*) is available (no *CUDA compiler* required)

  * ``storage_gpu``

If a *CUDA compiler* or a *HIP compiler* is available

  * ``stencil_gpu``
  * ``layout_transformation_gpu``
  * ``boundaries_gpu``

If a *CUDA compiler* and *MPI* is available

  * ``gcl_gpu`` (can be disabled by the user if the MPI implementation is not CUDA-aware)

After linking to the |GT| backend, we recommend to call the CMake function
``gridtools_setup_target(<target> [CUDA_ARCH <compute_capability>])`` on your target.
The function helps abstracting differences in how CUDA code is compiled
(e.g. *Clang* uses a different flag than *NVCC* for the CUDA architecture). Additionally, using this function
allows to compile the same *.cpp* file for both CUDA and host, without having to wrap the implementation in a
*.cu* file.

.. code-block:: cmake

 add_library(my_library source.cpp)
 target_link_libraries(my_library PUBLIC GridTools::stencil_cuda)
 gridtools_setup_target(my_library CUDA_ARCH sm_60)


^^^^^^^^^^^^^^^
AMD GPU Support
^^^^^^^^^^^^^^^

Further, |GT| can also be compiled for AMD GPUs using AMD’s HIP. To compile |GT| you need the Clang-based HIP compiler which is available with ROCm 3.5 and later.


------------
Contributing
------------

Contributions to the |GT| set of libraries are welcome. However, our policy is that we will be the official maintainers
and providers of the |GT| code. We believe that this will provide our users with a clear reference point for
support and guarantees on timely interactions. For this reason, we require that external contributions to |GT|
will be accepted after their authors provide to us a signed copy of a copyright release form to ETH Zurich.
