===========================
GT4Py: GridTools for Python
===========================

GT4Py is a DSL compiler framework for weather and climate modeling, allowing users
to write codes in a simplified stencil language using standard Python syntax and
executing them with great performance on different machine architectures, such as
CPUs and GPUs.

**How does it work?** GT4Py takes your stencils written in GTScript and generates
optimized C++ code, which is compiled just-in-time to deliver the best
possible performance. GT4Py's GTScript language is readable and concise.
As an extension of Python it has a familiar look and feel to anyone who knows
Python and interoperates nicely with scientific Python software stack (e.g. NumPy, SciPy).

**Great! So what does it look like in practice?** Look at the following example of
a Laplacian operation coded in GTScript, and note how similar the GTScript representation is to the mathematical notation:

.. image:: stencil.png

+--------------------------------------------------------+---------------------------------------+
| .. code-block:: python                                 | :math:`\begin{align}                  |
|                                                        | B_{i, j, k} = & - 4 A_{i, j, k} \\    |
|    @stencil(backend="gt:cpu_kfirst")                   | & + (A_{i+1, j, k} + A_{i-1, j, k} \\ |
|    def laplacian(A: Field[float], B: Field[float]):    | & + A_{i, i+1, k} + A_{i, j-1, k})    |
|        with computation(PARALLEL), interval(...):      | \end{align}`                          |
|            B = - 4. * A + ( A[I + 1] + A[I - 1] +      |                                       |
|                             A[J + 1] + A[J - 1] )      |                                       |
+--------------------------------------------------------+---------------------------------------+

The default performance backends use the `GridTools <https://github.com/GridTools/gridtools>`_ C++ library to target typical modern multi-core CPU and GPU architectures.
GridTools is a C++ framework developed by `CSCS <https://cscs.ch>`_ for writing performance-portable stencil operations.
In contrast to GT4Py it operates on a much lower level.

GT4Py can also generate easy-to-read numpy code that illustrates how the stencil parallelization works.
This also is easier for debugging, since the user can put breakpoints in this function.

It is also possible to generate backends for other performance portability libraries, or even hook into other frameworks at a deeper level.
As an example of this, GT4Py comes with built-in integration to the `DaCe <https://github.com/spcl/dace>`_ data-centric parallel programming library. 

GT4Py Features:

* High-Level:
  GT4Py takes care of common concerns, such as storage layout, reducing boiler plate code and allowing the intent of the code to shine through.

* Performant & Portable:
  GT4Py generates and compiles highly optimized code wherever GridTools is available.

* Familiar:
  GTScript is a `DSL <https://en.wikipedia.org/wiki/Domain-specific_language#Usage_patterns>`_ in Python and easy learn for anyone with Python experience.

* Versatile:
  GT4Py is not limited to GridTools as a backend for compiling optimized code but rather can be extended.
  GTScript code can also be used to generate optimized stencils that can be called from programs not written in Python (C++, Fortran, ...).


.. commented_out The :doc:`Quick Start Guide <quickstart>` contains some simple examples
.. commented_out of how to use GT4Py to define and run your own stencils. It is probably worth
.. commented_out to take a look there before going into the reference documentation.
   
.. toctree::
   :hidden:

   self

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   gtscript
   arrays
   lang_design
   commandline
   apiref
   indices
   license

.. commented_out gtscript
