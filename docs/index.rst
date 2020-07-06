
===========================================================
GT4Py: GridTools for Python
===========================================================

`GridTools <https://github.com/GridTools/gridtools>`_ allows you to write
highly performant :emphasis:`and` portable :emphasis:`declarative` stencil 
kernels for weather and climate simulations in C++.  Write closer to the 
mathematical formulation and let the framework optimize performance for you.

GT4Py takes it a step further: kernels written in GT4Py's GTScript language are
even more concise and readable. Plus, being an extension of Python syntax,
GTScript is easily readable for anyone who knows some Python.

Great! - so what does it look like in practice? Look at the following example of
a laplacian operation coded in GTScript!

.. image:: stencil.png

Note how similar the GTScript representation is to the mathematical notation:

+-------------------------------------------------------+---------------------------------------+
| .. code-block:: python                                | :math:`\begin{align}                  |
|                                                       | B_{i, j, k} = & - 4 A_{i, j, k} \\    |
|    @stencil(backend="gtcuda")                         | & + (A_{i+1, j, k} + A_{i-1, j, k} \\ |
|    def laplacian(A: Field[float], B: Field[float]):   | & + A_{i, i+1, k} + A_{i, j-1, k})    |
|        with computation(PARALLEL), interval(...):     | \end{align}`                          |
|            B = - 4 * A + ( A[1, 0, 0] + A[-1, 0, 0] + |                                       |
|                            A[0, 1, 0] + A[0, -1, 0] ) |                                       |
+-------------------------------------------------------+---------------------------------------+

Features:

* **High-Level**:
  GT4Py takes care of common concerns, such as storage layout, reducing
  boiler plate code and allowing the intent of the code to shine through

* **Performant & Portable**: 
  GT4Py generates and compiles highly optimized code wherever GridTools is
  available

* **Familiar**: 
  GTScript is an `embedded DSL
  <https://en.wikipedia.org/wiki/Domain-specific_language#Usage_patterns>`_ in
  Python and easy to grasp for anyone with python experience

* **Versatile**: 
  GT4Py is not limited to GridTools as a backend for compiling optimized code
  but rather can be extended. GTScript code can also be used to generate
  optimized stencils that can be called from programs not written in Python
  (C++, Fortran, ...)


.. commented_out The :doc:`Quick Start Guide <quickstart>` contains some simple examples
.. commented_out of how to use GT4Py to define and run your own stencils. It is probably worth
.. commented_out to take a look there before going into the reference documentation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   apiref
   indices
   license

   GDPs/gdp-index

.. commented_out gtscript
