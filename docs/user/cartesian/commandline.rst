Commandline
===========

GT4Py installs the `gtpyc` command (introduced in
:doc:`GDPs/gdp-0001-standalone-cli`). Implementation is ongoing.

Usage
-----

List available backends
+++++++++++++++++++++++

.. code-block:: bash

   $ gtpyc list-backends

               computation    bindings        CLI-enabled
   ---------   -----------    --------        -------------
   <backend>   <lang>         <lang>, <lang>  <Yes | No>
   ...

Lists the currently implemented backends with computation language and possible
language bindings.  The last column informs whether CLI support was implemented
for the backend.

Generate stencils from a GTScript module
++++++++++++++++++++++++++++++++++++++++

Assume the following file structure:

.. code-block:: bash

   $ tree .
   pwd
   ├── constants.py
   └── stencils.gt.py

``stencils.gt.py`` contains the GTScript code to be compiled to stencils. The contents might look
something like the following example.

.. code-block:: python
   :caption: my_stencils.gt.py

   # [GT] using-dsl: gtscript

   from numpy import float64

   from .constants import PI


   @function
   def square(inp_field):
      return inp_field * inp_field


   @lazy_stencil
   def stencil_a(inp_field: Field[float64], out_field: Field[float64]):
      with computation(PARALLEL), interval(...):
         out_field = square(inp_field)


   @lazy_stencil
   def stencil_b(inp_field: Field[float64], out_field: Field[float64]):
      from __externals__ import COMPILE_TIME_VALUE
      with computation(PARALLEL), interval(...):
         out_field = PI * inp_field + COMPILE_TIME_VALUE

Then

.. code-block:: bash

   $ gtpyc gen my_stencils.gt.py --backend=gt:cpu_ifirst --output-path=./my-stencils-out

Will generate the C++ stencil code files using the "gt:cpu_ifirst" backend for each
stencil in the global namespace of ``gstencils.gt.py`` (also ones imported from
other modules) and output them inside ``my-stencils-out/``. The code files for
each stencil are put in a subfolder named after the stencil:

.. code-block:: bash

   $ tree .my-stencils-out/
   my-stencils-out
   ├── stencil_a/
   │   ├── computation.cpp
   │   └── computation.hpp
   └── stencil_b/
       ├── computation.cpp
       └── computation.hpp

The line

.. code-block:: python

   # [GT] using-dsl: gtscript

in ``my_stencils.gt.py`` is equivalent to writing

.. code-block:: python

   from gt4py.cartesian.gtscript import *

in a python module. It makes all symbols from the `gt4py.gtscript` module available.
