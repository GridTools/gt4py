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

   $ gtpyc gen my_stencils.gt.py --backend=gtx86 --output-path=./my-stencils-out

Will generate stencil code for each stencil and will write the following files:

.. code-block:: bash

   $ tree .my-stencils-out/
   my-stencils-out
   ├── stencil_a/
   │   ├── computation.cpp
   │   └── computation.hpp
   └── stencil_b/
       ├── computation.cpp
       └── computation.hpp
