===========================
GTScript Language Reference
===========================

This document describes the GTScript language and will be most useful to those implementing stencils in GTScript.

--------
Stencils
--------

Stencils are the entry points for GTScript. All of GTScript is valid inside functions decorated with :code:`gt4py.gtscript.stencil`. Take the following example:

.. code:: python

   import numpy as np

   from gt4py import gtscript

   @gtscript.stencil
   def field_plus_one(source: gtscript.Field[np.float64], target: gtscript.Field[np.float64]):
      with computation(PARALLEL), interval(...):
          target = source[0, 0, 0] + 1


When we decorate a function with :code:`gt4py.gtscript.stencil`, we are marking it to be interpreted as written in GTScript. This causes GT4Py to parse it into something on which it can apply stencil-specific optimizations. It can then hand the optimized version off to a backend, which can apply further optimizations (for example hardware-specific ones) and finally turn their further optimized version into something Python can execute.

More about how to use :code:`gt4py.gtscript.stencil` can be found in the :doc:`Quick Start Guide <quickstart>`

---------
Functions
---------

Stencils can not call other stencils (this is a limitation in GT4Py's design), however commom computations can be shared as GT4Py Functions.

Functions can also contain some GTScript code, but they are only meaningful when called from a Stencil. Think of them like common GTScript snippets that can be reused in Stencils. The difference is that functions are not allowed to contain the following elements of GTScript:

  * :code:`from __externals__ import`

  * :code:`computation` 

  * :code:`interval`

  * in-place modification of input variables

As opposed to Stencils, they must return their results. Don't worry however, this does not imply that data gets copied around unnecessarily when we run the final result. GT4Py will see to that.

For added clarity, let's rephrase the above example using a function:

.. code:: python

   import numpy as np

   from gt4py import gtscript

   @gtscript.stencil
   def field_plus_one(source: gtscript.Field[np.float64], target: gtscript.Field[np.float64]):
      with computation(PARALLEL), interval(...):
          target = plus_one(source)


   @gtscript.function
   def plus_one(field: gtscript.Field[np.float64]):
      return field[0, 0, 0] + 1
