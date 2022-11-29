===========================
GTscript Language Reference
===========================

This document describes the GTScript language and will be most useful to those implementing stencils in GTScript.

--------
Stencils
--------

Stencils are the entry points for GTScript. All of GTScript is valid inside functions decorated with ``@gtscript.stencil``. Take the following example:

.. code:: python

   import numpy as np

   from gt4py import gtscript

   @gtscript.stencil
   def copy_field(source: gtscript.Field[np.float64], target: gtscript.Field[np.float64]):
      with computation(PARALLEL), interval(...):
          target = source[0, 0, 0]


When we decorate a function with ``@gtscript.stencil``, we are marking it to be interpreted as written in GTScript. This causes GT4Py to parse it into something on which it can apply stencil-specific optimizations. It can then hand the optimized version off to a backend, which can apply further optimizations (for example hardware-specific ones) and finally turn their further optimized version into something Python can execute.

GTScript developers can exert a certain amount of control on backends, however that is covered elsewhere.

.. TODO: find and reference where that is covered
