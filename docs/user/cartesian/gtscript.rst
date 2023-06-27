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

Stencils can not call other stencils (this is a limitation in GT4Py's design), however commom computations can be shared as GT4Py functions.

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


By comparing to the version without a function above, we realize that functions are called within computation contexts and should be written accordingly.

--------------------
Computation Contexts
--------------------

A computation context describes a pass over 3-dimensional data stored in fields, which GT4Py can optimize for you. Depending on the type of context, inside we may:

  * **read and write** the current element of a field: :code:`some_field[0, 0, 0]` or :code:`some_field` (shorthand)
  
  * **read** another element on the same vertical plane (horizontal offset): :code:`some_field[-1, 5, 0]`

  * **read** another element on a lower plane (negative vertical offset) **special meaning in forward contexts** :code:`some_field[1, 2, -1]`

  * **read** another element on a higher plane (positive vertical offset) **special meaning in backward contexts** :code:`some_field[1, 2, 1]`


Vertical offsets have special meanings in forward or backward contexts, where the elements in the direction we come from are considered already processed. In forward contexts, this is the case for negative vertical offsets, in backward contexts for positive ones.

We create a context using the `computation` context manager. Since it is a GTScript builtin, we do not have to import it (in fact, it does not exist). It will be read and translated by GT4Py when the function is processed as GTScript.

.. code-block:: python
   :emphasize-lines: 3
   
   @gtscript.stencil
   def some_stencil(some_field: gtscript.Field[np.float64]):
      with computation(PARALLEL):
          with interval(...):
              ...
  

GT4Py is responsible for running it all for each element in the right order, parallelizing when possible. The machine code equivalent to a computation context would be a triple-nested for loop over each dimension.

GT4Py is of course also aware that computations that access offset values can not be run in certain margin areas of the data (where the offset values would not exist). For horizontal offsets, GTScript will automatically take care of the margins and leave them untouched. If we want to enforce horizontal boundary conditions, we require a separate stencil, to which we pass the boundarie's horizontal domain. For vertical offsets we can use computation :ref:`intervals<Interval Contexts>` to run different computations on the boundaries. This means we can "force" computations to run in the vertical margins where the offset elements do not exist. GTScript will fail to compile in those cases.

Parallel Contexts
-----------------

The name "parallel" comes from the idea that every vertical plane can be processed in parallel trivially. This is also why no vertical offsets are allowed.

.. code:: python

   import numpy as np
   
   from gt4py import gtscript

   @gtscript.stencil
   def laplacian(field: gtscript.Field[np.float64]):
      with computation(PARALLEL), interval(1, -1):
          field = field[-1, 0, 0] + field [1, 0, 0] + field[0, -1, 0] + field[0, 1, 0] + field[0, 0, -1] + field[0, 0, 1] - 8.0 * field[0, 0, 0]


In this example every offset element, including :code:`field[0, 0, -1]` and :code:`field[0, 0, 1]` refer to the values of these elements before the stencil is run.

Forward Contexts
----------------

The name "forward" describes the mental model of processing each plane in order from lowest vertical index to hightest.

.. code:: python

   import numpy as np
   
   from gt4py import gtscript

   @gtscript.stencil
   def cumsum(field: gtscript.Field[np.float64]):
      with computation(FORWARD), interval(1, None):
          field += field[0, 0, -1]


In this example, :code:`field[0, 0, -1]` refers to the value of the element directly below **after processing**. Each element of :code:`field` will afterwards contain a sum of all the elements previously in it's vertical column up to and including itself.

Backward Contexts
-----------------

The name "backward" describes the mental model of processing each plane in reverse order, from highest to lowest vertical index.

.. code:: python

   import numpy as np
   
   from gt4py import gtscript

   @gtscript.stencil
   def reversed_cumsum(field: gtscript.Field[np.float64]):
      with computation(BACKWARD), interval(0, -1):
          field += field[0, 0, 1]


In this example, :code:`field[0, 0, -1]` refers to the value of the element directly above **after processing**. Each element of :code:`field` will afterwards contain a sum of all the previous elements above it in it's vertical column down to and including itself.


.. _Interval Contexts:

-----------------
Interval Contexts
-----------------

Because it is very common to do different computations on different horizontal planes, depending on their vertical position, GTScript has vertical intervals built in:

.. code-block:: python
   :emphasize-lines: 8,10,12

   import numpy as np

   from gt4py import gtscript

   @gtscript.stencil
   def use_vertical_intervals(field: gtscript.Field[np.float64], updated_vertical_boundary: gtscript.Field[np.float64]):
      with computation(PARALLEL):
          with interval(0, 1):
              field = updated_vertical_boundary[0, 0, 0]
          with interval(1, -1):
              field = field[0, 0, -1] + field[0, 0, 1] - 2 * field[0, 0, 0]
          with interval(-1, None):
              field = updated_vertical_boundary[0, 0, 0]


The above stencil would update the bottom and top vertical level and at the same time compute vertical finite difference in between (using the old boundary values, since it is inside a parallel compute context)

Interval contexts may only occur inside compute contexts. They do not alter the vertical offset semantics of the given compute context they are in. The intervals inside a given compute context may not overlap.
