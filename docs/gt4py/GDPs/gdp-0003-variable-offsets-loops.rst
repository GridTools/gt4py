==================================
GDP 3 — Variable Offsets and Loops
==================================

:Author: Johann Dahm <johannd@vulcan.com> - Vulcan Climate Modeling
:Status: Draft
:Type: Feature
:Created: 21-05-2021
:Discussion PR: `TBD <discussion_pr>`_
:Implementation: `TBD <impl_pr>`_


Abstract
--------

GT4Py has limited control flow in the form of if-then conditionals, based on either compile- or run-time values.
This GDP extends this capability by additionally supporting a limited set of loops, which arise in remapping a Lagrangian coordinate in FV3.

In order to make the control flow usable, this also extends the field offset to allow the vertical to be a general expression evaluating to an integer type, that could involve fields or parameters.

In order to index vertical loops, we also give access to the current vertical index only in the context of defining loop bounds.


Motivation and Scope
--------------------

While loops
+++++++++++

The FV3 dynamical core has a step in which it remaps a Lagrangian coordinate back to a regular grid.
This portion of the code reduces sets of adjacent levels, accumulating the results of a field at these levels into an output field.
Since there are one or more levels being summed, this is a natural fit for a while loop with a termination condition.

.. code-block:: python

    def vertical_reduction(
      heights: Field[float],
      qin: Field[float],
      qout: Field[float],
      k_index: Field[IJ, int],
      next_height: Field[IJ, float],
      dz: float
    ):
      with computation(FORWARD), interval(...):
        next_height += dz
        while current_height < next_height:
          qout += qin[0, 0, k_index]
          current_height += heights[0, 0, k_index]
          k_index += 1

This example shows a vertical reduction, where the statements inside the while loop execute some number of times.
An integer field is used to track the vertical index, which is increased by one each time through the loop.

The variable vertical offset is treated as a regular offset, so cannot be given to field accesses on the left side of assignments.

There is another case where loops are necessary from the top or bottom to the current vertical level.
In this case, using a ``while`` loop, one would write:

.. code-block:: python
    # k_index: Field[IJ, float] argument
    while k_index < index(K):
      # ...
      k_index += 1

However exposing ``index(K)`` as a magical built-in gtscript function to get the current index in general increases the scope of the DSL too far.
Instead, we propose adding a ``for`` loop concept that can use ``index(K)`` only within the scope of defining the loop bounds, and only on the vertical index.

For loops
+++++++++

The syntax for for loops follows that from Python: there is a target and an iterator to loop over.
The iterator can one of:

1. The slice of the ``K`` axis array -- or the entire array to iterate over the entire vertical.
2. A ``range`` call, which can have arguments that are literal integers, or expressions involving ``index(K)`` -- the current vertical index.

To motivate this, here are two different syntaxes for a vertical ``scan`` operation:

.. code-block::
    with computation(FORWARD), interval(...):
      for index in range(0, index(K)):
        field += other[0, 0, index]

The alternative using the ``K`` axis object is:

.. code-block::
    with computation(FORWARD), interval(...):
      for index in K[:index(K)]:
        field += other[0, 0, index]


Usage and Impact
----------------

Additional frontend feature and IR nodes, but no impact to existing code.


Backward Compatibility
----------------------

Introduces a new feature.


Detailed Description
--------------------

- Note that integer index variables need to be declared and initialized outside the stencils because they are 2D, and declaring them inside would make them automatically 3D (or at least behave as if they were a 3D field).


Implementation
--------------


FV3 Example
-----------


.. code-block:: python

    def lagrangian_contributions(
        q: FloatField,
        pe1: FloatField,
        pe2: FloatField,
        q4_1: FloatField,
        q4_2: FloatField,
        q4_3: FloatField,
        q4_4: FloatField,
        dp1: FloatField,
        lev: IntFieldIJ,
    ):
        with computation(FORWARD), interval(...):
            v_pe2 = pe2
            v_pe1 = pe1[0, 0, lev]
            pl = (v_pe2 - v_pe1) / dp1[0, 0, lev]
            if pe2[0, 0, 1] <= pe1[0, 0, lev + 1]:
                pr = (pe2[0, 0, 1] - v_pe1) / dp1[0, 0, lev]
                q = (
                    q4_2[0, 0, lev]
                    + 0.5
                    * (q4_4[0, 0, lev] + q4_3[0, 0, lev] - q4_2[0, 0, lev])
                    * (pr + pl)
                    - q4_4[0, 0, lev] * 1.0 / 3.0 * (pr * (pr + pl) + pl * pl)
                )
            else:
                qsum = (pe1[0, 0, lev + 1] - pe2) * (
                    q4_2[0, 0, lev]
                    + 0.5
                    * (q4_4[0, 0, lev] + q4_3[0, 0, lev] - q4_2[0, 0, lev])
                    * (1.0 + pl)
                    - q4_4[0, 0, lev] * 1.0 / 3.0 * (1.0 + pl * (1.0 + pl))
                )
                lev = lev + 1
                while pe1[0, 0, lev + 1] < pe2[0, 0, 1]:
                    qsum += dp1[0, 0, lev] * q4_1[0, 0, lev]
                    lev = lev + 1
                dp = pe2[0, 0, 1] - pe1[0, 0, lev]
                esl = dp / dp1[0, 0, lev]
                qsum += dp * (
                    q4_2[0, 0, lev]
                    + 0.5
                    * esl
                    * (
                        q4_3[0, 0, lev]
                        - q4_2[0, 0, lev]
                        + q4_4[0, 0, lev] * (1.0 - (2.0 / 3.0) * esl)
                    )
                )
                q = qsum / (pe2[0, 0, 1] - pe2)
            lev = lev - 1


Copyright
---------

This document has been placed in the public domain.
