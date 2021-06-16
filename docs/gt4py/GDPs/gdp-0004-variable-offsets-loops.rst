==================================
GDP 4 — Loops and Variable Offsets
==================================

:Author: Johann Dahm <johannd@vulcan.com> - Vulcan Climate Modeling
:Status: Draft
:Type: Feature
:Created: 21-05-2021
:Discussion PR: `https://github.com/GridTools/gt4py/pull/426 <discussion_pr>`_
:Implementation: TBD


Abstract
--------

GT4Py has limited control flow in the form of if-then conditionals, based on either compile- or run-time values.
This GDP extends this capability by additionally supporting a limited set of loops, which arise in remapping a Lagrangian coordinate and looping over extra data dimensional fields.

Loops bounds can be any scalar integer (parameter or compile-time literal) or can be related to the vertical axis, in which case the bounds are unknown until run-time.
There are also patterns that contain loops from the top or bottom of the vertical to the current vertical index (or an expression involving the vertical index).

Also introduced here are *variable offsets*, related to *index fields*, which are fields (usually IJ) that can be used to index the vertical direction. In order to limit the scope, these are not allowed to be used to index the horizontal dimensions.


Motivation and Scope
--------------------

We have encountered the use for variable offsets and while loops in the Lagrangian remapping in FV3, and ICON seems to have a similar remapping pattern as well.

In the physics routines, there are vertical reduction patterns, as well as loops over higher dimensional axes.

The example below shows an inner loop over vertical levels in cloud microphysics:

.. code-block:: fortran

    do k = kbot - 1, k0, - 1
      do m = k + 1, kbot
        if (zt (k + 1) >= ze (m)) exit
          ! ...
          tz (m) = tz (m) - sink * icpk (m)
          qi (k) = qi (k) - sink * dp (m) / dp (k)
        endif
      enddo
    enddo

The example below in radiation shows a pattern of looping over higher dimesnional axes with the need to reference based on current higher dimension index:
.. code-block:: fortran

    do k = 1, nlay
      ! ...
      do ib = 1, nbdsw
        jb = nblow + ib - 1
        taucw(k,ib) = tauliq(jb)+tauice(jb)+tauran+tausnw
        ssacw(k,ib) = ssaliq(jb)+ssaice(jb)+ssaran(jb)+ssasnw(jb)
        asycw(k,ib) = asyliq(jb)+asyice(jb)+asyran(jb)+asysnw(jb)
      enddo
    enddo


Detailed Description
--------------------

For loops
+++++++++

The syntax for for loops follows that from Python: there is a target and an iterator to loop over.
The iterator can one of:

1. The slice of the ``K`` axis array -- or the entire array to iterate over the entire vertical.
2. A ``range`` call, which can have arguments that are literal integers, or expressions involving ``index(K)`` -- the current vertical index.

To motivate this, here are two different syntaxes for a vertical ``scan`` operation:

.. code-block::

    with computation(FORWARD), interval(...):
      for index in range(0, index(K) - 1):
        field += other[0, 0, index]

The alternative using the ``K`` axis object is:

.. code-block::

    with computation(FORWARD), interval(...):
      for index in K[:index(K)]:
        field += other[0, 0, index]


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

# However exposing ``index(K)`` as a magical built-in gtscript function to get the current index in general increases the scope of the DSL too far.
# Instead, we propose adding a ``for`` loop concept that can use ``index(K)`` only within the scope of defining the loop bounds, and only on the vertical index.

- Note that integer index variables need to be declared and initialized outside the stencils because they are 2D, and declaring them inside would make them automatically 3D (or at least behave as if they were a 3D field).


Implementation
--------------

Frontend
++++++++

The gtscript frontend requires changes to allow parsing of the ``for`` and ``while`` statements.
These are also Python builtins, so it is easy to parse with an additional visitor method.

IR concepts
+++++++++++

As with ``if-else`` control flow, there need to be nodes in the definition IR/gtir that represent the bounds and condition, as well as the body of the statement (another list of statements).

The following implementation of ``For`` would be added to ``common.py``:

.. code-block:: python

    class For(GenericNode, Generic[StmtT, ExprT]):
      target: Str
      start: Union[ExprT, AxisBound]
      end: Union[ExprT, AxisBound]
      step: Int
      body: StmtT

From this implementation, each level of the IRs can simply inherit the implementation, but use their own ``Stmt`` and ``Expr`` types.

Since bounds for the for loops can depend on the size of the vertical axis, or the current index, these concepts need to be added to the IRs as well.

Backends
++++++++

The backends will need to
- generate code for the loops
- generate references to the current vertical index and vertical axis size.


Alternatives
------------

For loops (reductions) are a core feature that are required in multiple places for model development.
On the other hand, while loops seem to occur only in the Lagrangian remapping portion, but are common to multiple models (FV3, ICON).
Given this, if either feature were to be omitted, it would be possible to omit while loops and use a custom kernel or another DSL for the Lagrangian remapping.


FV3 Examples
------------

Lagrangian remapping
++++++++++++++++++++

The following shows the Lagrangian remapping written using the gt4py while loop and variable offsets.
Using this reduces what was 80*3 stencils to a single stencil with a single stage.
The resulting speedup was immense!

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
