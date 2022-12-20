GTScript Language Design
========================

The following principles are a guideline for designing the GTScript DSL. We try to follow these principles if we can. In some cases we cannot fulfill all principles and a trade-off has to be made and justified.

The principles are not magic, they mainly summarize the obvious.

Trivia: GTScript is an embedded DSL in Python, therefore language syntax is restricted to valid Python syntax.

1. Language constructs should behave the same as their equivalent in other languages, especially as equivalent concepts in Python or well-known Python libraries (e.g. NumPy).
   
   Motivation: The DSL should be readable by applying common sense and common programming language knowledge.

2. Semantic differences should be reflected in syntactic differences.
   
   Motivation: Spotting semantic differences is much harder than spotting syntactic differences.

3. Regular use-cases should be simple, special cases can be complex.
   
   Motivation: If a trade-off has to be made, the most common, standard use-cases should be expressed in the simplest possible way. To cover all cases, corner cases might require more complex language constructs.

4. Language constructs are required to have an unambiguous translation to parallel code and need to allow translation to efficient code in the regular use-cases.
   
   Motivation: When translating DSL to executable code, we must not make correctness errors, therefore we cannot allow ambiguous language constructs. If we fail,
   
   - the user will run into hard to debug problems,
   
   - the toolchain developer cannot reason about the code and will fail in writing correct optimizations.
   
   On purpose, performance is second and, on purpose, the requirement to produce efficient code is restricted to regular use-cases. Obviously, for a performance portable language, the regular use-cases are required to have an efficient translation. But this principle acknowledges that we cannot exclude that for some special cases an efficient translation cannot be found.

Parallel Model
--------------

The iteration domain is a 3d domain: ``I`` and ``J`` axes live on the horizontal spatial plane, and axis ``K`` represents the vertical spatial dimension. Computations on the horizontal plane are always executed in parallel and thus ``I`` and ``J`` are called parallel axes, while computations on K are executed sequentially and thus ``K`` is called a sequential axis.

A ``gtscript.stencil`` is composed of one or more ``computation``. Each ``computation`` defines an iteration policy ``(FORWARD, BACKWARD)`` and is itself composed of one or more non-overlapping vertical ``interval`` specifications, each one of them representing a vertical loop over the ``K`` axis with the iteration policy of the computation. Intervals are specified in their order of execution with each interval containing one or more statements.

The effect of the program is as if statements are executed as follows:

1. computations are executed sequentially in the order they appear in the code,
2. vertical intervals are executed sequentially in the order defined by the iteration policy of the computation
3. every vertical interval is executed as a sequential for-loop over the ``K``-range following the order defined by the iteration policy,
4. within a stencil, it is illegal to assign to an external field (or aliases pointing to the same memory location) if it is also read with horizontal offset in any expression that is (transitively) used to compute the r.h.s. of the assignment,
5. for ``if-else`` statements, the condition is evaluated first, then the ``if`` and ``else`` bodies are evaluated with the same rules as above,
6. execution of a program is illegal if any field access in any branch is outside of array bounds.

Examples
^^^^^^^^

In the following, the code snippets are not always complete GTScript snippets, instead parts are omitted (e.g. by ...) to highlight the important parts. The domain is defined by the intervals ``[i,I]``, ``[j,J]``, ``[k,K]``.

In the following, ``k <= K``,

Rule 4
""""""

The following cases are forbidden:

Write after read with offset

.. code:: python

    with computation(FORWARD)
        with interval(...):
            b = a[1,1,0]
            a = 0.

Shifted self-assignment

.. code:: python

    with computation(FORWARD):
        with interval(...):
            a = a[1,1,0]

Shifted self-assignment with temporary

.. code:: python

   with computation(PARALLEL):
       with interval(...):
           tmp = a
   with computation(PARALLEL):
       with interval(...):
           a = tmp[1,1,0]

These cases are forbidden as, in general, there is no efficient mapping to a blocked execution.

no specific loop order in k
"""""""""""""""""""""""""""

.. code:: python

    with computation(...):
        with interval(k, K):
            a = tmp[1, 1, 0]
            b = 2 * a[0, 0, 0]

behaves like

.. code:: python

    # NumPy semantics
    a[i:I, j:J, k:K] = tmp[i+1:I+1, j+1:J+1, k:K]
    b[i:I, j:J, k:K] = 2 * a[i:I, j:J, k:K]

forward iteration in k
""""""""""""""""""""""

.. code:: python

    with computation(FORWARD):
        with interval(k, K):
            a = tmp[1, 1, 0]
            b = 2 * a[1, 1, 0]

behaves like

.. code:: python

    for k_ in range(k, K):
        a[i:I+1, j:J+1, k_] = tmp[i+1:I+2, j+1:J+2, k_] # extended compute domain
        b[i:I, j:J, k_] = 2 * a[i+1:I+1, j+1:J+1, k_]

backward computation in k with interval specialization
""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. code:: python

    with computation(BACKWARD):
        with interval(k, -2): # lower interval
            a = tmp[1, 1, 0]
            b = 2 * a[0, 0, 0]
        with interval(-2, K): # upper interval
            a = 1.1
            b = 2.2

behaves like

.. code:: python

    for k_ in reversed(range(K-2, K)): # upper interval
        a[i:I, j:J, k_] = 1.1
        b[i:I, j:J, k_] = 2.2

    for k_ in reversed(range(k, K-2)): # lower interval
        a[i:I, j:J, k_] = tmp[i+1:I+1, j+1:J+1, k_]
        b[i:I, j:J, k_] = 2 * a[i:I, j:J, k_]

Note that intervals where exchanged to match the loop order.

Variable Declarations
---------------------

Variable declarations inside a computation are interpreted as temporary field declarations spanning the actual computation domain of the ``computation`` where they are defined.

Example
^^^^^^^

.. code:: python

    with computation(FORWARD):
        with interval(1, 3):
            tmp = 3

behaves like:

.. code:: python

    tmp = Field(domain_shape)  # Uninitialized field (random data)
    for k_ in range(0, 3):
        tmp[i:I, j:J, k_] = 3   # Only this vertical range is properly initialized

Compute Domain
--------------

The computation domain of every statement is extended to ensure that any required data to execute all stencil statements on the compute domain is present.

Example
^^^^^^^

On an applied example, this means:

.. code:: python

    with computation(...), interval(...):
        u = 1
        b = u[-2, 0, 0] + u[1, 0, 0] + u[0, -1, 0] + u[0, -2, 0]

translates into the following pseudo code:

.. code:: python

    for k_ in range(k, K):
        u[i-2:J+1, j-2:J, k_] = 1
        b[i:I, j:J, k_] = u[i-2:I-2, j:J, k_] + u[i+1:I+1, j:J, k_] + u[i:I, j-1:J-1, k_] + u[i:I, j-2:J-2, k_]

Conditionals
------------

GTScript supports 2 kinds of conditionals:

- conditionals on scalar expressions
- conditionals on field expressions

Conditionals on scalar expressions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Each statement inside the if and else branches is executed according to the same rules as statements outside of branches.
- There is no restriction on the body of the statement.

Example for scalar conditions
"""""""""""""""""""""""""""""

.. code:: python

    with computation() with interval(...):
        if my_config_var:
            a = 1
            b = 2
        else:
            a = 2
            b = 1

translates to:

.. code:: python

    for k_ in range(k, K):
        parfor ij:
            if my_config_var:
                a[i, j, k_] = 1
        parfor ij:
            if my_config_var:
                b[i, j, k_] = 2
        parfor ij:
            if not my_config_var:
                a[i, j, k_] = 2
        parfor ij:
            if not my_config_var:
                b[i, j, k_] = 1

Conditionals on field expressions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- The condition is evaluated for all gridpoints and stored in a mask.
- Each statement inside the if and else branches is executed according to the same rules as statements outside of branches.

Example for conditionals on field expressions
"""""""""""""""""""""""""""""""""""""""""""""

.. code:: python

    with computation():
        with interval(...):
            if field:
                a = 1
                b = 2
            else:
                a = 2
                b = 1

translates to:

.. code:: python

    for k_ in range(k, K):
        parfor ij:
            mask[i, j] = (field[i, j, k_] != 0)
        parfor ij:
            if mask[i, j]:
                a[i, j, k_] = 1
        parfor ij:
            if mask[i, j]:
                b[i, j, k_] = 2
        parfor ij:
            if not mask[i, j]:
                a[i, j, k_] = 2
        parfor ij:
            if not mask[i, j]:
                b[i, j, k_] = 1

or in Numpy notation

.. code:: python

    for k_ in range(k, K):
        mask = field[:, :, k_] != 0
        a[:, :, k_] = np.where(mask, 1, a[:, :, k_])
        b[:, :, k_] = np.where(mask, 2, b[:, :, k_])
        a[:, :, k_] = np.where(~mask, 2, a[:, :, k_])
        b[:, :, k_] = np.where(~mask, 1, b[:, :, k_])

(if and else branch is on purpose not written as a single ``np.where``).
The following cases are illegal:

.. code:: python

    with computation():
        with interval(...):
            if field:
                b = a[1, 0, 0] # read with offset in 'I' from updated field 'a'
                a = 1
                c = a[0, 1, 0] # read with offset in 'J' from updated field 'a'

    with computation():
        with interval(...):
            if field:
                a = 1
            else:
                b = a[1, 0, 0] # read with offset in 'I' from updated field 'a'

    with computation(...):
        with interval(...):
            if field:
                a = a[0, 1, 0] # self assignment with offset (i.e. a read with offset and write)

Loops
-----

While
^^^^^

GTScript has limited support for while loops, which iterate a set of statements nested inside it a number of times until all IJ points fail the condition. Note that this means certain IJ indices could execute the statements a different number of iterations. The syntax is:

.. code:: python

    with computation(FORWARD), interval(...):
        while a < b:
            c += 1
            a += 1

This translates to

.. code:: python

    for k_ in range(k, K):
        parfor ij:
            mask[i, j] = (a[i, j, k_] < b[i, j, k_])
        while any(mask):
            parfor ij:
                if mask[i, j]:
                    c[i, j, k_] += 1
            parfor ij:
                if mask[i, j]:
                    a[i, j, k_] += 1
            parfor ij:
                mask[i, j] = (a[i, j, k_] < b[i, j, k_])

however due to the blocking model used, there is no way to enforce synchronizations between the nested statements and the mask update. This is a subtle but important point to remember when writing while loops. The final parallel model behaves as

.. code:: python

    for k_ in range(k, K):
        parfor ij:
            mask[i, j] = (a[i, j, k_] < b[i, j, k_])
        while any(mask):
            parfor ij:
                if mask[i, j]:
                    c[i, j, k_] += 1
                    a[i, j, k_] += 1
                mask[i, j] = (a[i, j, k_] < b[i, j, k_])

The conclusion from this is that the user is not allowed to write to fields in the body of the while loop that are used in the mask or elsewhere in the body with a horizontal offset. The gtscript frontend implemented in gt4py contains checks to ensure that a user cannot write code incompatible with this restriction.
