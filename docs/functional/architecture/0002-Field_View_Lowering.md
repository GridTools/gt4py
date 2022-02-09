---
tags: [frontend]
---

# Field View Lowering

- **Status** valid
- **Authors** Rico Häuselmann (@dropd), Till Ehrengruber (@tehrengruber)
- **Created** 2022-02-09
- **Updated** 2022-02-09


## Background

The lowering must convert from a function body in field view, which allows temporary assignment statements to a single expression iterator IR, which is functional.

Example (type annotations omitted):

```python
@fieldop
def temp(a):
  tmp = a
  return tmp
```

Would need to be turned into a single expression. While this case is trivial to solve by hand by simply replacing of ``a`` for ``tmp`` (yielding `deref(a)`), we require
an algorithm that works in all cases.

## Algorithm Choice

### Guiding Principles

The choice of algorithm was guided by:

* Yield correct Iterator IR with as little special-casing as possible
* As simple and readable to code as possible
* Avoid passing information down into subtree visitors (as long as that does not clash with simplicity and redability)

### Algorithm

We chose the following algorithm:

1. Lower the return value expression into an **iterator expression** and store it in ``current_expression``

2. For each assign statement in reverse order:

  1. lower the right-hand side of the assign statement into an **iterator expression**

  2. wrap ``current_expression`` in a let-expression (see below) that exposes the lowered right-hand side as the left-hand side.

3. dereference the ``current_expression``.

Or in pseudocode:

```
current_expression <- lower(return_value_expr)
for assign in reversed(assigns):
  current_expression <- let assign.lhs_name = lower(assign.rhs) in current_expression

DEREF(current_expression)
```

The let expression ``let VAR = INIT_FORM in FORM`` written out in iterator view looks as follows:

```python
(lambda VAR: FORM)(INIT_FORM)
```

### Discussion

#### Avoids Subexpression Duplication

One property of this algorithm is that it does not duplicate subexpressions unneccessarily, unlike inlining in cases like the following:

```python
@fieldop
def inline_duplication(a):
  tmp1 = a * 2
  tmp2 = tmp1 + 1
  return tmp1 + tmp2

@fundef
def inlined(a):
  return plus(
    mult(deref(a), 2),    # \
    plus(                 #  }- duplicated
      mult(deref(a), 2),  # /
      1
    )
  )
```

This was not a major goal but a happy accident, since it is expected that a common subexpression elimination optimization will be run on the Iterator IR anyway.

#### Subexpression Lifting

A consequence of this algorithm is that all field view expressions must be lowered to iterator expressions. This may lead to some not strictly necessary
lifting and dereferencing but is in line with the intuition that in field view every expression is a field expression (even scalar literals, which are not implemented yet at the time of writing).

Examples:

```
a + b -> lift(lambda a, b: plus(deref(a), deref(b)))(a, b)

# future consideration
a + 1 -> lift(lambda a: plus(deref(a), deref(lift(lambda: 1)())))(a)
```

One might be tempted to eliminate the ``deref(lift(lambda: 1)())`` as extraneous. However, the same thing could be rewritten as

```
tmp = 1
a + tmp -> (lambda tmp: lift(lambda a: plus(deref(a), deref(tmp))))((lift lambda: 1)())
```

Where the algorithm makes the assumption that every assignment target (or let variable) is an iterator expression.
This means, while the ``deref(lift(...))`` could be avoided in some cases, it would require special casing.
This would mean complicating the lowering without gaining correctness, and therefore contradicts our guiding principles.
