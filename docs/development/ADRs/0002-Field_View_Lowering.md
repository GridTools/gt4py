---
tags: [frontend]
---

# Field View Lowering

- **Status** deprecated
- **Authors** Rico Häuselmann (@dropd), Till Ehrengruber (@tehrengruber)
- **Created** 2022-02-09
- **Updated** 2023-03-22

## Updates:

### 2023-03-22:

The lowering was significantly simplified after the introduction of neighbor lists in Iterator IR.
The new lowering strategy is described in the lowering module.

### 2022-04-08: Added scalar literal support

The rationale that "any FOAST node should be lowered to an ITIR iterator expression" has changed to

> Any non-scalar FOAST Node should be lowered to an ITIR iterator expression. Scalar FOAST nodes should be lowered to an ITIR value expression.

#### Consequences

The lowering of parent nodes now has to deal with two kinds of expressions. In practice this means in some visitor methods, lowered child nodes have to be dereferenced if and only if they are iterator expressions. This is achieved via a `to_value` function, which takes a FOAST node and returns `deref` or a noop depending on the type of the FOAST node (which determines whether the node is lowered to iterator or value). The returned callable can then be used with the lowered node to achieve the desired effect.

Since the lowering currently does not have a usecase for lifting lowered child nodes if they are value expressions, the corresponding `to_iterator` method has been omitted. It can be introduced at a later time, if a usecase for it is found, it should be analogous to `to_value` and therefore easy to implement (in about four lines).

### 2022-02-22: Added reduction lowering

Lowering reductions is not completely straightforward (at the time of writing) because expressions inside a reduction in field view may contain fields shifted to neighbor dimensions.

These are lowered to partially shifted iterators which can not be dereferenced directly. The iterator builtin `reduce` therefore takes an expression which works on values after they have been shifted to a concrete neighbor and dereferenced by the system. Everywhere else it remains easier to lower everything to an iterator expression.

This leads to the following solution, likely subject to change when the behaviour of `reduce` or partially shifted iterators change:

The expression inside the reduction is lowered using a separate `NodeTranslator` instance which

1. collects shifted (and non-shifted) iterators in the course of the lowering
2. lowers everything under the assumption that everything is a value and
3. reinserts the (shifted) iterators as arguments to the `reduce` call.

This solution is proposed also because the special rules for `reduce`, which are likely to change are isolated in this separate lowering parser.

## Background

The lowering must convert from a function body in field view, which allows temporary assignment statements to a single expression iterator IR, which is gt4py.next.

Example (type annotations omitted):

```python
@fieldop
def temp(a):
  tmp = a
  return tmp
```

Would need to be turned into a single expression. While this case is trivial to solve by hand by simply replacing of `a` for `tmp` (yielding `deref(a)`), we require an algorithm that works in all cases.

## Algorithm Choice

### Guiding Principles

The choice of algorithm was guided by:

- Yield correct Iterator IR with as little special-casing as possible
- As simple and readable to code as possible
- Avoid passing information down into subtree visitors (as long as that does not clash with simplicity and readability)

### Algorithm

We chose the following algorithm:

1. Lower the return value expression into an **iterator expression** and store it in `current_expression`

2. For each assign statement in reverse order:

3. lower the right-hand side of the assign statement into an **iterator expression**

4. wrap `current_expression` in a let-expression (see below) that exposes the lowered right-hand side as the left-hand side.

5. dereference the `current_expression`.

Or in pseudocode:

```
current_expression <- lower(return_value_expr)
for assign in reversed(assigns):
  current_expression <- let assign.lhs_name = lower(assign.rhs) in current_expression

DEREF(current_expression)
```

The let expression `let VAR = INIT_FORM in FORM` written out in iterator view looks as follows:

```python
(lambda VAR: FORM)(INIT_FORM)
```

### Discussion

#### Avoids Subexpression Duplication

One property of this algorithm is that it does not duplicate subexpressions unnecessarily, unlike inlining in cases like the following (lifting of integer literals omitted, see below for more on that):

```python
@fieldop
def inline_duplication(a):
  tmp1 = a * 2
  tmp2 = tmp1 + 1
  return tmp1 + tmp2

@fundef
def inlined(a):
  return deref(lift(lambda a: plus(
    deref(lift(lambda a: mult(deref(a), 2))(a)),    # \
    deref(lift(lambda a: plus(                      #  }- duplicated
      deref(lift(lambda a: mult(deref(a), 2))(a)),  # /
      1
    ))(a))
  ))(a)
)

@fundef
def let_style(a):
  return deref(
      call(lambda tmp1:
        call(lambda tmp2: lift(lambda tmp1, tmp2: plus(deref(tmp1), deref(tmp2)))(tmp1, tmp2)
      )(lift(lambda tmp1: plus(deref(tmp1), 1))(tmp1)
    )(lift(lambda a: mult(deref(a), 2)))(a)  # <-- only occurs once
  )
```

This is fortunate, because at the time of writing, common subexpression elimination for iterator IR is not yet implemented and efforts for optimizing iterator IR can potentially focus on other areas.

#### Subexpression Lifting

A consequence of this algorithm is that all field view expressions must be lowered to iterator expressions. This may lead to some not strictly necessary lifting and dereferencing but is in line with the intuition that in field view every expression is a field expression (even scalar literals, which are not implemented yet at the time of writing).

Examples:

```
a + b -> lift(lambda a, b: plus(deref(a), deref(b)))(a, b)

# future consideration
a + 1 -> lift(lambda a: plus(deref(a), deref(lift(lambda: 1)())))(a)
```

One might be tempted to eliminate the `deref(lift(lambda: 1)())` as extraneous. However, the same thing could be rewritten as

```
tmp = 1
a + tmp -> (lambda tmp: lift(lambda a: plus(deref(a), deref(tmp))))((lift lambda: 1)())
```

Where the algorithm makes the assumption that every assignment target (or let variable) is an iterator expression.
This means, while the `deref(lift(...))` could be avoided in some cases, it would require special casing.
This would mean complicating the lowering without gaining correctness, and therefore contradicts our guiding principles.

## Iterator IR helpers

While implementing the lowering and specifically the tests for it, it quickly became clear that using the `iterator.ir` nodes directly to build trees and tree snippets leads to extremely verbose code. The structure of the patterns got lost in keyword arguments and `FunCalls` of `FunCalls`.

On the other hand **iterator** view code can represent the same tree or tree snippet much more readably with the drawback that there is no way of obtaining the `iterator.ir` nodes tree of such code, without executing it through a backend, which stores the tree as a side effect. Converting iterator IR to iterator view code was also considered but requires executing through a backend with code generation. Executing is not desirable because (a) it requires some boilerplate and (b) it does not allow comparing invalid snippets.

Therefore `gt4py.next.ffront.itir_makers` was written to allow direct building of iterator IR tree snippets with syntax that matches iterator view closely and makes the patterns visible. It allows implicit usage of string literals as variable names to increase readability wherever it is unambiguously possible. It does not check the validity of the built snippets by design.

Finally, the improvement in clarity is so striking that these makers are also used in the lowering itself.

## Deferred implementation of tuple returns

In the course of implementing the lowering it turned out that while it is clear what `return a, b` should do in field view, it is not clear how to achieve that by lowering to iterator IR (in general, not specifically with the chosen algorithm). Therefore the tests that return multiple fields have been skipped for now and the lowering makes no special effort to generate valid IR from tuple expressions as return values.

## Temporary variable renaming

Since temporary variables are no longer inlined, the renaming that happens in the SSA pass now goes through into the lowered IR, requiring the new names to be valid `SymbolNames`. This renaming should consequently be checked for and made more robust against user variable name collisions.

## Operator signature FOAST <-> ITIR

On iterator level the arguments to all stencils / functions used inside a fencil closure need to be iterators (due to the compiled backend using a single SID composite to pass the arguments). Following the FOAST value <-> ITIR value, FOAST field <-> ITIR iterator correspondence, all field operator arguments whose type on FOAST level is a value, i.e. scalar or composite thereof, are expected to be values on ITIR level by the rest of the lowering. As a consequence we transform all values into iterators before calling field operators (to satisfy the former constraint) and deref them immediately inside every field operator (to satisfy the latter constraint).
