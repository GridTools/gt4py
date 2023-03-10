---
tags: [backend]
---

# C++ Backend

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2022-02-18
- **Updated**: 2022-04-22

Overview of C++ backend decisions.

## General

The backend will not be able to transform the IR, therefore any patterns that need IR transformations need to happen before C++ code generation.

The following built-ins require to be presented in a canonicalized form

- `scan`
- `lift` if interpreted as temporary (see below).

**The implication is that the C++ backend can only support a subset of the Iterator IR specification.**

## Decisions

### GTFN IR

We introduce an IR which is close to the generated code.
It is different to Iterator IR in the following:

- no `lift` built-in
- allows allocation of temporaries
- Iterator IR built-ins are replaced by the corresponding C++ operator, e.g. `plus(a,b)` -> `a+b`

### Scan

Scan need to be canonicalized for the lowering to GTFN IR. The canonicalized form will be described elsewhere.

### Lift

In the backend, `lift` can only be interpreted as inlined as the interpretation as temporary requires transformation. As there can only be a single implementation in the C++ backend, we simplify by not allowing **any** `lift`s to be present in the IR for the C++ backend. This decision can be revised later.

### Reduce

Additionally, we decided to unroll the reduction, before generating the backend code. Reductions are mainly syntax sugar. There is no clear performance implication. Additionally this simplifies the backend implementation and removes the complication of partial shifts in the backend. This decision can be revised later.

These are reasons why reductions might be introduced in the backend at some point:

1. Performance implications for check for skip values.

   If we use the reduce builtin, we can implement an early exit in the reduction loop, assuming there are no non-skipped values after the first skip value. This assumption might be problematic anyway as we might have to reorder neighbor tables, if the mesh generator does not comply with this assumption.

2. Implementation as runtime loops.

   If we would implement the reduction as runtime loops (with compile time bounds), we could leave the decision of unrolling to the C++ compiler. Note that this would require to implement `shift`s with runtime offsets (currently offsets are required to be compile-time).

Note: Originally, we introduced `reduce` to be able to check for skip values only once for all iterators in the reduce. However the single check can be implemented in the unrolled version, too.

Expand `reduce(lambda acc, a, b: acc+a*b,0)` to

```python
def unrolled_reduce(a,b):
    acc_init = 0

    shifted_a_0 = shift(0)(a)
    shifted_b_0 = shift(0)(b)
    acc_0 = if_(can_deref(shifted_a_0), acc_init+deref(shifted_a_0)*deref(shifted_b_0), acc_init)

    shifted_a_1 = shift(1)(a)
    shifted_b_1 = shift(1)(b)
    acc_1 = if_(can_deref(shifted_a_1), acc_0+deref(shifted_a_1)*deref(shifted_b_1), acc_0)

    // ...
```

## Summary

The IR, when presented to the C++ backend must have the following properties:

- `scan` canonicalized
- `lift` that are interpreted as temporaries are canonicalized, no other (inline) `lift`s
- no `reduce`
