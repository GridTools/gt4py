# Behavior of the `round()` function in GT4Py

In the context of rounding modes, facing disagreeing standard implementations in python and C++ backends, we decided to override python's default rounding to achieve agreement in rounding behavior for all backends. We considered to implement the python standard in C++ and accept a performance penalty in the `numpy` backend.

## Context

The default in python (since version 3) is to split ties by rounding to the nearest even value (known as "Banker's rounding"):

- `round(0.5) == 0.0`
- `round(-0.5) == 0.0`
- `round(1.5) == 2.0`
- `round(2.5) == 2.0`

This is in contrast to C++ and Fortran where the default is to round away from zero:

- `round(0.5) == 0.0`
- `round(-0.5) == -1.0`
- `round(1.5) == 2.0`
- `round(2.5) == 3.0`

From a DSL point of view, it is paramount to implement the same rounding behavior in all backends, especially in the context of chaotic systems such as NWPs.

## Decision

We choose to keep the default rounding (away from zero) of the performance backends and use a custom implementation of the `round()` function in python powered backends, namely `numpy` and `debug` backends.

## Consequences

As a consequence

- all backends use the same rounding behavior,
- the `numpy` and `debug` backend get a performance overhead,
- the performance backends (`gt:*` and `dace:*`) are not impacted.

## Alternatives considered

### Adopt banker's rounding in C++ backends

We usually align with python/numpy standards (link to literal precision ADR) in the frontend following the argument of least surprises. In this case, we argue that the performance of performance backends is too important to adopt banker's rounding in C++ powered backends.
