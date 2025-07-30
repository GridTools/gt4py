# Literal precision

In the context of porting a numerical weather prediction (NWP) model, porting mixed precision code, we realized there's a need to control the default literal precision to pass numerical validation tests against data serialized from original model runs.

## Context

Scenario: You are porting an NWP model from FORTRAN to GT4Py stencils. To have setup end-to-end numerical regression tests with data serialized from the original FORTRAN code and want to validate your port by running with that serialized data, expecting equal/similar results. Your original FORTRAN code generally has a type definition for the default float type (such that the model can run with single- and double-precision floats). Certain parts of the original FORTRAN code have mixed precision calculations, i.e. places where authors opted for hard-coded single- or double-precision calculations by design.

To truthfully re-create the numerics of the FORTRAN code in the example above, we need two things

1. a way to set the "general float type"
2. means to specifically cast to single- / double-precision in mixed-precision calculations.

To complicate the problem, python doesn't distinguish single- / double-precision floats.

## Decision

We chose to address this problem with the following (frontend) features:

1. A configurable global default for literal precision. Defaults to the system's architecture (e.g. 64-bit calculations 64-bit systems) with the environment variable `GT4PY_LITERAL_PRECISION` as override.
2. Per stencil, the precision can be tuned via `literal_precision` as part of the `BuildOptions`.
3. Within stencil code, literals like `42` and `42.0` will adhere the (global) precision default.
4. Type annotations `int` and `float` adhere to the (global) precision default. Annotations `int32`, `int64`, `float32`, and `float64` are added to specialize if needed.
5. Similarly, `int()` and `float()` casts adhere to the (global) precision default. Casts `int32()`, `int64()`, `float32()`, and `float64()` are added to specialize if needed.

## Consequences

We believe that with the above decision, we give users all the tools they need to truthfully port mixed precision codes. Ported code might become more cluttered (with type annotations and casts). We see no way around this issue.
