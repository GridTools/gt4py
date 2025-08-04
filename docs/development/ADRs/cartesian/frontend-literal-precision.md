# Literal precision

This frontend feature was developed for seamlessly porting Numerical Weather Prediction (NWP) models from FORTRAN to GT4Py. Some FORTRAN-based NWP models are built with the capability to to run in either single- or double-precision. Normally, this is controlled at compile-time. In order to duplicate that behavior, there is a need for a way to control the default literal precision. This feature also simplifies validation as validation is normally done against data serialized from original model runs.

## Context

Consider the following scenario: You are porting an NWP model from FORTRAN to GT4Py stencils. To have setup end-to-end numerical regression tests with data serialized from the original FORTRAN code and want to validate your port by running with that serialized data, expecting equal/similar results. Your original FORTRAN code generally has a type definition for the default float type (such that the model can run with single- and double-precision floats). Certain parts of the original FORTRAN code have mixed precision calculations, i.e. places where authors opted for hard-coded single- or double-precision calculations by design.

To truthfully re-create the numerics of the FORTRAN code in the example above, we need three things:

1. a way to set the "general float type"
2. means to specifically cast to single- / double-precision in mixed-precision calculations.

To complicate the problem, python doesn't distinguish single- / double-precision floats. Also, 64-bit versions of python (the default running on 64-bit systems), will use 64-bit integers by default and python's type-casting will propagate these, e.g.

```none
>>> import numpy as np
>>> f_32 = np.float32(42)
>>> type(f_32)
<class 'numpy.float32'>
>>> an_int = 5
>>> type(5 * f_32)
<class 'numpy.float64'>
```

To truthfully replicate the numerics of the FORTRAN code, we thus not only need to control float, but also integer precision.

## Decision

We chose to address this problem with the following (frontend) features:

1. A configurable global default for int/float literal precision. Defaults to the system's architecture (e.g. 64-bit calculations 64-bit systems) with the environment variables `GT4PY_LITERAL_INT_PRECISION` and `GT4PY_LITERAL_FLOAT_PRECISION` as overrides.
2. Per stencil, the precision can be tuned via `literal_int_precision` and `literal_float_precision` as part of the `BuildOptions`.
3. Within stencil code, literals like `42` and `42.0` will adhere the (global) precision default.
4. Type annotations `int` and `float` adhere to the (global) precision default. Annotations `int32`, `int64`, `float32`, and `float64` are added to specialize if needed.
5. Similarly, `int()` and `float()` casts adhere to the (global) precision default. Casts `int32()`, `int64()`, `float32()`, and `float64()` are added to specialize if needed.

## Consequences

We believe that with the above decision, we give users all the tools they need to truthfully port mixed precision codes. Ported code might become more cluttered (with type annotations and casts). We see no way around this issue.

We'd also like to point out that the above decision keeps python's defaults while allowing users to customize integer and floating point precision if they need to.

## Alternatives considered

### Not have it

As argued above, not having this feature would

- make it impossible to run NWP models at 32-bit (float) precision (which is done for performance reasons)
- make it nearly impossible to test the behavior of the ported model at 32-bit (float) precision.

### Not distinguish int/float precision

We evaluated a version with just `literal_precision` (and `GT4PY_LITERAL_PRECISION`) that always sets int and floats to the same precision. This alternative is simpler, which makes it easier to maintain. On the other hand, we needlessly tie int and float precision together. In the interest of flexibility, we decided to allow users to specify int and float precisions separately.
