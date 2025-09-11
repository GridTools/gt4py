# Round functions in gtscript

In the context of adding support for a `round()` function in `gtscript`, facing divergent implementations in backend languages, we decided to implement `round()` with tie-breaking to even and `round_away_from_zero()` with tie-breaking away from zero to achieve consistent results while leaving the choice of behavior up to users. We considered multiple alternatives (see below), which all have their advantages and disadvantages.

## Context

There are basically two predominant ways to round ties (e.g. when rounding 1.5 to an integer it's as close to 1.0 as it is to 2.0). The default in python, Kotlin, Julia, and C# is to split ties by rounding to the nearest even value, e.g.

- `round(-0.5) == 0.0`
- `round(0.5) == 0.0`
- `round(1.5) == 2.0`
- `round(2.5) == 2.0`

This is also what the IEEE754 floating point standard defines. In contrast, programming languages like Go, Rust, Ruby, C++, and Fortran default to round away from zero, e.g.

- `round(-0.5) == -1.0`
- `round(0.5) == 0.0`
- `round(1.5) == 2.0`
- `round(2.5) == 3.0`

From a DSL point of view, it is paramount to implement the same behavior in all backends, regardless of the programming language of the backend.

## Decision

We choose to implement `round()` with the tie breaking as suggested by the standard. In addition, we implement the function `round_away_from_zero()` which breaks ties by rounding away from zero. This allows, for example, to truthfully port FORTRAN-based code.

## Consequences

As a consequence

- all backends use the same rounding behavior
- users can choose which rounding behavior suits their needs

## Alternatives considered

### Adopt the python behavior of the `round()` function

One argument in favor of this option is that it aligns with the choice of the frontend language: If GT4Py stencils look like python code, they should also behave like python code.

One argument against this option is that the behavior of python's `round()` function is surprising to domain scientist. Atmospheric scientists, unaware of the intricacies of the floating point standard, are confused by the fact that both 1.5 and 2.5 round up/down to 2.0.

From a technical point of view, this option can be implemented with standard library calls in all current backends.

### Adopt the C++ behavior of the `round()` function

One argument in favor of this option is that the behavior of `std::round()` is expected by domain scientists. Atmospheric scientists, unaware of the intricacies of the floating point standard, expect 1.5 and 2.5 to round up to 2.0 and 3.0 respectively.

One argument against this option is that is doesn't align with the behavior fo the frontend language: While GT4Py stencils look like python code, in this particular case, they don't behave like python code.

From a technical point of view, this option can be implemented with standard library calls in the `gt:*` and `dace:*` backends. The `numpy` and `debug` backends will need a custom implementation of the `round()` function.

### Allow users to configure the behavior of the `round()` function with an argument

One argument in favor of this option is that users can choose: Given that there are good reasons for python and C++ behavior of the `round()` function, we let our users choose what works best for their use-case. By choosing a default, GT4Py can still nudge its users towards one option or another.

One argument against this option is the maintenance overhead of supporting both options.

From a technical point of view, this option can be implemented with an optional second argument of round function that GT4Py exposes, i.e.

```py
def round(x, rounding_mode = ROUND_MODE_DEFAULT):
    """Computes the integer value nearest to `x` (in floating-point format), rounding halfway cases based on rounding_mode.

    This function finds the nearest integer value (in floating point format) to the given number `x`, e.g. `round(2.3) returns `2.0` since
    2.3 is between 2.0 and 3.0 and 2.3 is closer. The behavior in case of 2.5 (exactly halfway between 2.0 and 3.0) is defined by
    the `rounding_mode`. There are two options:

    `ROUND_AWAY_FROM_ZERO` always rounds away from 0.0, e.g. round(-0.5) evaluates to -1.0 and round(0.5) evaluates to 1

    `ROUND_TO_EVEN` rounds to the closest even value, e.g. round(1.5) and round(2.5) both round to 2.0

    Args:
        x: Number to round.
        rounding_mode (optional): How to treat halfway cases (see description). See RoundingMode for options.
    """
    pass
```

The default rounding mode will be centrally defined in `src/gt4py/cartesian/definitions.py` (value subject to discussions) and can be changed with the environment variable `GT4PY_ROUND_MODE_DEFAULT` (let me know if you find a better name).

```py
@enum.unique
class RoundingMode(enum.Enum):
    ROUND_AWAY_FROM_ZERO = enum.auto()
    ROUND_TO_EVEN = enum.auto()

def _get_default_rounding_mode() -> RoundingMode:
    mode = os.environ.get("GT4PY_ROUND_MODE_DEFAULT", default="ROUND_TO_EVEN")

    if mode == "ROUND_AWAY_FROM_ZERO":
        return RoundingMode.ROUND_AWAY_FROM_ZERO

    if mode == "ROUND_TO_EVEN":
        return RoundingMode.ROUND_TO_EVEN

    known = ["ROUND_AWAY_FROM_ZERO", "ROUND_TO_EVEN"]
    raise ValueError(f"Unexpected rounding mode default '{mode}'. Expected one of {", ".join(known)}.")

ROUND_MODE_DEFAULT: RoundingMode = _get_default_rounding_mode()
```

That way, users of GT4Py can adjust the rounding mode to their use-cases, both globally and on a case-by-case basis.

One technical argument against this alternative is the relatively high maintenance overhead compared to having two separate functions.
