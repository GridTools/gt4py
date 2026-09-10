# IntEnum frontend support

The frontend is not supporting 3.11+ `IntEnum` - which are very useful to express flags and control flow in a structured way. We build the support for the following code as an example of a canonical use:

```python
@gtscript.enum
class MyEnum(IntEnum):
    A = 42
    B = 1000


@gtscript.stencil
def enum(field: gtscript.Field[float], order: MyEnum):  # type: ignore
    with computation(PARALLEL), interval(0, 1):
        if order > MyEnum.A:
            field[0, 0, 0] = MyEnum.B
```

## Context

`IntEnum` enforces a type consistency in the elements of `Enum` making it a safe way to express flags and integer constants. It is standard AST as of 3.11+. Supporting in `gt4py.cartesian` would allow a modern python expression to be supported and clean up code around a very common behavior of users to write in hardcoded flag values.

Futher development would allow generic `Enum` with valid types of elements (`float`, `bool`) while disallowing unsupported runtime types (`str`).

## Decision

Implementation presents a `@gtscript.enum` construct which registers `Enum` for use within `gt4py.cartesian`.

We will replace all instances of those enum in stencils by their constant values, allowing the implementation to be focused on the frontend and not downstream into the IRs.

## Consequences

`IntEnum` are considered frozen - if the value is swapped during runtime it won't be reflected.

## Alternatives considered

`IntEnum` can be emulated with a naming convention and simple constants defined in at a Module level. But `Enum` guarantees structure, docstrings and can even be extended for non runtime operations with methods.

We could expand the system to all `Enum` by introducing a type check during registration of the enum in gt4py making sure the type of the parameters are legitimate in the context of `gt4py.cartesian`.
