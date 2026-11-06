---
tags: [embedded, fields, array-api]
---

# Array API Embedded Fields

- **Status**: draft
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2026-11-06
- **Updated**: 2026-11-06

In the context of the embedded execution of field operators on NumPy, CuPy, JAX and
further array libraries, facing a field class per library that only selects a NumPy-like
module and an implementation written against NumPy names and signatures, we decided to
have one `NdArrayField` written against the Array API standard. It takes its array
operations from a per-namespace `ArrayOps` object that libraries specialise by
inheritance. This achieves support for any standard-conforming library without new
classes, conformance checked with `array-api-strict`, and library-specific behaviour that
the field does not know about.

## Context

`NdArrayField` carried the array module as a class variable, and each library had a field
class and a connectivity class that only set it (JAX also changed item assignment). The
implementation called NumPy names (`power`, `transpose(x, axes)`, `concatenate`),
constructors (`xp.dtype(type)`) and methods (`.astype`, `.item()`, `.dtype.type`). A library
whose namespace is only Array API compatible, like torch through `array_api_compat`, needed a
namespace wrapper re-adding the NumPy names and two field classes per device.

## Decision

- `NdArrayField` and `NdArrayConnectivityField` are concrete for every library. A field's
  operations are `array_ops.of(field.ndarray)`: an `ArrayOps` instance, cached by array type,
  for the namespace `array_api_compat.array_namespace` returns.
- `ArrayOps` implements everything the field does to its buffer using only the Array API
  standard. Field code calls standard functions on `ops.xp` and the `ArrayOps` methods where a
  library may need to differ: elementwise functions by name, scalar operands, `where`, item
  assignment, conversion to NumPy, buffer info, dtype translation.
- A library with different needs subclasses `ArrayOps`, overrides only those methods, and is
  registered with `array_ops.register(is_namespace, factory)`. Registration does not import the
  library. Replacing a whole `fbuiltins` implementation for one namespace goes through
  `ArrayOps.builtin_overrides`.
- `common._field` and `common._connectivity` build an `NdArrayField` from any Array API object.
- `NdArrayField.array_ns` remains the library's own module.
- The embedded unit tests and the embedded ffront tests run on `array-api-strict`.

## Consequences

- A conforming library works unregistered. NumPy, CuPy, JAX and torch need small `ArrayOps`
  subclasses for what the standard leaves open (NumPy scalars, item assignment of immutable
  arrays) or lacks (`gamma`, `cbrt`, `fmod`).
- Trace-time behaviour of JAX or torch (domain inference on concrete connectivity tables) plugs
  in through `ArrayOps` hooks, without field subclasses.
- The per-library field classes are removed.
- `gamma`, `cbrt` and `fmod` are not available for libraries that do not provide them.

## Alternatives considered

- **A field subclass per library** (the previous design): every library needs a class even when
  it adds nothing, and the class is the only extension point.
- **A per-operation override table keyed by namespace**: the overrides of one library are
  scattered over entries that each need the library's state, and fallbacks need their own chain.
  It is kept only for replacing whole builtins (`builtin_overrides`), where an entry per builtin
  is the natural unit.
