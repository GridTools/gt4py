---
tags: [backend, embedded, allocators]
---

# Embedded Backends

- **Status**: valid
- **Authors**: Enrique González Paredes (@egparedes)
- **Created**: 2026-10-28
- **Updated**: 2026-10-28

In the context of selecting embedded execution for programs and operators, facing `backend=None`
as its only representation — which has no allocator and no name, cannot tell embedded execution
apart from "not configured", and forces every consumer to special-case `None` — we decided to
represent embedded execution by `EmbeddedBackend` objects, a concrete kind of the abstract
`Backend` of [ADR 0029](0029-Backend-Structure-and-Pipeline-Simplification.md), with one instance
per array namespace (NumPy, CuPy, JAX), to make the NumPy instance the default backend, and to make
`backend=None` an error, to achieve one representation for every way a program can run and
working allocation (`allocator=program`) for embedded programs. We accept a loud break for code
that passes `backend=None`, a widened allocator-factory protocol, and a changed metrics key for
embedded calls.

## Context

Before this ADR, `backend=None` meant embedded execution and `DEFAULT_BACKEND` was `None`. This
has several consequences:

- **No allocator.** Embedded execution allocates with an array namespace (NumPy, CuPy, JAX), but
  `None` cannot say which. `Program.__gt_allocator__` raises for embedded programs, so
  `gtx.zeros(..., allocator=program)` does not work for them, and test suites carry their own
  stand-ins: gt4py's `EmbeddedDummyBackend` (name + array namespace, plus a fake attribute to look
  like a compiled backend), icon4py's "testing infrastructure doesn't allow to specify embedded
  backend aka `None` for cupy" note.
- **No name.** Embedded calls are reported in metrics under a hardcoded `'embedded'` label.
- **`None` is ambiguous.** It reads as "no backend configured", and every consumer special-cases
  it (`backend is None` checks in the decorators, in tests, in applications).
- **Allocation by factory assumes a field-buffer allocator.** `FieldConstructor` only unwraps
  `__gt_allocator__` inside its custom-layout branch, after checking for an array namespace. An
  allocator factory whose `__gt_allocator__` is an array namespace would be pushed into the
  `TensorBuffer` path.

ADR 0029 made `Backend` an abstract base with `name` and `allocator`, a `CompiledBackend` kind, and
a single dispatch on the kind, anticipating an embedded kind.

## Decision

### `EmbeddedBackend`

```python
@dataclasses.dataclass(frozen=True)
class EmbeddedBackend(Backend):
    allocator: ArrayNamespace = numpy  # validated to be an array namespace
```

- A sibling of `CompiledBackend`, not a configuration of it: it has no frontend, compiler or
  loader. Execution is identical for every instance — embedded operators dispatch on the array
  types of their arguments — so `allocator` only decides where gt4py allocates buffers for this
  backend.
- Instances, in `gt4py.next.program_processors.runners.embedded`: `embedded_numpy` (also exported
  as `gtx.embedded_numpy`), `embedded_cupy` and `embedded_jax`. The latter two require the optional
  packages; accessing them without the package raises an `AttributeError` that names it.
- CuPy and JAX are **instances**, not subclasses. A subclass with its own execution behaviour (for
  example JAX execution under `jax.jit`) is possible later and would plug into the same dispatch.

### `backend=None` is an error; the default is `embedded_numpy`

- `decorator.DEFAULT_BACKEND` becomes `embedded_numpy`. `@gtx.program`, `@gtx.field_operator` and
  `@gtx.scan_operator` without `backend=` therefore still run embedded on NumPy.
- `backend=None` raises `TypeError` at construction — in the decorators, `from_function`,
  `with_backend` and `dataclasses.replace` — with a message naming the replacement: omit `backend`
  or pass an `EmbeddedBackend` such as `gtx.embedded_numpy`. There is no deprecation period, in
  line with ADR 0029's no-alias policy: a `None` that silently kept working would hide the new
  objects from exactly the code that should adopt them.
- Scope: this concerns the field-view decorators (`gt4py.next.ffront`). The iterator-level
  `fendef(..., backend=None)` keeps meaning *iterator* embedded execution (`iterator/embedded.py`),
  a different execution model used internally by the roundtrip backend and the iterator tests; an
  `EmbeddedBackend` is not accepted there.
- The dispatch helper of ADR 0029 replaces its `None` case by `EmbeddedBackend`; any other
  `Backend` still raises `TypeError`.

### Allocation

- `Backend.__gt_allocator__` moves to the base class and returns the backend's allocator (unwrapping
  a factory); `CompiledBackend` keeps its narrower return type. `Program.__gt_allocator__` returns
  its backend's, for every kind.
- `FieldBufferAllocatorFactoryProtocol.__gt_allocator__` may return an array namespace as well as a
  field-buffer allocator. `FieldConstructor` unwraps factories *before* deciding between the
  array-API path and the custom-layout path, so `gtx.zeros(..., allocator=program)` allocates with
  NumPy/CuPy/JAX for an embedded program and with the backend's layout for a compiled one.
  `is_field_allocator_factory_for(obj, device)` is true only when the factory yields a field-buffer
  allocator for that device type.

### Name

Embedded calls report metrics under `f"{program}<{backend.name}>"`, the format of compiled calls,
instead of the hardcoded `'embedded'`.

### Copying

Backends are immutable configuration objects: `Backend.__deepcopy__` returns the backend itself,
and `Program.__gt_allocator__` is a plain property rather than a cached one. Without these, every
program using the default backend would stop being deep-copyable — the NumPy module held by
`embedded_numpy` cannot be copied (`TypeError: cannot pickle 'module' object`), and the cached
allocator would put it into the program. With them, programs can be deep-copied whatever their
backend, also after being used as `allocator=`. A copy shares the backend, including caller-owned
state such as a DaCe external workspace; copies are not isolated from each other, exactly like two
programs on the same backend. A program that already holds compiled variants still copies its
compiled-programs pool, which fails for DaCe-compiled programs; that limitation predates this ADR
and is not addressed here. Backends are not picklable (compiled backends never were, and programs
are not picklable either); this ADR does not change that.

### Compiling an embedded program

`.compile()` on a program or operator with an `EmbeddedBackend` raises `RuntimeError` saying that
embedded programs are not compiled (it used to say "Cannot compile a program without backend").

## Consequences

- Every way to run a program is a `Backend` object with a name and an allocator; code that needs
  to know the kind uses `isinstance(backend, EmbeddedBackend | CompiledBackend)` instead of
  `backend is None`.
- Code passing `backend=None` (notebooks, applications, tests) fails loudly and must pass
  `gtx.embedded_numpy` or omit `backend`. Code mapping "embedded" to `None` in its own
  configuration (icon4py's backend table) must map it to an `EmbeddedBackend`. Code that reads
  `backend.allocator.device_type` for any `Backend` must handle an array-namespace allocator.
- `gtx.zeros(..., allocator=program)` and `allocator=backend` work for embedded programs; the
  test suite drops `EmbeddedDummyBackend` and uses the real instances.
- `copy.deepcopy` of programs keeps working for embedded programs and also works for compiled
  ones that hold no compiled variants yet; the copy shares the backend object (and its
  caller-owned state). Deep-copying a program with DaCe-compiled variants still fails, as before.
- Code that classifies backends by name (e.g. `backend.name.startswith("run_dace_")`,
  `"embedded" if backend is None else backend.name`) should switch to types
  (`isinstance(backend, EmbeddedBackend)`); the default backend's name is `embedded_numpy`.
- The metrics key of embedded calls changes from `prog<'embedded'>` to `prog<embedded_numpy>`
  (and similar for CuPy/JAX), so dashboards keyed on the old string need updating.
- Third-party allocator-factory implementations keep working; code that *consumes*
  `__gt_allocator__` must accept an array namespace.

## Alternatives considered

- **Keep `None` as an alias of `embedded_numpy`**: rejected; it keeps the ambiguity and the special
  cases alive and would silently pick NumPy for code that meant "whatever the arrays are".
- **Deprecate `None` for one release**: rejected for consistency with ADR 0029's no-alias policy;
  the break is at decoration time and its message names the fix.
- **One `EmbeddedBackend` without an allocator** (allocation decided elsewhere): rejected; the
  allocator is exactly what the embedded case was missing, and per-namespace instances are what
  test matrices and applications already model by hand.
- **Subclasses per namespace (`NumPyBackend`, `CuPyBackend`, ...)**: rejected for now; they would
  differ only in a default value. Subclasses are reserved for differences in execution behaviour.
- **Unwrap `__gt_allocator__` only for programs**: rejected; any factory can legitimately yield an
  array namespace once embedded backends are factories, so the constructor must handle it
  generally.
