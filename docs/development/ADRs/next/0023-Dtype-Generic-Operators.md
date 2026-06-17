---
tags: []
---

# [Dtype-Generic Operators]

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2026-06-15
- **Updated**: 2026-06-15

Field operators (and programs calling them) may be **generic in the field dtype**,
spelled with a native value-constrained `typing.TypeVar` so the same annotation is
meaningful to mypy and to the DSL frontend. Each concrete call is specialized
(monomorphized) at call time.

```python
FloatT = typing.TypeVar("FloatT", gtx.float32, gtx.float64)


@gtx.field_operator
def diff(
    a: gtx.Field[gtx.Dims[I, J], FloatT], b: gtx.Field[gtx.Dims[I, J], FloatT]
) -> gtx.Field[gtx.Dims[I, J], FloatT]:
    return a - b
```

## Context

`common.Field` is already a runtime-introspectable generic protocol, so
`Field[Dims[I, J], T]` with a value-constrained `TypeVar` is a valid, mypy-visible
annotation today; what was missing is the DSL side. The internal type system had
`DeferredType` ("some type, maybe constrained") but no notion of *identity* — it
could not express "the *same* unknown dtype in two parameters and the return type",
the essence of generics. The runtime monomorphization machinery already existed
(grown for scan operators): `CompiledProgramsPool` keys a per-call specialization
cache on the concrete argument types.

Prior art (numpy.typing, jaxtyping, Numba, Taichi, Triton, DaCe) converges on the
two choices adopted here: a real generic annotation that static checkers can see,
and monomorphization at call time.

## Decision

### User-facing spelling

A **value-constrained** type parameter inside the real generic `Field` class,
spelled with PEP 695 `def op[T: (float32, float64)](...)` (preferred at the 3.12+
floor) or the equivalent module-level `TypeVar("T", float32, float64)` (accepted,
produces the same runtime objects). Value-constrained — not `bound=` — because each
use must resolve to exactly one listed type, which makes the dtype predicates
decidable and the variant set finite (eager precompilation possible). `bound=`-only
and unconstrained type variables are rejected with a clear message.

### `ts.TypeVarType`

A new `DataType` subclass carrying `name` and `constraints: tuple[ScalarType, ...]`.

- Subclassing `DataType` lets it fit unchanged into `FieldType.dtype` (widened to
  `ScalarType | ListType | TypeVarType`), `TupleType`/`NamedCollectionType` members,
  and `foast.Symbol`.
- **Identity is the name**, scoped to one operator signature. Two *distinct*
  same-named `TypeVar` objects in one signature are rejected at parse time (with PEP
  695 this is impossible by construction). As a frozen eve `DataModel` it gets
  deterministic `eq`/`hash`/`content_hash` for cache keys.
- `ts.DeferredType` is **not** replaced. The two mechanisms coexist: `DeferredType`
  means "not yet inferred" (and currently also encodes the scan operators' *dims*
  genericity); `TypeVarType` means "universally quantified over the constraint set".
  A single `type_info.is_generic` predicate recognizes both.

### Decisions D1–D5

- **D1 — Decoration-time body checking with opaque `TypeVarType`.** The body is
  type-checked once, at decoration time, with `T` treated as an opaque scalar.
  Errors are reported in the user's vocabulary (`T`), not in instantiated terms.
  (Rejected: skip-until-instantiation — breaks the decoration-time-errors UX;
  finite monomorph-check — duplicates compile work and reports in the wrong
  vocabulary. The finite check survives only as a test-suite cross-check.)
- **D2 — Value-constrained TypeVars only.** Finite variant set ⇒ decidable dtype
  predicates and eager `.compile()` of all members. `bound=` is a future extension.
- **D3 — Strict no-promotion.** `promote(T, T) = T`; mixing `T` with a concrete
  scalar/dtype (including literals: `a * 2.0`) is a decoration-time error naming the
  type variable. `astype(x, T)` is the designated remediation (a fast-follow). We
  pre-commit to strict-first rather than inheriting Numba-style silent promotion;
  this is expected to be the main ergonomics complaint and is revisited via a named
  "generic literals" follow-up, not by relaxing the default.
- **D4 — Monomorphize at FOAST level; never lower generic GTIR.** Specialization is
  direct type substitution over the typed FOAST, with a full re-run of type
  deduction as a soundness backstop under `__debug__`. (Rejected: lowering generic
  FOAST and concretizing at GTIR level — `foast_to_gtir` bakes dtypes into literals
  and casts; GTIR has no syntax for "dtype of param x".)
- **D5 — Binding is a first-class `type_info` utility.**
  `bind_type_vars(params, args)` (structural match, consistency + exact-match
  checks) and `substitute_type_vars(type_, binding)` (recursion over every TypeSpec).
  `accepts_args` keeps its boolean interface; callers needing the binding use the
  new API.

### Monomorphization strategy

- **Direct operator call with a backend:** `FieldOperator.__call__` →
  `CompiledProgramsPool`. The pool already detects generic signatures
  (`is_generic`), keys the cache on the full concrete substitution
  (`arg_specialization_key`), and forwards concrete types as `CompileTimeArgs`. A
  new `foast_specialize` toolchain step (after `func_to_foast`) computes the binding
  and substitutes throughout the FOAST tree; everything downstream runs on a
  concrete artifact.
- **Generic operator called from a concrete program:** the binding is fully static
  at program decoration. The fieldop signature checks bind-and-substitute, and a
  PAST monomorphization pass (run in `past_to_itir`) recomputes the binding from the
  typed call-site args, name-mangles the callee per binding (e.g.
  `diff__float32`), and swaps in a specialized callable via a new
  `GTCallable.__gt_specialize__(binding)`. Two bindings of one operator naturally
  become two GTIR `FunctionDefinition`s.
- **Embedded mode:** nearly free — the original Python definition runs on real
  fields once decoration tolerates generic signatures.

### Cache-key story

The pool's `arg_specialization_key` hashes all argument types, so the full
substitution is in the key — distinct dtypes hit distinct variants. Value-constrained
TypeVars make eager precompilation of all variants possible via the existing
`.compile()` API.

## Out of scope / deferred (with forward-compatibility notes)

- **Generic scan operators** — rejected with a clear message (needs `init: T`
  coercion semantics). Nothing in the utilities hardcodes `FieldOperatorType`.
- **`bound=` TypeVars** — infinite constraint sets; predicates by bound; no eager
  precompile.
- **`astype(x, T)` / generic scalar constructors** — the D3 remediation; requires a
  `ConstructorType` over `TypeVarType`.
- **Builtin coverage** — `where`, `broadcast`, reductions, `concat_where`, neighbor
  fields are audited and widened incrementally; until then a generic argument to an
  un-audited builtin is a clear decoration-time error (math builtins already work).
- **Dimension genericity** — a separate effort (the true fix for the scan
  `DeferredType`/fabricated-`Dimension` hack). The binding utilities are kept
  **dtype-scoped** here: `bind_type_vars`/`substitute_type_vars` map names to
  `ScalarType` only, and same-name rejection is specified over dtype type variables
  only. Widening the binding environment to dimensions and generalizing same-name
  rejection across type-parameter kinds is explicitly deferred to that work.
- **PEP 696 dtype defaults** (unparameterized `Field` means `float64`) and
  **mypy-plugin un-blurring** of `float32`/`float64` — later, coordinated with the
  `Field` annotation cleanup (gt4py #1415/#1416).
