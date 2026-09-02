---
tags: []
---

# Dimensions as Types

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt), Enrique González Paredes (@egparedes)
- **Created**: 2026-08-27
- **Updated**: 2026-09-02

Why-statement: In the context of statically type-checking user code that annotates fields with
their dimensions, facing the fact that a concrete dimension was an *instance* and therefore not
usable as a type annotation, we decided to make a concrete dimension a *class* whose instances are
the indices along it, to achieve plugin-free type checking under any PEP 484 checker. We considered
keeping and extending the mypy plugin, and accept a wider migration surface plus a change to
`repr()`.

## Context

`Dimension` was a frozen dataclass, so a concrete dimension was a value:

```python
IDim = gtx.Dimension("IDim")
a: gtx.Field[gtx.Dims[IDim], gtx.float64]  # error: Variable "IDim" is not valid as a type
```

The shipped workaround was `gt4py.next.type_system.mypy_plugin`, which substituted placeholder
classes `_DimA`–`_DimD` and then `_AnyDim` for dimension arguments. Drivers for replacing it:

- It tracked **at most four distinct dimensions per run, globally**, keyed by argument name.
  Everything past the fourth collapsed to `_AnyDim`.
- **No `TypeVar`s over dimensions** were possible, so no dimension-generic operator could be typed.
- **Only mypy was served.** pyright users got nothing, and every downstream project had to install
  the plugin.
- The same plugin **blurred dtypes** (`float`/`float32`/`float64`) to suppress an unrelated false
  positive, erasing exactly the information dtype-generic fields need.
- It carried an **undocumented naming requirement**: the fallback hook matched on
  `fullname.endswith("Dim")`, so `I = gtx.Dimension("I")` silently got no support at all.

See GridTools/gt4py#2503.

## Decision

A concrete dimension is a **class**; an index along it is an **instance** of that class:

```python
class IDim(gtx.DimensionIndex): ...


class KDim(gtx.DimensionIndex, kind=gtx.DimensionKind.VERTICAL): ...


IDim  # the dimension    -- annotated `gtx.Dimension`
IDim(0)  # an index into it -- annotated `IDim`
```

This is the shape `enum.Enum` uses: the class is the collection, the instances are its members.
Two objects make it work:

- **`DimensionMeta(type)`** — the metaclass, carrying the API that belongs to the *dimension*:
  `__add__`/`__sub__` building connectivities, the comparison operators building `Domain`s,
  `__repr__`, `__eq__`/`__ne__`/`__hash__`. Binary operators applied to a class object dispatch
  through its metaclass, so this is the only place they can live.
- **`DimensionIndex`** — the class users subclass, and the type whose *instances* are indices.

`common.Dimension` is a PEP 695 alias for `type[DimensionIndex]`, re-exported as `gtx.Dimension`.

There is deliberately **no** separate root above `DimensionIndex`: everything admissible in
`Dims[...]` is user-declarable. A wider root can be added once the requirements of
non-user-declarable dimension types are actually known (see the alternatives below). ADR 0026's
name-prefix-based staggering is untouched by this change.

### Indices are instances, so no checker needs special handling

`I(0)` is an ordinary instantiation, so no checker needs help with it: mypy `--strict` and
pyright 1.1.411 both accept `assert_type(IDim(0), IDim)` and agree line-for-line on the intended
errors, with **zero** suppressions anywhere in the definition. Any design in which `I(0)` returns
something that is *not* an instance of `I` needs suppressions instead, because that is precisely
where the two checkers diverge (see the alternatives below).

Indices are also dimension-typed. `IDim(0)` is an `IDim`, so a function that indexes `IDim`
cannot be handed a `KDim` index — a class of error that a shared `NamedIndex` could not express.

### `tag` names the dimension, `value` stays the index

A dimension needs a string name, and an index needs an integer position. Once indices are instances
of the dimension class, both live in one namespace, so they cannot both be called `value`.

They are split the other way round from what the old `Dimension.value` suggests: the **dimension's
name is `tag`**, and **`value` keeps its existing meaning on the index**. Two reasons. The
vocabulary already exists — `common.Tag` is `TypeAlias = str`, and `iterator/embedded.py` already
keys position dictionaries by `dim.value` as though it were a tag (`pos[source_dim.value]`). And it
puts the churn where it is cheapest: renaming the dimension name touches ~100 sites in `src/`
and 3 in `tests/`, while leaving every index expression — including all downstream `named_index.value`
reads — untouched.

The reverse assignment does not work at all: keeping `value` as the dimension name and adding an
integer `value` to instances is rejected by mypy outright (`Cannot assign to class variable "value" via instance`, plus a `str`/`int` mismatch), even though it appears to work at runtime. Verified,
not assumed.

`tag` is also set on `DimensionIndex` itself, not only by `__init_subclass__` on subclasses: the
metaclass reads it in `__repr__`, `__eq__` and `__hash__`, and an annotation naming the root
reaches those — `eve` memoizes its validator factory on the annotation object and hashes it.

`.dim` survives as a property returning `type(self)`, so existing `named_index.dim` reads keep
working unchanged.

### `Dimension` is a PEP 695 alias, so the old factory spelling fails loudly

`Dimension` is annotation-only: `type Dimension = type[DimensionIndex]`. The programmatic
constructor is a separate function, `common.dimension(tag, kind)`, re-exported as
`gtx.dimension` and used by the IR boundaries that rebuild a dimension from its tag
(`flip_staggered`, `itir_to_gtfn_ir`, the dace lowering, `roundtrip`).

The PEP 695 spelling is deliberate, and it is the reason `gtx.Dimension("I")` *raises* instead
of quietly doing the wrong thing. A plain `TypeAlias` for `type[X]` is a `types.GenericAlias`,
and calling one forwards to its `__origin__` while discarding the arguments:

```python
type[DimensionIndex]("I")  # -> type("I") -> <class 'str'>,  no error
list[int]("ab")  # -> list("ab") -> ['a', 'b']
```

so every unmigrated `IDim = gtx.Dimension("I")` downstream would bind `str` and fail much later,
somewhere unrelated. A `TypeAliasType` is not callable at all, so the same line raises
`TypeError` at the call site. That is worth a great deal for a migration every downstream user
has to perform anyway.

The cost is real but small and localized. `get_origin()` of a PEP 695 alias is `None` rather
than the aliased origin, so any site dispatching on an annotation's *shape* must resolve it
first with `xtyping.resolve_annotation` (added in #2841). Exactly one in-tree site needed it:
`ffront.fbuiltins._type_conversion_helper`, which dispatches on `get_origin(t) is type` for the
builtin function signatures annotated `common.Dimension`. Without the resolve it falls through
to `AssertionError("Illegal type encountered.")` -- loud, and caught by 10 existing tests.

Two things make this narrower than it first appears. `eve.type_validation` already resolves
whole-annotation aliases, so datamodel fields validate identically either way. And the gap
#2841 called worst -- `eve/traits.py:118`, where a `SymbolRef` behind an alias is never
collected -- does not apply here: this alias contains no `SymbolRef`, and both alias flavours
skip that branch identically, since `isinstance(type[X], type)` is `False` for both.

What does carry forward is a maintenance obligation: `eve.datamodels` stores annotations
*unresolved*, so a future dispatch site reading `__datamodel_fields__[...].type` sees the alias
rather than `type[DimensionIndex]` and must resolve. Normalizing annotations where they are
stored is the general fix, and remains an open ADR decision (#2841).

### Pickling goes through `copyreg`, not `__reduce__`

A `DimensionMeta.__reduce__` would be dead code. `pickle.Pickler.save` short-circuits any object
whose type is a subclass of `type` straight to `save_global`, *before* consulting `__reduce_ex__`:

```python
if issubclass(t, type):
    self.save_global(obj)
    return
```

Only the `copyreg` dispatch table is consulted earlier, so `copyreg.pickle(DimensionMeta, ...)` is
mandatory. This matters because `eve.utils.content_hash` pickles to fingerprint build-cache entries
(ADR 0023) and `otf/runners.py` ships compilation tasks to a `spawn`-based `ProcessPoolExecutor`.
Without it, factory-made classes are unpicklable outright and declared ones would pickle by
*reference*, making fingerprints depend on the declaring module.

Index *instances* need nothing extra: once the class round-trips, the default `__reduce_ex__`
carries the instance, including for classes the factory built and for `deepcopy`. All four
combinations (declared/factory × class/instance) were verified to round-trip with identity or
equality preserved.

`gt4py.next.fingerprinting` needs the same treatment for a *different* reason: its deconstructor
dispatch would route a dimension class to the `type` handler and fingerprint it by fully qualified
name. A `DimensionMeta` entry keyed on `(tag, kind)` is registered alongside the existing
`enum.EnumMeta` entry, which has the same shape of problem.

### Identity semantics

Equality and hashing cover **`tag` and `kind`**, exactly as the frozen dataclass covered
`value` and `kind`. The factory
**interns**, and a class-statement declaration registers itself in the same registry (first
declaration wins), so the factory resolves to an already-declared class rather than shadowing it,
and unpickling restores identity rather than an equal-but-distinct class.

Index instances compare by `(type(self), self.value)`, so `IDim(0) == IDim(0)` and
`IDim(0) != KDim(0)`, and they remain usable as dict keys — which `custom_layout_allocators.py`
relies on.

Duplicate names are deliberately **not** rejected: nothing rejected them before, and this tree
declares a dimension named `"I"` many times across `tests/next_tests`. Because "first declaration
wins", `Dimension("I") is <some module's I>` is *not* guaranteed when several modules declare that
tag — only equality is.

Known residual: `typing` caches subscriptions by hash, so for two distinct classes both named `I`,
`Field[Dims[I]] is Field[Dims[I2]]` evaluates to `True` and the annotation binds whichever was
subscripted first. Under `value`+`kind` equality this is *consistent* with runtime semantics, but a
type checker still sees two distinct types, so the static and runtime views disagree exactly here.

### Annotation convention

- Annotations: a dimension is **`common.Dimension`**, a PEP 695 alias for
  `type[common.DimensionIndex]`, re-exported as `gtx.Dimension`.
- Declarations: `class IDim(gtx.DimensionIndex): ...`.
- Indices: an index along `IDim` is annotated `IDim`; the general case is `common.DimensionIndex`.
- The dimension's name is `dim.tag`; an index's position is `idx.value`.
- Runtime guards: `isinstance(x, common.Dimension)` → **`isinstance(x, common.DimensionMeta)`**.
- Structural patterns: `case common.Dimension() as dim` → `case common.DimensionMeta() as dim`.

`DimensionMeta` is exported from `gt4py.next.common` for the guard form, but deliberately *not*
re-exported from `gt4py.next`: user code should not need it.

## Consequences

Easier:

- `gtx.Field[gtx.Dims[IDim], gtx.float64]` type-checks under mypy **and** pyright with no plugin,
  and a dimension mismatch between two fields is a real static error.
- `TypeVar("D", bound=gtx.DimensionIndex)` works, unblocking dimension-generic operators.
- Indices carry their dimension in the type, so mixing them is a static error.
- `Dimension` — the name users write in annotations — keeps the spelling it had before this
  change, so the 208 annotation sites read as they always did.
- The dimension half of `mypy_plugin.py` is deleted rather than maintained; the plugin now only
  blurs scalar precision and keeps a deprecated `TreatDimensionsAsTypes` alias.
- No `# type: ignore` is needed for the instantiation path, and `NamedIndex` disappears entirely.

Harder / accepted:

- **`Dimension.value` → `.tag` is a breaking rename**, ~100 sites in `src/` and 3 in `tests/`. Index
  expressions are unaffected: `NamedIndex` is removed, but its `.value` and `.dim` spellings survive
  on the instance, so downstream index code keeps working. ICON4Py needs a note for `.tag`.
- **`repr()` changed** from `Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)` to
  `I[horizontal]`. `str()` is unchanged, so every error message built with `str(dim)` is
  byte-identical, but doctests that echoed a dimension needed updating.
- `NamedRange` stays a namedtuple, so it is no longer structurally parallel to an index. The two
  still meet in `AbsoluteIndexElement`; unifying them is deliberately out of scope.
- Constructing an index is ~1.15× the cost of the old namedtuple (130ns vs 113ns measured);
  attribute access is ~20% cheaper. Neither is significant for the embedded paths that build them.
- This is the **first PEP 695 alias in `src/`**, so a dispatch site that reads an annotation's
  shape must resolve it (`xtyping.resolve_annotation`) rather than calling `get_origin()`
  directly. One site needed it today; `eve.datamodels` stores annotations unresolved, so the
  obligation carries forward to new ones. See #2841.
- The metaclass operator methods carry `# type: ignore[misc]` for a definition-site-only mypy
  restriction on generic self-types. pyright needs no such suppression.

## Alternatives considered

### Keep and extend the mypy plugin

- Good, because no migration is needed.
- Bad, because the four-dimension global cap and the name-keyed substitution are inherent to the
  approach, not incidental.
- Bad, because it can never serve pyright, which has no plugin mechanism.
- Bad, because dtype blurring and dimension handling are entangled in one plugin.

### No instance level: keep `NamedIndex` and have `__new__` return it

A dimension class has no instances at all; `I(0)` keeps returning the pre-existing `NamedIndex`
namedtuple.

- Good, because `NamedIndex` inherits equality, hashing and `__str__` from `NamedTuple` for free,
  and the dimension's name and the index position never share a namespace, so no rename is needed.
- Bad, because `__new__` returning a non-instance is exactly where the checkers disagree — mypy
  ignores a metaclass `__call__` and reads `__new__`, pyright honours `__call__` and infers `Any`
  from it — so it needs the same overload pair on *both* `DimensionMeta.__call__` and
  `DimensionIndex.__new__`, plus three `# type: ignore[misc]` for the definition-site rule that
  `__new__` must return an instance of its class. Those are new suppressions, introduced by a change whose
  purpose is removing false positives.
- Bad, because every index has the same type, so no signature can say *which* dimension it
  indexes.
- Bad, because a `NamedIndex` is also a `Sequence`, which the index-shape guards in `common.py`
  have to be written around.

### Keep `value` as the dimension name and rename the index to `.index`

- Good, because `Dimension.value` keeps working, so no in-tree rename is needed.
- Bad, because it does not type-check: an instance attribute cannot shadow a `ClassVar`, so the
  index has to take a new name regardless. Verified, not assumed.
- Bad, because it puts the break on *every* index expression, downstream included, whereas naming
  the dimension `tag` touches ~100 in-tree sites and no index code.

### A separate `DimensionBase` root above the user-declarable class

- Good, because parameterized dimension types (a future `Staggered[D]`, `Local[C]`) could be
  admitted into `Dims[...]` without being user-declarable.
- Bad, because nothing needs it yet, and it costs a second exported name and a wider alias for no
  present benefit. Deferred until the requirements of such dimensions are known; ADR 0026's
  name-prefix staggering is unaffected either way.

### A plain `TypeAlias` for `Dimension` instead of a PEP 695 alias

- Good, because `get_origin()` then returns `type`, so no dispatch site needs to resolve the
  alias, and `src/` keeps having no PEP 695 aliases at all.
- Bad, because `type[X]` is a `types.GenericAlias`, and calling one forwards to `__origin__`
  while discarding the arguments -- so an unmigrated `gtx.Dimension("I")` evaluates to `str`
  with no error, and fails somewhere unrelated much later. A silent wrong value is a far worse
  migration experience than a `TypeError` at the call site.

### A `TYPE_CHECKING` split: the alias for checkers, a callable shim at runtime

- Good, because it would keep `gtx.Dimension("I")` working, warning and returning the class.
- Bad, because `eve.datamodels` resolves annotations at *run time*, so a field annotated
  `Dimension` sees the shim object rather than a type and validator construction fails outright.
  Non-viable: `type[Dimension]` fields are exactly what this change introduces.

### Free functions instead of metaclass operators (`shift(I, 1)`)

- Good, because it sidesteps the `type: ignore[misc]` suppressions on the metaclass operators.
- Bad, because it changes user-facing notation for no benefit — pyright 1.1.411 accepts the
  metaclass form with zero diagnostics, and the suppressions are definition-site only.

### Identity-based (nominal) equality

- Good, because it would match what type checkers see.
- Bad, because it breaks the many files that independently declare `IDim`, and changes the meaning
  of `Dimension("I") == Dimension("I")`, which downstream code relies on.

## References

- GridTools/gt4py#2503 — dimensions are not valid as types.
- ADR 0023 — Fingerprinting (why value-based pickling of dimensions matters).
- ADR 0026 — Staggered Dimensions (the `_Staggered` name prefix, untouched by this change).
- `typing_tests/test_next.yaml` and `typing_tests/test_pyright.py` — the checker-level tests.
