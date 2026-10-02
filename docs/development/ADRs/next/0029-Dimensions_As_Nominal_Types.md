---
tags: []
---

# Dimensions as Nominal Types

- **Status**: proposed
- **Authors**: Enrique González Paredes (@egparedes)
- **Created**: 2026-09-18
- **Updated**: 2026-10-02

A concrete dimension becomes a **class**, and an index along it an **instance** of
that class — the shape `enum.Enum` uses, where the class is the collection and
the instances are its members:

```python
class IDim(gtx.CartesianAxisIndex): ...


class KDim(gtx.CartesianAxisIndex, kind=gtx.DimensionKind.VERTICAL): ...


class Cell(gtx.DimensionIndex): ...  # a mesh location: not a Cartesian axis


IDim  # the dimension    -- annotated `gtx.Dimension`
IDim(0)  # an index into it -- annotated `IDim`
```

so `gtx.Field[gtx.Dims[IDim], gtx.float64]` is valid for any PEP 484 checker with
no gt4py mypy plugin.

A dimension's **identity is the Python type**, and its `tag` — the string that
crosses into the IR and the generated code — is its **qualified Python name**,
`f"{cls.__module__}.{cls.__qualname__}"`. A Cartesian axis — a dimension with
index arithmetic and a staggered partner — is a subclass of `CartesianAxisIndex`;
mesh locations subclass `DimensionIndex` directly.

## Context

`Dimension` was a frozen dataclass whose instances are values, not types. Two
consequences drove this change.

**Dimensions were not usable as types.** `Field[[IDim], float64]` needed a
dedicated mypy plugin that substituted at most four distinct placeholders per run
(`_DimA`..`_DimD`, then `_AnyDim` for everything after), made `TypeVar`s over
dimensions impossible, and served only mypy — no other checker. See #2503.

**Names are load-bearing in too many places.** A neighbor connectivity is
currently spread over four independently authored strings that must agree by
string equality and are never checked against each other at declaration time: the
`FieldOffset` tag, the Python variable it is bound to, the local `Dimension`'s
name, and the `offset_provider` key. Whichever one reaches
`common.get_offset` depends on the execution path and the operation. Making a
dimension's identity its Python type is the prerequisite for collapsing those
names into one declaration ([ADR 0030](0030-Connectivities_As_Types.md) covers the
connectivity half).

## Decision

### Identity is the Python type; the tag is the qualified name

Two dimension classes are the same dimension if and only if they are the same
class. The static view (checkers see nominal types) and the runtime view
(equality is `is`) agree by construction, and the tag is a unique string that is
also a valid IR spelling.

The alternative — `(name, kind)` value equality plus an interning registry, so
that independently declared same-named dimensions stay interchangeable — was
considered and rejected. It decouples the Python type's identity from the IR's,
and needs a registry, a `copyreg` hook and a custom fingerprint deconstructor to
paper over that gap. It also cannot give a dimension nested inside another
declaration a unique name without further convention, which the connectivity
work requires.

One concrete argument in favour of nominal identity: under `(name, kind)`
equality the `typing` subscription cache aliases `Field[Dims[I]]` and
`Field[Dims[I2]]` for two *distinct* same-named classes, so the static and
runtime views disagree exactly there. Under type identity that aliasing
disappears.

### Consequences of the tag being a qualified name

1. **Reconstruction from the IR is an import.** `common.resolve(tag)` imports the
   module and walks the qualname; nested declarations resolve naturally. The IR
   references a Python type exactly the way `pickle` references a class. Where the
   module path ends is memoized, because type inference calls it once per
   `AxisLiteral`; the attribute walk is repeated, so a redefined declaration is
   found. An `AxisLiteral` stores only the tag: its `kind` is the resolved
   dimension's, so the two cannot disagree.

   A purely dotted tag does not record where the module path ends and the
   qualname begins, so `resolve` tries the *longest importable prefix* and walks
   the remainder. A collision requires a module path and an attribute chain to
   have the same spelling; a real module always wins. `pickle` avoids the
   ambiguity by storing the two parts separately, and that remains available if
   the residual ever bites.

2. **Types reaching the IR must be importable**, i.e. declared at module level.
   `__init_subclass__` rejects a `<locals>` qualname as an early heuristic; it is
   neither necessary nor sufficient (`type("Dyn", ...)` inside a function passes,
   a class deleted after creation passes), so the authoritative check remains
   pickle's own `save_global`.

   **Interactive `__main__`** -- a notebook, the REPL, `python -c` -- has no
   `__file__` for a worker to re-import, so a class declared there pickles in the
   parent (which has it) and then fails to unpickle in a spawn worker. This is not
   left as a documented limitation: the repository's own Quickstart and workshop
   notebooks declare dimensions interactively and are run in CI. Instead the
   process runner detects a job that references a class from an interactive
   `__main__` and compiles it in the calling thread, with a warning -- the same
   fallback it already takes for an executor that cannot be pickled. A *script's*
   `__main__` is re-imported by spawn workers (as `__mp_main__`), so scripts keep
   parallel compilation, provided they have the `if __name__ == "__main__":` guard
   the worker pool already requires. `(tag, kind)` value identity with a registry
   avoided this by pickling dimensions by value; that is the one case where it
   was strictly more convenient.

3. **No registry and no blanket `copyreg`.** Module-level classes pickle by
   reference, which is correct. A *narrow* `copyreg` registration is still
   required for parametrized dimensions — see Staggered below.

4. **Cache fingerprints depend on module paths.** A dimension is fingerprinted by
   qualified name, so moving a declaration between modules invalidates compiled
   artifacts. Its `kind` is fingerprinted too: it decides a field's layout order and
   the scan axis, so a dimension redefined under the same name with another `kind`
   (a re-run notebook cell) does not reuse artifacts. A staggered dimension is
   fingerprinted through its base, and its base is fixed by interning. This is a consequence for the build cache of ADR 0023, not a
   reversal of it. The generic `type` deconstructor is correct for the *lenient*
   fingerprint variant; the STRICT variant rejects a parametrized dimension,
   which is not importable under its qualified name.

5. **Generated identifiers need injective mangling.** A dot is illegal in a C++
   identifier, in a DaCe symbol, and in `eve`'s `SymbolName`
   (`^[a-zA-Z_]\w*$`). One shared pair, used by every backend:

   ```python
   def codegen_name(tag: Tag) -> str:
       return tag.replace("_", "_u").replace(".", "_d").replace("[", "_l").replace("]", "_r")


   def from_codegen_name(name: str) -> Tag:
       return re.sub(
           r"_([udlr])", lambda m: {"u": "_", "d": ".", "l": "[", "r": "]"}[m.group(1)], name
       )
   ```

   A *prefix* escape, not `_ -> __` followed by `. -> _`: the latter is **not
   injective**, since a dot becomes a single underscore and `".."` collides with
   an escaped `"_"`. Every `_` in the output is the first character of a
   two-character escape, so decoding is unambiguous. Brackets are escaped too: a
   parametrized tag such as `Staggered[pkg.K]` contains them, and they would
   otherwise survive into the identifier. Names grow, which is what
   gtfn's existing `TagDefinition.alias` mechanism is for.

6. **Staggered dimensions become a real parametrized type.** ADR 0026's
   `_Staggered` *name prefix* cannot survive type identity: `Dimension(f"_Staggered{name}")`
   names no importable type, and the prefix cannot recover the base dimension's
   module. `Staggered[D]` replaces it and supersedes that part of ADR 0026.

   A PEP 695 generic does not work: `Staggered[KDim]` would be a
   `typing._GenericAlias`, not a class, so it fails `issubclass` and eve's
   `type[DimensionIndex]` validation, and its tag cannot name the base. Instead a
   metaclass `__getitem__` builds and **interns a real class**, paired with a
   `TYPE_CHECKING` declaration so checkers still see an ordinary generic:

   - bases are `(Staggered,)` and deliberately **not** `(Staggered, base)`:
     a staggered dimension is a *different* dimension, so
     `issubclass(Staggered[KDim], KDim)` must be false. Only `kind` is inherited.
     `Staggered` itself derives from `AnyCartesianAxisIndex`, and its parameter is
     bounded on `CartesianAxisIndex` (see *Cartesian axes* below).
   - `is_staggered(dim)` is `"base" in dim.__dict__` and
     `as_non_staggered(dim)` is `dim.base` — structural, no string sniffing.
     `issubclass(dim, Staggered)` would be wrong, because it is also true of the
     bare base, which has no `base`.
   - `resolve` gains a `<qualname>[<tag>]` grammar and evaluates
     `Staggered[resolve(inner)]`, hitting the same intern table, so a staggered
     dimension round-trips through the IR to the *same* class object. This is the
     one place where "resolution is an import" is not literally true.
   - the intern table is keyed by a dimension *class*, not by a user-authored
     name. It is memoization of a type constructor, as `typing`'s own
     subscription cache is — not the name-keyed registry this ADR rejects.
   - `Staggered[KDim].__qualname__` contains brackets, which
     `pickle.save_global` cannot look up, so `copyreg` is registered on the
     staggered metaclass. It must fall back to by-reference pickling for the bare
     base, which is also an instance of that metaclass.

### Cartesian axes are a level of the hierarchy

A **Cartesian axis** is an index space with integer index arithmetic and exactly one
staggered partner — the dimensions `CartesianConnectivity` acts on. One axis of a
Cartesian grid is a 1-dimensional cell complex with exactly two cell classes; a
declared axis and its `Staggered[...]` name those two, and `Staggered` is the
involution that swaps them. Mesh locations are not axes: an unstructured mesh does
not factor into per-axis cell classes, so it has no half cells and no index
arithmetic. Two levels below the root encode this:

```
DimensionIndex                            # the root; every `type[DimensionIndex]` keeps its meaning
├── AnyCartesianAxisIndex                 # either cell class of a Cartesian axis
│   ├── CartesianAxisIndex                # a declared axis: what users subclass
│   └── Staggered[D: CartesianAxisIndex]  # its derived partner
└── (direct subclasses)                   # mesh locations, and index spaces without geometry
```

Both levels sit *below* `DimensionIndex`, so `Staggered[K]` stays a `DimensionIndex`
and no annotation or `issubclass` guard in the tree widens. The bound is on the
*declared* level, and `Staggered[K]` is only an `AnyCartesianAxisIndex`, so:

| Rejected                                        | Statically (mypy, pyright) | At runtime                                    |
| ----------------------------------------------- | -------------------------- | --------------------------------------------- |
| `Staggered[Staggered[K]]`                       | `[type-var]`               | `TypeError`                                   |
| `Staggered[Cell]`, staggering a local dimension | `[type-var]`               | `TypeError`                                   |
| `Cell + 1`, `Cell - 1`                          | `[operator]`               | `TypeError`; a `DSLError` in a field operator |

The last row needs `DimensionMeta.__add__` / `__sub__` declared with the self-type
`cls: type[AnyCartesianAxisIndex]`. Both checkers bind it correctly at every call
site and both reject it at the definition site, with different diagnostics (mypy
`[misc]`, pyright `reportGeneralTypeIssues`), so it costs two separately spelled
suppressions. The runtime check covers unannotated code; hand-written iterator IR,
which names dimensions by tag, is not checked. `as_offset(dim, field)` needs index
arithmetic too and takes an `AnyCartesianAxisIndex`. Comparisons are deliberately *not* restricted: `D == n`
and `D < n` build a `Domain` on every dimension, as `concat_where` over a mesh
location requires.

Whether a dimension is an axis or a mesh location is a decision per declaration —
`IDim` and `Cell` are both `HORIZONTAL`, and only the first is an axis — so it cannot
be derived from `kind`. `CartesianAxisIndex` and `AnyCartesianAxisIndex` are both
exported as `gtx.*`.

### `Dimension` is annotation-only

`common.Dimension` becomes a PEP 695 alias for `type[DimensionIndex]`, so the
removed `gtx.Dimension("I")` spelling raises rather than misbehaving: a plain
`TypeAlias` for `type[X]` is a `types.GenericAlias`, and calling one forwards to
`__origin__` while discarding the arguments — it would evaluate to `str` with no
error. A `TypeAliasType` is simply not callable.

Its cost is that `get_origin()` of such an alias is `None`, so a site dispatching
on an annotation's shape must resolve it first (`xtyping.resolve_annotation`,
added in #2841).

### Naming

A dimension's name is `.tag` (typed `common.Tag`, which already existed) and
`.value` keeps its meaning as the index position, on the *instance*. The reverse
split does not type-check at all — an instance attribute cannot shadow a
`ClassVar` — and this direction leaves every index expression untouched.
Reading `.value` on a dimension *class* raises a metaclass `AttributeError`
pointing at `.tag`, rather than returning the `__slots__` member descriptor and
surfacing much later as a missing offset-provider key.

`tag` is a metaclass **property**, so it cannot drift from the type. A class-body
`tag = "..."` would therefore be silently ignored, which is exactly the renaming
pattern downstream code uses — so `__init_subclass__` raises on it.

### Display uses the unqualified name

The `tag` is qualified, but it is *identity and IR spelling*, not a display name.
User-facing diagnostics use `cls.__qualname__`, so `Field[[IDim], float64]` reads
the same as before instead of becoming `Field[[pkg.mod.IDim], float64]`. This is
not a third name concept: `__qualname__` is a Python builtin attribute, and for a
nested declaration it is already the readable form (`V2E.Local`). `repr()` shows
the full tag and the kind, which disambiguates the rare case of two same-named
dimensions from different modules appearing in one message.

Ordering follows the same rule, and it matters more than it looks: `order_dimensions`
determines a field's *canonical dimension order*. Keying it on `tag` would make
that order depend on **which module each dimension is declared in**, so moving a
declaration would silently reorder a field's dimensions. It is therefore keyed on
the unqualified name, with `tag` only breaking ties between same-named dimensions
from different modules so the order stays total.

### Equality between dimensions is identity

`DimensionMeta.__eq__` stays, for the `I == 5` → `Domain` overload that
`concat_where` uses, but it does not compare dimensions: for a dimension operand it
returns `NotImplemented`, the reflected call does the same, and Python falls back to
identity. So `I == J` is always a `bool`, and it is `True` only for the same class —
"equality is `is`" holds for every dimension operand. The one non-`bool` result is
`dim == <integer>`, which builds a `Domain`, and `Domain.__bool__` raises. A dict
lookup reaches that path only if an integer key and a dimension-class key share a
hash bucket in one dict; class-keyed mappings in the tree (domains, offset
providers) mix classes with strings at most, and `str` against a class compares
`False`. That residual hazard is accepted.

`DimensionMeta` must declare `__hash__ = type.__hash__` explicitly: Python sets
`__hash__ = None` on any class body defining `__eq__` without it. Without it every
dimension class is unhashable and `ts.DimensionType` fails at *import*.

## Consequences

- `common.NamedIndex` is deleted; `.dim` and `.value` keep working, on the index
  instance.
- The dimension half of `mypy_plugin.py` is deleted; only the mixed-precision
  hooks remain. Static checking is now available to pyright as well as mypy.
- Every declaration in the tree — including docs, workshop notebooks and
  `examples/`, which `test_examples` executes — becomes a class statement. There
  is no `dimension(tag, kind)` factory for user code, so there is no minimal
  migration form.
- Test modules that declared dimensions inside test functions must move them to
  module level. Where two same-named function-local dimensions were silently the
  same dimension, they are now distinct — each resulting failure is a real
  finding.
- `repr()` of a dimension is `I[horizontal]`; `str()` is unchanged, so error
  messages are byte-identical.

## Alternatives considered

- **`(name, kind)` value equality with an interning registry.** See Decision. The
  test-tree consequence of rejecting it is real: many function-local dummy
  dimensions must move to module level.
- **Keeping the mypy plugin.** Serves one checker, caps distinct dimensions at
  four per run, and blocks `TypeVar`s over dimensions.
- **`AxisLiteral` carrying the dimension class** instead of `value: str`. Would
  remove `resolve` from the type-inference hot path, but changes IR node shape in
  the same step as the dimension rewrite. Deferred.
- **`Staggered` as a PEP 695 generic.** Does not produce a class; see Decision 6.
- **A sibling root above `DimensionIndex`** for the staggered dimensions. Gives the
  same static ban on double staggering, but `Staggered[K]` stops being a
  `DimensionIndex`, so every annotation and `issubclass` guard that must accept a
  staggered dimension widens. The axis levels sit below the root instead.
- **A structural discriminator** (a `Protocol` with a `Literal[False]` class
  variable the staggered class overrides). Needs a suppressed incompatible
  override, gives an opaque diagnostic, and buys only the double-staggering ban:
  with no axis concept, `Cell + 1` and `Staggered[Cell]` stay unchecked.
- **`Staggered[D: AnyCartesianAxisIndex]`**, bounding on either cell class. Readmits
  `Staggered[Staggered[K]]`; the bound has to name the declared level.
- **A second partner constructor** for the other alignment (`i + 1/2` instead of
  `i - 1/2`). Gives three cell classes per axis where an axis has two and makes
  `flip_staggered` partial. Either alignment is already reachable by choosing which
  member of the pair to declare.

## References

- Implements the `shared/dimensions-as-types` proposal (gt4py_knowledge#27,
  @havogt) and the `egparedes/connectivities-as-types` proposal
  (gt4py_knowledge#32).
- Closes the static-typing gap reported in #2503.
- Supersedes the `_Staggered` name-prefix mechanism of
  [ADR 0026](0026-Staggered_Dimensions.md); the indexing convention there is
  unchanged.
- Consequence for the build cache of [ADR 0023](0023-Fingerprinting.md).
- The Cartesian axis levels follow the specification in gt4py_knowledge#36.
- An alternative to #2844 (closed, superseded), which implemented the same
  class-shaped dimension with value identity.
