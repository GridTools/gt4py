---
tags: []
---

# Dimensions as Nominal Types

- **Status**: proposed
- **Authors**: Enrique González Paredes (@egparedes)
- **Created**: 2026-09-18
- **Updated**: 2026-09-18

A concrete dimension becomes a **class**, and an index along it an **instance** of
that class — the shape `enum.Enum` uses, where the class is the collection and
the instances are its members:

```python
class IDim(gtx.DimensionIndex): ...


class KDim(gtx.DimensionIndex, kind=gtx.DimensionKind.VERTICAL): ...


IDim  # the dimension    -- annotated `gtx.Dimension`
IDim(0)  # an index into it -- annotated `IDim`
```

so `gtx.Field[gtx.Dims[IDim], gtx.float64]` is valid for any PEP 484 checker with
no gt4py mypy plugin.

A dimension's **identity is the Python type**, and its `tag` — the string that
crosses into the IR and the generated code — is its **qualified Python name**,
`f"{cls.__module__}.{cls.__qualname__}"`.

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
names into one declaration (a follow-up ADR covers the connectivity half).

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
   references a Python type exactly the way `pickle` references a class. It is
   memoized, because type inference calls it once per `AxisLiteral`.

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

   **Known limitation**: interactive `__main__` — the REPL, notebooks,
   `python -c` — cannot be resolved. The `spawn`-based compile workers
   re-execute the main *script* as `__mp_main__`, so a dimension declared in a
   file's `__main__` does resolve, provided the script has the
   `if __name__ == "__main__":` guard the worker pool already requires.

3. **No registry and no blanket `copyreg`.** Module-level classes pickle by
   reference, which is correct. A *narrow* `copyreg` registration is still
   required for parametrized dimensions — see Staggered below.

4. **Cache fingerprints depend on module paths.** A dimension is fingerprinted by
   qualified name, so moving a declaration between modules invalidates compiled
   artifacts. This is a consequence for the build cache of ADR 0023, not a
   reversal of it. The generic `type` deconstructor is correct for the *lenient*
   fingerprint variant; the STRICT variant rejects a parametrized dimension,
   which is not importable under its qualified name.

5. **Generated identifiers need injective mangling.** A dot is illegal in a C++
   identifier, in a DaCe symbol, and in `eve`'s `SymbolName`
   (`^[a-zA-Z_]\w*$`). One shared pair, used by every backend:

   ```python
   def codegen_name(tag: Tag) -> str:
       return tag.replace("_", "_u").replace(".", "_d")


   def from_codegen_name(name: str) -> Tag:
       return re.sub(r"_([ud])", lambda m: "_" if m.group(1) == "u" else ".", name)
   ```

   A *prefix* escape, not `_ -> __` followed by `. -> _`: the latter is **not
   injective**, since a dot becomes a single underscore and `".."` collides with
   an escaped `"_"`. Every `_` in the output is the first character of a
   two-character escape, so decoding is unambiguous. Names grow, which is what
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

`DimensionMeta` must declare `__hash__ = type.__hash__` explicitly: Python sets
`__hash__ = None` on any class body defining `__eq__` without it, and `__eq__`
stays for the `I == 5` → `Domain` overload. Without it every dimension class is
unhashable and `ts.DimensionType` fails at *import*.

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

## References

- Implements the `shared/dimensions-as-types` proposal (gt4py_knowledge#27,
  @havogt) and the `egparedes/connectivities-as-types` proposal
  (gt4py_knowledge#32).
- Closes the static-typing gap reported in #2503.
- Supersedes the `_Staggered` name-prefix mechanism of
  [ADR 0026](0026-Staggered_Dimensions.md); the indexing convention there is
  unchanged.
- Consequence for the build cache of [ADR 0023](0023-Fingerprinting.md).
- An alternative to #2844, which implements the same class-shaped dimension with
  value identity.
