---
tags: [typing, datamodels]
---

# Runtime Resolution of PEP 695 Type Aliases

- **Status**: valid
- **Authors**: Enrique González Paredes (@egparedes)
- **Created**: 2026-08-05
- **Updated**: 2026-08-06

In the context of users writing `type MyField = Field[...]` in datamodel and DSL annotations, facing the fact that `TypeAliasType` is an opaque object which no runtime introspection helper unwraps, we decided to resolve aliases at the existing annotation-dispatch funnels through a single `eve.extended_typing.eval_type_alias` helper, to achieve uniform support at every nesting depth without eagerly rewriting stored annotations, accepting that an alias whose value is not yet defined is only validated on first instantiation.

## Context

Python 3.12 (PEP 695) introduces `type X = ...`, which creates a `typing.TypeAliasType` object instead of binding the annotation directly. The GT4Py Python floor is now 3.12, so users can and will write these aliases.

Three properties make this awkward at runtime:

- `typing.get_type_hints` (and therefore `eve.extended_typing.get_partial_type_hints`) returns the `TypeAliasType` object unresolved. It never evaluates `__value__`, so the existing `NameError`-to-`ForwardRef` fallback never triggers for aliases.
- `__value__` is evaluated lazily and raises `NameError` until every name it mentions exists. An alias may legitimately be defined before its target.
- `gt4py.eve.extended_typing` re-exports both `typing` and `typing_extensions` with star imports, so the name `TypeAliasType` resolves to `typing_extensions.TypeAliasType`. That class is _not_ the class of a native `type X = ...` alias, so a plain `isinstance(x, xtyping.TypeAliasType)` check silently fails to match. Additionally, `MyGenericAlias[int]` is a `types.GenericAlias` which proxies attribute lookups to its origin, so `hasattr(x, "__value__")` is true for it as well while `isinstance()` is not.

Drivers:

- Nested annotations (`list[MyAlias]`, `tuple[MyAlias, ...]`, `Optional[MyAlias]`) have to work, not only top-level ones.
- Neither `gt4py.eve` nor `gt4py.next` should grow scattered `isinstance` checks against a moving target.
- The failure mode for a genuinely broken annotation has to stay a normal, understandable error.

Note that PEP 695 _generics_ (`class Model[T](DataModel)`) already work unchanged, because `__parameters__` is still populated for them. Only the alias form is affected.

## Decision

1. `gt4py.eve.extended_typing` owns the concept, exposing `is_type_alias(obj)` (checking both the `typing` class and the `typing_extensions` backport) and `eval_type_alias(annotation)`. The latter follows alias chains, substitutes type parameters for parametrized generic aliases, returns the identical object for non-aliases, and raises `TypeError` for recursive or over-nested aliases.

2. Resolution happens at the **annotation-dispatch funnels**, not at annotation-storage time: `eve.type_validation.SimpleTypeValidatorFactory.__call__`, `eve.datamodels.core._make_type_converter` and `eve.extended_typing.get_represented_types`. All three already recurse into type arguments, so nesting is covered for free, and the stored annotation keeps the alias name for reprs and introspection.

   A funnel left out of this list does not raise: it falls through to whatever its no-match branch is. For `get_represented_types` that branch returns an empty tuple, which turns every downstream `isinstance()` against the result into a constant `False`. When adding a new consumer that dispatches on annotation shape, wire it in here.

3. A `NameError` from `eval_type_alias` is a **signal, not an error** inside `eve`: `type_validation` lets it propagate, and `datamodels.field_type_validator_factory` catches it and installs the existing `ForwardRefValidator`, deferring validation to the first instantiation, which is the same treatment string forward references already get. `datamodels._make_type_converter` does the same with a `DeferredTypeConverter`, so coerced and validated fields defer alike.

4. Every **other** exception is wrapped by `eval_type_alias` into a `TypeError` naming the alias and the underlying cause. An alias value is arbitrary user code evaluated on first access, so it can fail in any way — a typo'd dtype (`type F = Field[Dims[I], np.foat64]`) raises `AttributeError`. Only `NameError` means "not defined _yet_"; anything else will never resolve, so it must fail where it is found instead of being deferred to a later instantiation that reports it worse.

   This is what lets consumers keep a single narrow `except`: `next.type_translation._resolve_type_alias` turns the `TypeError` into the `ValueError` that `type_from_annotation` already converts into a located `InvalidAnnotationError`, so a broken alias reaches the user as a DSL diagnostic pointing at the annotation rather than as a raw traceback. Consumers propagate the wrapped message verbatim instead of re-wording it, since it is the only text naming the actual cause.

## Consequences

Easier:

- Any annotation reachable through the dispatch funnels supports aliases at any nesting depth with no further work.
- Only one place has to change if CPython or `typing_extensions` alters the alias implementation.

Harder, and accepted:

- A field annotated with an alias whose value is not yet resolvable is validated on first instantiation instead of at class definition, so its errors surface later than for other fields. Its error message also reports the bare field name rather than the qualified `Model.field` one, matching the pre-existing behavior for forward references.
- `ClassVar` hidden behind an alias (`type CV = ClassVar[int]`) is not detected as a class variable by `datamodels`. This is not supported and no attempt is made to detect it.
- The resolution depth is capped (64 steps) so that recursive aliases fail fast instead of hanging.
- Support is for aliases used **as annotations**. An existing `X: TypeAlias = SomeClass` which is also used as a runtime value — as a base class, a constructor, or an `isinstance()` argument — cannot be mechanically rewritten to `type X = ...`, because a `TypeAliasType` is not the class it stands for. `common.Tag` (subclassed at `iterator/embedded.py:102`) is one such case. This fails loudly at import, so it is not a silent hazard, but it does mean ruff's `UP040` cannot be enabled repo-wide.

## Alternatives considered

### Rewrite resolved annotations eagerly in `_make_datamodel`

Unwrap aliases right after `get_partial_type_hints` and store the resolved annotation on the field.

- Good, because every downstream consumer sees a plain annotation with no further changes.
- Bad, because it forces eager evaluation of `__value__`, breaking aliases which reference names defined later.
- Bad, because it destroys the alias name in `__annotations__`, field reprs and generated `__init__` signatures.

### `isinstance` checks at each call site

- Good, because it needs no new API.
- Bad, because the correct check is non-obvious (two distinct classes, and `hasattr` is a trap) and would be re-derived wrongly at each site.
- Bad, because it goes against the ruff `typing-modules = ['gt4py.eve.extended_typing']` convention that typing concepts live in a single module.

## References

- [PEP 695 - Type Parameter Syntax](https://peps.python.org/pep-0695/)
- `src/gt4py/eve/extended_typing.py` (`is_type_alias`, `eval_type_alias`)
