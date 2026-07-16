# @gtscript.lazy_function()

In the context of extending `gt4py.cartesian` with run-time dependent types, we needed a way to defer the annotation of `gtscript.function`s and inject code before that happens. We decided to extend the `gtscript` language with a `lazy_function(before_annotation=None, after_annotation=None)` decorator allowing us to do so.

## Context

NDSL is extending `gt4py.cartesian` with run-time dependent types, which we would like users to be able to use as type hints in `gtscript.stencil` and `gtscript.function` definitions. To do so, we have a registry in place that lets users declare run-time dependent types. These declarations result in intermediate "mockup types", which can be resolved to the real type after it has been registered.

The above declare/register approach works well for stencils because we deny users the possibility to use the `@stencil` decorator. Instead, we force usage of a "stencil factory" that allows us to inject the real types before forwarding stencils to gt4py for annotation.

For `gtscript.functions`, we have a problem with the above approach because `@gtscript.function` decorators are executed a python parsing time and at this time we don't have the real type information yet. This results in the inconsistent behavior that NDSL users can use run-time dependent types in stencils but not in functions.

## Decision

We decided to extend the `gtscript` language with a `@gtscript.lazy_function(before_annotation=None, after_annotation=None)` allowing user extending `gt4py.cartesian` to delay the annotation process of those functions until they are actually used from within a stencil or another function.

## Consequences

This change expands the `gtscript` language, which comes with maintenance overhead and allows power users of `gt4py.cartesian` to intercept function annotations before and after annotation happens.

In NDSL, this allows us to default to `gtscript.lazy_functions(before_annotation=resolve_deferred_types)` allowing NDSL to inject code to automatically resolve deferred types.

## Alternatives considered

### Avoid using run-time dependent types

Avoiding types that are run-time dependent solves many issues. It is, however, the given scope of this problem. One could forbid usage of run-time dependent type annotations in `gtscript.function`s. In NDSL, this is, however, inconsistent with the seemingly allowed usage in stencil code and in general forces more mental load to `gt4py` users by having to mentally map untyped function parameters.

### Extend the current `gtscript.function`

It might be possible to extend the existing `gtscript.function` decorator with a flag to distinguish eager and delayed execution. One could argue that this would require less duplicated code. However, I'd argue the bulk of the work is adding support in the frontend, which needs to happen in either case.

## References

Related NDSL issue: <https://github.com/NOAA-GFDL/NDSL/issues/504>.
