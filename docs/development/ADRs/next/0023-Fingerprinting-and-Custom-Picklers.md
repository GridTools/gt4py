---
tags: []
---

# [Fingerprinting and Custom Picklers]

- **Status**: valid
- **Authors**: Enrique González Paredes (@egparedes)
- **Created**: 2026-06-09
- **Updated**: 2026-06-09

In the context of identifying GT4Py objects (frontend stages, IR nodes, OTF
workflow steps, ...) for caching and content-based comparison, facing a
collection of ad-hoc, per-class fingerprinting routines that were hard to keep
consistent and stable across interpreter runs, we decided to build a single
general fingerprinting mechanism on top of `pickle` + a content hash, and to
customize *what* gets fingerprinted through two complementary, declarative
hooks: **metadata-based picklers** (which tailor the *state* of an object via
`__getstate__`) and **manually defined custom picklers** built with
`functools.singledispatch` (which tailor the *serialization* of whole type
hierarchies via `reducer_override`). We considered keeping the bespoke
`fingerprint()` methods and the `add_content_to_fingerprint` single-dispatch
visitor, and accept that everything we fingerprint must now be picklable (or be
made picklable through a custom reducer).

## Context

A *fingerprint* is a stable, content-based identifier of an object: two objects
with the same semantically relevant content must produce the same fingerprint,
and the value must be reproducible across interpreter restarts (so it can key a
persistent build cache). GT4Py needs fingerprints in several places:

- The OTF build cache (`CachedStep`) keys compiled artifacts by a fingerprint
  of the workflow step **and** its input, so a cached compiled program can be
  reused only when both are unchanged.
- The frontend stages (FOAST/PAST definitions) and the iterator/GTIR nodes are
  fingerprinted to dispatch and cache toolchain steps.
- Several passes generate deterministic symbol names from a node fingerprint.

Before this change, fingerprinting was scattered and inconsistent:

- `eve.Node` subclasses carried a bespoke `fingerprint()` method, and ffront /
  iterator IR each built their own pickler via a now-removed
  `eve.concepts.skipping_fields_node_pickler` helper.
- `ffront/stages.py` had a hand-written `functools.singledispatch` visitor
  (`add_content_to_fingerprint`) that had to be extended, type by type, for
  every new stage and every new field type it might contain.
- `CachedStep` keyed its cache with `hash_function=hash`. The builtin `hash` is
  **not** stable across interpreter runs (it is salted for `str`/`bytes` and is
  identity-based for most objects), so it could not safely key a persistent
  cache, and it did not fingerprint the *step* itself at all — only the input.

The drivers for a redesign were therefore:

- **Stability**: a fingerprint must be deterministic across processes and across
  Python versions, independent of dict/set insertion order and of object
  identity.
- **One mechanism**: avoid one fingerprinting path per subsystem; reuse the same
  core everywhere and customize it locally.
- **Selective content**: some fields must be excluded from the fingerprint
  (source locations, inferred types that are filled in later, non-semantic
  caches), and some values are not naturally (or stably) picklable
  (functions, types, modules).
- **Cache-version safety**: a way to globally invalidate cached builds when the
  toolchain changes in a way the fingerprint cannot capture.

## Decision

We standardize on **content hashing via pickling** and expose two orthogonal
ways to customize the bytes that are hashed.

### Core: content hashing via pickling

`eve.utils.content_hash(*args, pickler_type=...)` pickles its arguments with the
given `pickler_type` into a byte buffer and hashes the bytes (default
`xxhash.xxh64`). A *fingerprinter* is just `content_hash` bound to a specific
pickler:

```python
fingerprinter = functools.partial(content_hash, pickler_type=SomePickler)
```

Pickling is a good substrate because it already knows how to recursively
traverse arbitrary object graphs; we only need to *steer* it. Steering happens
on the two ends of the pickle protocol: the object's own `__getstate__`, and the
pickler's `reducer_override`.

### Customizing object *state*: metadata-based picklers

`gt4py.next.utils.MetadataBasedPicklingMixin` gives dataclass-like classes a
`__getstate__` that returns the same shape of state as the default pickle
implementation, **but** drops any field annotated with `pickle=False` in the
`GT4PY_META` field-metadata namespace:

```python
@dataclasses.dataclass(frozen=True)
class CachedStep(..., utils.MetadataBasedPicklingMixin):
    step: Workflow[StartT, EndT]
    key_function: Callable[..., HashT] = dataclasses.field(
        metadata=utils.gt4py_metadata(pickle=False)
    )
    cache: ... = dataclasses.field(
        default_factory=dict, metadata=utils.gt4py_metadata(pickle=False)
    )
```

Here the *runtime* fields (`key_function`, `cache`) are excluded from the
object's pickled state, so they do not pollute the fingerprint — only the
semantically meaningful `step` does. This is a **declarative, per-class** opt-out
that the object controls itself; `_get_metadata_based_state_getstate` derives and
caches the appropriate `__getstate__` from the field metadata and the class's
`__slots__`/`__dict__` layout, and we deliberately keep the default
`__setstate__` so the customized state stays round-trippable.

The mixin is applied broadly to the objects we fingerprint and cache: workflow
steps (`NamedStepSequence`, `MultiWorkflow`, `StepSequence`, `CachedStep`, ...),
the ffront stage classes (via a shared `BaseStage`), `CompileTimeArgs`, the OTF
`ProgramSource`/`BindingSource`/`CompilableProject`, the backends, etc.

### Customizing *serialization*: custom picklers via single dispatch

Object-controlled state is not enough for two cases: (a) types we do not own
(`dict`, `set`, `type`, `FunctionType`, `ModuleType`) and (b) types whose
*default* serialization is not stable or not semantic. For these we attach a
`reducer_override` to a `pickle.Pickler` subclass, implemented as a
`functools.singledispatch` callable so that a registered reducer applies to a
type **and all of its subclasses** (unlike the exact-type matching of pickle's
`dispatch_table`). The default implementation returns `NotImplemented`, which
makes pickle fall back to its normal handling.

Three small building blocks in `eve.utils` make these picklers easy to assemble:

- `singledispatcher(default, implementations={type: reducer, ...})` — build a
  single-dispatch reducer from a registry.
- `pickle_reducer_factory(get_new=..., get_new_args=..., get_state=...)` — build
  one reducer (the `(callable, args, state)` tuple pickle expects) from small
  lambdas, so the common case is a one-liner.
- `merge_dispatchers(*dispatchers, default=...)` — compose the registries of
  several single-dispatch reducers into one, which is what lets picklers be
  built up in layers (see below).

Concrete picklers built this way:

- `StableContainerPickler` — sorts `dict` items and `set` elements (so insertion
  order does not change the fingerprint) and serializes `type`/`function`/`module`
  **by fully qualified reference** instead of by value (a function/type is
  identified by its import path, avoiding both instability and infinite reducer
  recursion). It drives the default `stable_fingerprinter`.
- `skipping_fields_node_fingerprinter("location", "type", ...)` — a cached
  factory returning a fingerprinter (and optionally its pickler) for `eve.Node`s
  that recursively skips the named child fields. The iterator/GTIR
  `semantic_fingerprinter` skips `location` and `type` (types are filled in later
  by inference and must not change identity); the FOAST `semantic_fingerprinter`
  skips `location`.
- `SemanticPickler` (ffront stages) — fingerprints DSL stage objects, serializing
  a `types.FunctionType` by its **source code + closure variables** rather than by
  reference, which is what makes two textually identical field operators
  fingerprint equal.

> Implementation note: the picklers that must override built-in containers
> (`StableContainerPickler`, `SemanticPickler`) subclass `eve.utils.PurePickler`
> (the pure-Python `pickle._Pickler`). The accelerated C `pickle.Pickler` has
> fast paths that bypass `reducer_override` for built-ins like `dict`/`set`,
> whereas the pure-Python pickler routes every object through it. Picklers that
> only override our own `Node` types can keep using the faster C `pickle.Pickler`.

### Composing picklers

Because reducers are single-dispatch registries, picklers compose by merging.
The ffront `SemanticPickler` is assembled from the stable-container reducers, the
node-skipping reducers, and a function-by-source reducer:

```python
_base = merge_dispatchers(
    StableContainerPickler.reducer_override, foast.semantic_pickler.reducer_override
)
reducer = merge_dispatchers(
    _base,
    singledispatcher(
        default=...,
        implementations={
            types.FunctionType: pickle_reducer_factory(get_state=lambda f: (source, closure)),
        },
    ),
)
```

This is the key ergonomic win: a subsystem describes only the *deltas* it needs
on top of the shared, stable base.

### Integration with the build cache

`CachedStep.cache_key(inp)` ties the pieces together:

```python
def cache_key(self, inp):
    return utils.stable_fingerprinter((config.BUILD_CACHE_VERSION_ID, self, self.key_function(inp)))
```

- `self` is the (metadata-pickled) step, so changing the step invalidates the
  cache. This requires the step — and any function it wraps — to be picklable
  (e.g. a module-level function, not a lambda or local closure).
- `self.key_function(inp)` contributes the input's identity.
- `config.BUILD_CACHE_VERSION_ID` (defaulting to the gt4py version, overridable
  via the `BUILD_CACHE_VERSION_ID` env var) is a global salt that lets us force
  incompatibility with previously cached builds when the toolchain changes in a
  way the fingerprint alone would not capture.

## Consequences

What becomes easier:

- One fingerprinting mechanism is reused everywhere; adding a new cached/compared
  type usually means either applying `MetadataBasedPicklingMixin` (and marking
  non-semantic fields `pickle=False`) or registering a reducer — no new bespoke
  `fingerprint()`/visitor code.
- Fingerprints are stable across processes and tolerant of dict/set ordering, so
  they can safely key a persistent build cache.
- Cross-subsystem reuse: picklers are layered with `merge_dispatchers`, so the
  stable-container behavior is shared rather than re-implemented.
- The build cache can be globally invalidated through a single version id.

What becomes harder / the trade-offs:

- Everything fed to a fingerprinter must be picklable, or a reducer must be
  provided for it. Non-picklable values (lambdas, local closures, live runtime
  objects) must be excluded via metadata or handled by a custom reducer. This is
  an explicit constraint now baked into `CachedStep`'s contract.
- The customization lives in two places (object `__getstate__` and pickler
  `reducer_override`); contributors need to know which knob to reach for —
  metadata for "which of *my* fields count", a reducer for "how a *type* (often
  one I don't own) is serialized".
- Correctness of caching now depends on reducers being faithful: a reducer that
  drops semantically relevant content can cause false cache hits. The defaults
  (skip `location`/`type`, serialize functions by source) are chosen
  conservatively for this reason.

## Alternatives considered

### Keep per-class `fingerprint()` methods + the `add_content_to_fingerprint` visitor

- Good, because it is explicit and needs no pickling.
- Bad, because it must be extended type-by-type for every new stage/field, it
  duplicated traversal logic that pickle already provides, and it produced
  separate, easily-diverging fingerprinting paths per subsystem.

### Use the builtin `hash` (previous `CachedStep.hash_function=hash`)

- Good, because it is trivial and fast.
- Bad, because `hash` is not stable across runs (salted strings, identity-based
  objects), so it cannot key a persistent cache, and the old design did not
  fingerprint the step itself at all.

### `pickle`'s `dispatch_table` instead of a single-dispatch `reducer_override`

- Good, because it is the documented pickle extension point.
- Bad, because it matches on exact type only; we explicitly want a reducer to
  apply to an entire class hierarchy (e.g. all `eve.Node` subclasses, all
  `dict` subclasses), which single dispatch gives us for free.

### A third-party deterministic hashing library (e.g. `deepdiff.DeepHash`)

- Good, because it handles many container types out of the box.
- Bad, because we need fine control over per-field inclusion and over
  by-reference vs. by-value serialization of functions/types, which is awkward to
  express through an external hasher; pickle + reducers keeps the control local
  and composable. (`deepdiff` remains available in `eve.utils` for diffing.)

## References

- `src/gt4py/next/utils.py` — `MetadataBasedPicklingMixin`, `gt4py_metadata`,
  `StableContainerPickler`, `stable_fingerprinter`,
  `skipping_fields_node_fingerprinter`.
- `src/gt4py/eve/utils.py` — `content_hash`, `singledispatcher`,
  `merge_dispatchers`, `pickle_reducer_factory`, `PurePickler`.
- `src/gt4py/next/ffront/stages.py` — `SemanticPickler`, `semantic_fingerprinter`.
- `src/gt4py/next/otf/workflow.py` — `CachedStep.cache_key`.
- `src/gt4py/next/config.py` — `BUILD_CACHE_VERSION_ID`.
- ADR [0011 - On The Fly Compilation](0011-On_The_Fly_Compilation.md) and
  [0017 - Toolchain Configuration](0017-Toolchain-Configuration.md) for the OTF
  toolchain and build cache this mechanism feeds.
