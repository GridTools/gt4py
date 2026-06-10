---
tags: []
---

# [Fingerprinting]

- **Status**: valid
- **Authors**: Enrique González Paredes (@egparedes)
- **Created**: 2026-06-09
- **Updated**: 2026-06-10

In the context of identifying GT4Py objects (frontend stages, IR nodes, OTF
workflow steps, ...) for caching and content-based comparison, facing a
collection of ad-hoc, per-class fingerprinting routines that were hard to keep
consistent and stable across interpreter runs, we decided to build a single
general **structural fingerprinting** mechanism — an iterative Merkle-style
content hash over the object graph — customized through per-type
**fingerprint handlers** and declarative **field metadata**. We considered
keeping the bespoke `fingerprint()` methods and the
`add_content_to_fingerprint` single-dispatch visitor, as well as hashing a
pickle byte stream produced by custom picklers, and accept that objects fed to
a fingerprinter must either match a handler, be a dataclass/datamodel, or
support the standard `__reduce_ex__` protocol.

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

- **Stability**: a fingerprint must be deterministic across processes,
  independent of dict/set insertion order and of object identity.
- **One mechanism**: avoid one fingerprinting path per subsystem; reuse the same
  core everywhere and customize it locally.
- **Selective content**: some fields must be excluded from the fingerprint
  (source locations, inferred types that are filled in later, non-semantic
  caches), and some values must be identified by reference rather than by value
  (functions, types, modules).
- **Robustness**: real inputs are large — lowered IR trees can nest far deeper
  than the Python recursion limit allows for recursive traversals.
- **Cache-version safety**: a way to globally invalidate cached builds when the
  toolchain changes in a way the fingerprint cannot capture.

## Decision

We standardize on a **structural fingerprinter**: a single iterative,
post-order traversal of the object graph that hashes content bottom-up,
Merkle-tree style. Each object contributes either a **leaf** (a raw byte
payload) or a **composite** (a tag plus children, whose digests are combined —
sorted first if the container is unordered). The implementation lives in
`gt4py.next.utils`.

### Core: decomposition + bottom-up hashing

A *fingerprint handler* decomposes an object into its contribution:

```python
FingerprintLeaf(payload: bytes)                              # terminal
FingerprintComposite(tag: bytes, children: tuple, ordered)   # recurse into children
```

`make_fingerprinter(extra_handlers)` builds a fingerprinting function from the
default handler registry plus optional per-type overrides (dispatched on the
object's MRO via `functools.singledispatch`). The default rules are:

- **Primitives** (`int`, `float`, `str`, `bytes`, ...) — leaves tagged with
  their concrete class, so `True`, `1` and `1.0` do not collide.
- **`tuple`/`list`** — ordered composites.
- **`dict`/`set`/`frozenset`** — *unordered* composites: the digests of the
  items are sorted before combining, so insertion order (and
  `PYTHONHASHSEED`-dependent set iteration order) never leaks into the
  fingerprint. Because the *digests* are sorted — not the values — keys do not
  need to be orderable or even comparable (`Dimension`-keyed dicts just work).
  `collections.OrderedDict`, whose equality is order-sensitive, stays ordered.
- **`type` / functions / modules** — leaves containing the fully qualified
  name: identified **by reference**, not by value. Non-importable callables
  (lambdas, local closures), whose qualified names are ambiguous, are rejected
  with a `TypeError` instead of silently colliding.
- **Dataclasses and datamodels** — composites of class + field values, honoring
  the field metadata opt-out (below).
- **Everything else** — decomposed via the standard `__reduce_ex__` protocol
  (with a pinned protocol version), which covers e.g. NumPy arrays by content
  without any special-casing.

The traversal is **iterative** (explicit work stack), so deeply nested inputs —
lowered IR trees routinely exceed the recursion limit budget of any recursive
scheme — cannot raise `RecursionError`. Digests are memoized by object
identity, which makes shared subgraphs cheap *and* keeps the fingerprint a pure
function of value: a graph that shares a sub-object and a graph with an equal
copy fingerprint identically. Self-referential structures (e.g. module-level
recursive functions appearing in their own closure variables) are handled with
relative back references.

### Selecting fields: declarative metadata

Fields of dataclasses/datamodels can opt out of fingerprinting through the
`GT4PY_META` field-metadata namespace:

```python
@dataclasses.dataclass(frozen=True)
class CachedStep(...):
    step: Workflow[StartT, EndT]
    key_function: Callable[..., HashT] = dataclasses.field(
        metadata=utils.gt4py_metadata(fingerprint=False)
    )
    cache: ... = dataclasses.field(
        default_factory=dict, metadata=utils.gt4py_metadata(fingerprint=False)
    )
```

Here the *runtime* fields (`key_function`, `cache`) do not pollute the
fingerprint — only the semantically meaningful `step` does. This is a
declarative, per-class opt-out read directly from the field definitions; it
does **not** alter how the class pickles, so fingerprinting and real
serialization stay independent concerns.

### Customizing per type: composing handlers

Subsystems describe only the *deltas* they need on top of the shared default
rules:

- `stable_fingerprinter` — the default fingerprinter (no extra handlers); used
  by the OTF build caches.
- `skipping_fields_node_fingerprinter("location", "type", ...)` — a cached
  factory returning a fingerprinter (and optionally its handler registry) for
  `eve.Node`s that recursively skips the named child fields. The iterator/GTIR
  `semantic_fingerprinter` skips `location` and `type` (types are filled in
  later by inference and must not change identity); the FOAST
  `semantic_fingerprinter` skips `location`.
- The ffront stages `semantic_fingerprinter` composes the FOAST node handlers
  with a `types.FunctionType` handler that decomposes a DSL definition function
  into its **source code + closure variables** rather than identifying it by
  reference — which is what makes two textually identical field operators
  fingerprint equal, and recompiles when a referenced helper changes:

```python
semantic_fingerprinter = utils.make_fingerprinter(
    {
        **foast.semantic_fingerprint_handlers,
        types.FunctionType: _decompose_definition_function,
    }
)
```

### Integration with the build cache

`CachedStep.cache_key(inp)` ties the pieces together:

```python
def cache_key(self, inp):
    return utils.stable_fingerprinter((config.BUILD_CACHE_VERSION_ID, self, self.key_function(inp)))
```

- `self` is the step, decomposed by fields, so changing the step configuration
  invalidates the cache. Functions referenced by the step are identified by
  qualified name, so they must be module-level (not lambdas or local closures).
- `self.key_function(inp)` contributes the input's identity.
- `config.BUILD_CACHE_VERSION_ID` (defaulting to the gt4py version, overridable
  via the `BUILD_CACHE_VERSION_ID` env var) is a global salt that lets us force
  incompatibility with previously cached builds when the toolchain changes in a
  way the fingerprint alone would not capture.

## Consequences

What becomes easier:

- One fingerprinting mechanism is reused everywhere; adding a new
  cached/compared type usually means nothing at all (dataclasses, datamodels
  and reducible objects are covered by default), marking non-semantic fields
  `fingerprint=False`, or registering one handler.
- Fingerprints are stable across processes, tolerant of dict/set ordering and
  of object-graph sharing, and robust against arbitrarily deep inputs.
- Fingerprinting no longer interferes with real pickling: classes keep their
  standard serialization behavior.
- The build cache can be globally invalidated through a single version id.

What becomes harder / the trade-offs:

- The traversal logic is our own (~200 lines) rather than delegated to
  `pickle`; it has to be maintained and its determinism guarded by tests.
- Correctness of caching depends on handlers being faithful: a handler that
  drops semantically relevant content can cause false cache hits. The defaults
  (skip `location`/`type`, decompose functions by source) are chosen
  conservatively for this reason.
- Objects identified by reference must be importable (module-level); lambdas
  and local closures in fingerprinted positions are a hard error.

## Alternatives considered

### Keep per-class `fingerprint()` methods + the `add_content_to_fingerprint` visitor

- Good, because it is explicit and simple.
- Bad, because it must be extended type-by-type for every new stage/field, it
  duplicated traversal logic, and it produced separate, easily-diverging
  fingerprinting paths per subsystem.

### Use the builtin `hash` (previous `CachedStep.hash_function=hash`)

- Good, because it is trivial and fast.
- Bad, because `hash` is not stable across runs (salted strings, identity-based
  objects), so it cannot key a persistent cache, and the old design did not
  fingerprint the step itself at all.

### Hash a pickle byte stream steered by custom picklers (first iteration of this work)

- Good, because pickle already traverses arbitrary object graphs, and
  `reducer_override` + `__getstate__` provide hooks to steer what is serialized.
- Bad, because the serializer's machinery leaks into the fingerprint and into
  the codebase in many small ways:
  - dict/set canonicalization requires *sorting the values*, which fails for
    non-orderable or mixed-type keys, and requires the pure-Python
    `pickle._Pickler` (private API, slow) because the C pickler bypasses
    `reducer_override` for exact builtin containers;
  - the pure-Python pickler traverses recursively, so realistic IR trees
    exhaust the Python recursion limit;
  - by-reference handling of types/functions needs synthetic reduce tuples
    (an `eval`-based constructor) to avoid infinite reducer recursion — fake
    serialization for a stream that is never deserialized;
  - pickle's memo encodes object *identity* (sharing structure) into the byte
    stream, so equal values could fingerprint differently;
  - the `__getstate__`-based field opt-out changes how the classes *really*
    pickle, breaking round-trips for unrelated use cases.
- The structural fingerprinter keeps the good part (generic traversal,
  composable per-type customization, the `__reduce_ex__` protocol as a
  universal fallback) without hashing a serialization format.

### `pickle`'s `dispatch_table` instead of single-dispatch handlers

- Good, because it is the documented pickle extension point.
- Bad, because it matches on exact type only; we explicitly want a handler to
  apply to an entire class hierarchy (e.g. all `eve.Node` subclasses), which
  single dispatch gives us for free.

### A third-party deterministic hashing library (e.g. `deepdiff.DeepHash`)

- Good, because it handles many container types out of the box.
- Bad, because we need fine control over per-field inclusion and over
  by-reference vs. by-value handling of functions/types, which is awkward to
  express through an external hasher; our own handlers keep the control local
  and composable. (`deepdiff` remains available in `eve.utils` for diffing.)

## References

- `src/gt4py/next/utils.py` — `FingerprintLeaf`, `FingerprintComposite`,
  `make_fingerprinter`, `stable_fingerprinter`,
  `skipping_fields_node_fingerprinter`, `gt4py_metadata`.
- `src/gt4py/next/ffront/stages.py` — `semantic_fingerprinter`.
- `src/gt4py/next/otf/workflow.py` — `CachedStep.cache_key`.
- `src/gt4py/next/config.py` — `BUILD_CACHE_VERSION_ID`.
- ADR [0011 - On The Fly Compilation](0011-On_The_Fly_Compilation.md) and
  [0017 - Toolchain Configuration](0017-Toolchain-Configuration.md) for the OTF
  toolchain and build cache this mechanism feeds.
