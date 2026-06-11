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
**deconstructors** and declarative **field metadata**. We considered
keeping the bespoke `fingerprint()` methods and the
`add_content_to_fingerprint` single-dispatch visitor, as well as hashing a
pickle byte stream produced by custom picklers, and accept that objects fed to
a fingerprinter must either match a deconstructor, be a dataclass/datamodel, or
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

We standardize on a **structural fingerprinter**: a content hash of the object
graph computed bottom-up, Merkle-tree style. The implementation (in
`gt4py.next.utils`) is layered as a **catamorphism** — a generalized fold over
trees — keeping three concerns explicitly separate:

1. **Deconstruction** (*what is the one-level structure of an object?*): a
   per-type registry of *deconstructors*, each peeling off exactly one
   level into an `EmptyDeconstruction` or a `Deconstruction`.
2. **Traversal scheme** (*in which order are objects visited?*): `reduce_object`,
   a generic iterative post-order fold with result memoization and cycle
   support, reusable with any result type.
3. **Reduction logic** (*how do results combine?*): the *collapser* (the
   algebra of the catamorphism), a single function collapsing a
   deconstruction — whose pieces have already been collapsed into results —
   into a result; for fingerprinting, a digest.

### Layer 1: one-level deconstruction

A *fingerprint deconstructor* produces one level of an object:

```python
Deconstruction(state: bytes, pieces: tuple)        # recurse into the pieces
EmptyDeconstruction(state: bytes)                  # terminal: no pieces
OrderInsensitiveDeconstruction(state, pieces)      # pieces combine in canonical order
```

`make_deconstructor(overrides)` builds a `Deconstructor` from the default
per-type registry plus optional overrides (dispatched on the object's MRO via
`functools.singledispatch`), with a customizable `fallback` covering
dataclasses, datamodels and `__reduce_ex__`-reducible objects; the
module-level `deconstruct` is the default deconstructor.
A fingerprinter is `reduce_object` partially applied to a deconstructor and
the digest collapser (`stable_fingerprinter` uses `fingerprint_deconstructor`,
whose `fingerprint_fallback` layers the `gt4py_metadata(fingerprint=False)`
field opt-out on the default rules). The default deconstruction rules are:

- **Primitives** (`int`, `float`, `str`, `bytes`, ...) — leaves tagged with
  their concrete class, so `True`, `1` and `1.0` do not collide.
- **`tuple`/`list`** — ordered nodes.
- **`dict`/`set`/`frozenset`** — *unordered* nodes: the digests of the
  items are sorted before combining, so insertion order (and
  `PYTHONHASHSEED`-dependent set iteration order) never leaks into the
  fingerprint. Because the *digests* are sorted — not the values — keys do not
  need to be orderable or even comparable (`Dimension`-keyed dicts just work).
  `collections.OrderedDict`, whose equality is order-sensitive, stays ordered;
  `collections.defaultdict` additionally contributes its `default_factory`.
- **`type` / functions / modules** — leaves containing the fully qualified
  name: identified **by reference**, not by value. The name is resolved through
  `sys.modules` and must yield the object itself; non-importable callables
  (lambdas, local closures), whose qualified names are ambiguous, as well as
  shadowed or reassigned globals, are rejected with a `TypeError` instead of
  silently colliding.
- **Dataclasses and datamodels** — nodes of class + field values, honoring
  the field metadata opt-out (below).
- **Everything else** — deconstructed via the standard `__reduce_ex__` protocol
  (with a pinned protocol version), which covers e.g. NumPy arrays by content
  without any special-casing.

### Layer 2: the traversal scheme (`reduce_object`)

`reduce_object(obj, *, deconstructor, collapser, allow_cycles, memoize)`
reduces an object bottom-up over its one-level deconstructions. The fold is
**iterative** (explicit two-phase work stack), so deeply nested inputs —
lowered IR trees routinely exceed the recursion limit budget of any recursive
scheme — cannot raise `RecursionError`. Results are memoized by object
identity, which makes shared subgraphs cheap, and self-referential structures
are collapsed (with `allow_cycles=True`) as back references by relative depth
(results under a cycle target are never memoized, as they are
context-dependent). Before collapsing an `OrderInsensitiveDeconstruction`,
the already-collapsed piece results are put into canonical sorted order, so
container iteration order never leaks into the result (this requires the
results to be orderable).
The driver is carrier-agnostic — nothing in it knows about hashing — so it can
serve other bottom-up reductions over arbitrary objects.

### Layer 3: the digest collapser

The single fingerprint collapser hashes a deconstruction uniformly as
`xxh64("node" + state + "pieces" + piece_digests)` — an
`EmptyDeconstruction` simply contributes no piece digests, the canonical
order of order-insensitive pieces is established by the traversal layer, and
cyclic back references arrive encoded in the state by their relative depth.
Together with the identity-based memoization this keeps the fingerprint a pure
function of *value*: a graph that shares a sub-object and a graph with an
equal copy fingerprint identically.

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

### Customizing per type: composing deconstructors

Subsystems describe only the *deltas* they need on top of the shared default
rules:

- `stable_fingerprinter` — the default fingerprinter (default deconstructor); used
  by the OTF build caches.
- `skipping_fields_node_fingerprinter("location", "type", ...)` — a cached
  factory returning a fingerprinter (and optionally its deconstructor overrides) for
  `eve.Node`s that recursively skips the named child fields. The iterator/GTIR
  `semantic_fingerprinter` skips `location` and `type` (types are filled in
  later by inference and must not change identity); the FOAST
  `semantic_fingerprinter` skips `location`.
- The ffront stages `semantic_fingerprinter` composes the FOAST node deconstructors
  with a `types.FunctionType` deconstructor that deconstructs a DSL definition function
  into its **source code + closure variables** rather than identifying it by
  reference — which is what makes two textually identical field operators
  fingerprint equal, and recompiles when a referenced helper changes:

```python
semantic_fingerprinter = functools.partial(
    utils.reduce_object,
    deconstructor=utils.make_deconstructor(
        {
            **foast.semantic_fingerprint_deconstructors,
            types.FunctionType: _deconstruct_definition_function,
        },
        fallback=utils.fingerprint_fallback,
    ),
    collapser=utils.fingerprint_collapser,
    allow_cycles=True,
)
```

### Integration with the build cache

`CachedStep.cache_key(inp)` ties the pieces together:

```python
def cache_key(self, inp):
    return utils.stable_fingerprinter((config.BUILD_CACHE_VERSION_ID, self, self.key_function(inp)))
```

- `self` is the step, content by fields, so changing the step configuration
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
  `fingerprint=False`, or registering one deconstructor.
- Fingerprints are stable across processes, tolerant of dict/set ordering and
  of object-graph sharing, and robust against arbitrarily deep inputs.
- Fingerprinting no longer interferes with real pickling: classes keep their
  standard serialization behavior.
- The build cache can be globally invalidated through a single version id.

What becomes harder / the trade-offs:

- The traversal logic is our own (~200 lines) rather than delegated to
  `pickle`; it has to be maintained and its determinism guarded by tests.
- Correctness of caching depends on deconstructors being faithful: a deconstructor that
  drops semantically relevant content can cause false cache hits. The defaults
  (skip `location`/`type`, deconstruct functions by source) are chosen
  conservatively for this reason.
- Objects identified by reference must be importable (module-level); lambdas
  and local closures in fingerprinted positions are a hard error. Structurally
  fingerprinting local functions (code object + defaults + closure cells) would
  lift the restriction and is a possible follow-up, but it is only stable per
  CPython version and the loud failure is preferred until a real use case
  appears.

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

### Build the catamorphism on `optree` pytrees instead of our own driver

The layering of this design (one-level deconstruction / generic fold /
collapsers) deliberately mirrors how a fingerprinter would be written over
[`optree`](https://github.com/metaopt/optree)'s `tree_flatten_one_level`, with
`EmptyDeconstruction`/`Deconstruction` playing the role of the pytree
registry entries. Using `optree` itself was evaluated and rejected:

- Good, because the pytree registry and the one-level deconstruction of builtin
  containers come for free, and the vocabulary (leaves, nodes, `is_leaf`,
  namespaces) is established in the array ecosystem.
- Bad, because its registry is looked up by **exact type** (no MRO dispatch),
  so a single deconstructor cannot cover an entire hierarchy like `eve.Node`
  (~100+ concrete IR node classes would each need registration, per
  skipped-field-set namespace), whereas `functools.singledispatch` gives us
  that for free.
- Bad, because `tree_flatten_one_level` is implemented in Python (a registry
  lookup plus a Python `flatten_func` call), so a node-by-node fold cannot
  benefit from `optree`'s C++ bulk traversal — which is also depth-capped
  (`MAX_RECURSION_DEPTH = 1000`) and has no memoization or cycle support, as
  pytrees are acyclic by definition.
- Bad, because it would add a compiled (pybind11) runtime dependency for what
  our own driver covers in well under a hundred lines of standard library code.

### `pickle`'s `dispatch_table` instead of single-dispatch deconstructors

- Good, because it is the documented pickle extension point.
- Bad, because it matches on exact type only; we explicitly want a deconstructor to
  apply to an entire class hierarchy (e.g. all `eve.Node` subclasses), which
  single dispatch gives us for free.

### A third-party deterministic hashing library (e.g. `deepdiff.DeepHash`)

- Good, because it handles many container types out of the box.
- Bad, because we need fine control over per-field inclusion and over
  by-reference vs. by-value handling of functions/types, which is awkward to
  express through an external hasher; our own deconstructors keep the control local
  and composable. (`deepdiff` remains available in `eve.utils` for diffing.)

## References

- `src/gt4py/next/utils.py` — `Deconstruction`, `EmptyDeconstruction`,
  `OrderInsensitiveDeconstruction`, `reduce_object`, `make_deconstructor`,
  `deconstruct`, `fingerprint_fallback`, `fingerprint_deconstructor`,
  `fingerprint_collapser`, `stable_fingerprinter`,
  `skipping_fields_node_fingerprinter`, `gt4py_metadata`.
- `src/gt4py/next/ffront/stages.py` — `semantic_fingerprinter`.
- `src/gt4py/next/otf/workflow.py` — `CachedStep.cache_key`.
- `src/gt4py/next/config.py` — `BUILD_CACHE_VERSION_ID`.
- ADR [0011 - On The Fly Compilation](0011-On_The_Fly_Compilation.md) and
  [0017 - Toolchain Configuration](0017-Toolchain-Configuration.md) for the OTF
  toolchain and build cache this mechanism feeds.
