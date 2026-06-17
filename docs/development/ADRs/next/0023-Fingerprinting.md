---
tags: []
---

# [Fingerprinting]

- **Status**: valid
- **Authors**: Enrique González Paredes (@egparedes)
- **Created**: 2026-06-09
- **Updated**: 2026-06-15

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
- **Selective content**: some fields must be excluded from *some* fingerprints
  (inferred types that are filled in later and non-semantic caches always; source
  locations only from the persistent build-cache key, see below), and some values
  must be identified by reference rather than by value (functions, types, modules).
- **Robustness**: real inputs are large — lowered IR trees can nest far deeper
  than the Python recursion limit allows for recursive traversals.
- **Cache-version safety**: a way to globally invalidate cached builds when the
  toolchain changes in a way the fingerprint cannot capture.

## Decision

We standardize on a **structural fingerprinter**: a content hash of the object
graph computed bottom-up, Merkle-tree style. The implementation (in
`gt4py.next.fingerprinting`) is layered as a **catamorphism** — a generalized
fold over trees — keeping three concerns explicitly separate:

1. **Deconstruction** (*what is the one-level structure of an object?*): a
   per-type registry of *deconstructors*, each peeling off exactly one
   level into an `EmptyDeconstruction` or a `Deconstruction`.
2. **Traversal scheme** (*in which order are objects visited?*): `catabolize`,
   a generic iterative post-order fold with result memoization and cycle
   support, reusable with any result type.
3. **Reduction logic** (*how do results combine?*): the *aggregator* (the
   algebra of the catamorphism), a single function aggregating a
   deconstruction — whose pieces have already been aggregated into results —
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
module-level `deconstruct` is the default deconstructor, whose `fallback`
layers the `gt4py_metadata(fingerprint=False)` field opt-out on the dataclass /
datamodel / `__reduce_ex__` rules.
A fingerprinter is `catabolize` partially applied to a deconstructor and the
digest aggregator; `make_fingerprinter(deconstructor=…, aggregator=…)` builds one
(`strict_fingerprinter` is the default deconstructor + aggregator). The default
deconstruction rules are:

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
  silently colliding. (In-memory keys relax this to a structural fallback — see
  *Strict vs. lenient: the durability axis* below.)
- **Dataclasses and datamodels** — nodes of class + field values, honoring
  the field metadata opt-out (below).
- **Everything else** — deconstructed via the standard `__reduce_ex__` protocol
  (with a pinned protocol version), which covers e.g. NumPy arrays by content
  without any special-casing.

### Layer 2: the traversal scheme (`catabolize`)

`catabolize(obj, *, deconstructor, aggregator, allow_cycles, memoize)`
reduces an object bottom-up over its one-level deconstructions. The fold is
**iterative** (explicit two-phase work stack), so deeply nested inputs —
lowered IR trees routinely exceed the recursion limit budget of any recursive
scheme — cannot raise `RecursionError`. Results are memoized by object
identity, which makes shared subgraphs cheap, and self-referential structures
are aggregated (with `allow_cycles=True`) as back references by relative depth
(results under a cycle target are never memoized, as they are
context-dependent). Before aggregating an `OrderInsensitiveDeconstruction`,
the already-aggregated piece results are put into canonical sorted order, so
container iteration order never leaks into the result (this requires the
results to be orderable).
The driver is carrier-agnostic — nothing in it knows about hashing — so it can
serve other bottom-up reductions over arbitrary objects.

### Layer 3: the digest aggregator

The single fingerprint aggregator hashes a deconstruction uniformly as
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
    input_fingerprinter: Callable[..., HashT] = dataclasses.field(
        metadata=utils.gt4py_metadata(fingerprint=False)
    )
    step_fingerprinter: fingerprinting.Fingerprinter = dataclasses.field(
        default=fingerprinting.lenient_fingerprinter,
        metadata=utils.gt4py_metadata(fingerprint=False),
    )
    cache: ... = dataclasses.field(
        default_factory=dict, metadata=utils.gt4py_metadata(fingerprint=False)
    )
```

Here the *runtime* fields (`input_fingerprinter`, `step_fingerprinter`, `cache`) do not pollute the
fingerprint — only the semantically meaningful `step` does. This is a
declarative, per-class opt-out read directly from the field definitions; it
does **not** alter how the class pickles, so fingerprinting and real
serialization stay independent concerns.

### Customizing per type: composing deconstructors

Subsystems describe only the *deltas* they need on top of the shared default
rules:

- `strict_fingerprinter` — the by-reference variant (built-in rules over the
  dataclass/datamodel field fallback) that rejects non-importable objects; used
  for the workflow-step half of the persistent OTF build caches (see the
  durability split below).
- `skipping_fields_node_deconstructor("location", "type", ...)` — a cached
  factory returning a `Deconstructor` for `eve.Node`s that recursively skips the
  named child fields, ready to compose into a fingerprinter via
  `make_fingerprinter`. The iterator/GTIR
  `semantic_fingerprinter` skips `location` and `type`: it keys the **persistent**
  (cross-process, on-disk) C++/SDFG build cache, which must stay stable when only
  source locations shift (an unrelated edit above a stencil, a moved checkout) and
  across the inference passes that fill in `type` after node creation. It composes
  the skip on top of `lenient_fingerprint_deconstructor`, so a program graph is
  hashed structurally rather than rejected.
- The FOAST/PAST `semantic_fingerprinter`, by contrast, skips **nothing** — it keys
  the **in-memory** frontend-stage caches, whose cached product (the lowered IR) has
  `SourceLocation`s baked in by `PreserveLocationVisitor`. If two textually identical
  operators defined at different locations shared a stage-cache entry, the second
  would be served the first's lowering with the wrong locations, mislabeling its
  errors, warnings and (dace) debug info. For the same reason the ffront stages
  compose those FOAST/PAST node deconstructors with a `types.FunctionType`
  deconstructor that deconstructs a DSL definition function into its full
  **`SourceDefinition` (source, filename, line/column offsets) + closure variables** —
  location-*sensitive*, and recompiling when a referenced helper changes — rather
  than identifying it by reference:

```python
semantic_fingerprinter = fingerprinting.make_fingerprinter(
    deconstructor=fingerprinting.make_deconstructor(
        {types.FunctionType: _deconstruct_definition_function},
        fallback=fingerprinting.lenient_fingerprint_deconstructor,
    ),
)
```

This split is deliberate and complementary: a location-only edit changes the
in-memory ffront key (so the operator is re-lowered in-process, cheaply, with
correct locations) but leaves the persistent ITIR key unchanged (so the
expensive C++/SDFG artifact is still a cache hit and is not rebuilt).

### Strict vs. lenient: the tolerance axis

By-reference identification (above) is what makes a key reproducible in a
*different* process, where only an import path uniquely identifies an object — a
lambda, a local closure or a dynamically-created class has no stable cross-process
identity. Keying every fragility one importable name at a time (the icon4py
`FrozenNamespace` closure constants, a `Mock` backend, a custom-backend workflow
step have each tripped it) does not scale, so `lenient_fingerprinter` is the
**default** everywhere and the strict, reject-on-non-importable behavior is opt-in
(see the durability split below).

`lenient_fingerprinter` therefore mirrors `strict_fingerprinter` but, instead of
rejecting a non-importable object, falls back to a **structural** identity:

- a **function** is hashed by its code object (bytecode + constants, recursing
  into nested code objects), its defaults, and its captured closure cells —
  *not* its `__globals__`, which the bytecode already references by name. This
  is collision-free and invalidates correctly when a lambda's body or a captured
  value changes, but the code object is only stable **per CPython version**: a
  structural function hash can shift across interpreter upgrades.
- a dynamically-created **type/module** is identified by its (unverified)
  qualified name.

For any graph *without* non-importable objects the two fingerprinters agree
exactly (the lenient overrides try by-reference first), so switching a cache
between them never spuriously invalidates it.

`CachedStep` makes the choice **explicit** through two independently chosen
fingerprinters, `step_fingerprinter` and `input_fingerprinter`. On the bare
constructor `step_fingerprinter` defaults to `lenient_fingerprinter`, while
`input_fingerprinter` is a required argument (every cache must say how to
identify its inputs):

- `step_fingerprinter` keys the immutable workflow-state half. The persistent
  call sites (`CachedStep.persistent`, used by the gtfn/dace `cached_translation`
  traits) pair it with `strict_fingerprinter`, so a non-importable **step**
  (lambda, local closure, dynamically-created backend object) is rejected with a
  hard `TypeError` instead of being baked into a non-reproducible on-disk key.
- `input_fingerprinter` keys the per-call input half. The gtfn/dace persistent
  caches use `compilable_program_fingerprinter` (the iterator/GTIR
  `semantic_fingerprinter`), which is **lenient**: the program graph being
  compiled is hashed structurally rather than rejected. In practice the ITIR of
  a compilable program holds no non-importable callables, so lenient and strict
  agree on it; tolerating the rare exception (a `Mock`-backed value, a
  dynamically-created connectivity type) is preferred over a hard failure.

The consequence of a lenient input half is that a persistent key is reproducible
across processes and machines, but only **per CPython version**: should a
program graph ever contain a structurally-hashed function, an interpreter upgrade
shifts its key (a spurious rebuild, never a stale-binary reuse, since a
structural hash still changes when the code changes).

An earlier design inferred the fingerprinter from `isinstance(self.cache, dict)`.
Making it an explicit parameter keeps the two halves of the key — input identity
(`input_fingerprinter`) and workflow-state identity (`step_fingerprinter`) —
under direct caller control, and decouples the durability decision from the
concrete cache type.

### Integration with the build cache

`CachedStep.cache_key(inp)` ties the pieces together:

```python
@functools.cached_property
def _step_fingerprint(self):
    # The step and the version salt are immutable, so this is computed once per instance.
    return self.step_fingerprinter((config.BUILD_CACHE_VERSION_ID, self))


def cache_key(self, inp):
    return self.step_fingerprinter((self._step_fingerprint, self.input_fingerprinter(inp)))
```

- `self` is the step, content by fields, so changing the step configuration
  invalidates the cache. Whether functions referenced by the step must be
  identified by qualified name (so, behind a persistent `FileCache`, module-level
  rather than lambdas or local closures) or may be hashed structurally is decided
  by the caller-chosen `step_fingerprinter` (see *the durability axis* above).
  Since the step is immutable, its fingerprint is computed once per instance and
  reused across lookups.
- `self.input_fingerprinter(inp)` contributes the input's identity.
- `config.BUILD_CACHE_VERSION_ID` (defaulting to the gt4py version, overridable
  via the `GT4PY_BUILD_CACHE_VERSION_ID` env var) is a global salt that lets us
  force incompatibility with previously cached builds when the toolchain changes
  in a way the fingerprint alone would not capture.

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
  drops content the cached output actually depends on causes false cache hits. What
  counts as "non-semantic" is therefore per-cache, not absolute — `location` is
  dropped from the persistent ITIR key (the C++ artifact does not depend on it) but
  kept in the frontend-stage keys (the lowered IR embeds it). Skips are chosen
  conservatively against the specific cached product for this reason.
- The **workflow step** behind a persistent `FileCache` must be importable
  (module-level): a lambda or local closure step is a hard error, by design
  (`CachedStep.persistent` pairs `step_fingerprinter=strict_fingerprinter`), so a
  non-reproducible on-disk step identity — the worst failure mode — cannot occur.
  Everything else is hashed with `lenient_fingerprinter`, which falls back to a
  structural identity (code object + defaults + closure cells for functions) only
  stable per CPython version. The price is two fingerprinters to maintain and the
  fact that the **input** half of a persistent key (e.g. the program graph behind
  the gtfn/dace caches) is reproducible only per CPython version, not across
  interpreter upgrades — acceptable because a structural hash still changes with
  the code (a stale binary is never reused; at worst a key shifts and triggers a
  rebuild).

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
aggregators) deliberately mirrors how a fingerprinter would be written over
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

- `src/gt4py/next/fingerprinting.py` — `Deconstruction`, `EmptyDeconstruction`,
  `OrderInsensitiveDeconstruction`, `catabolize`, `make_deconstructor`,
  `make_fingerprinter`, `deconstruct`, `fingerprint_aggregator`,
  `strict_fingerprinter`, `lenient_fingerprinter`,
  `skipping_fields_node_deconstructor`. Imported directly by
  `gt4py.next` consumers (not re-exported through `gt4py.next.utils`).
- `src/gt4py/next/utils.py` — `gt4py_metadata` and its namespace key
  `GT4PY_CLASS_METADATA_NS`, the per-field fingerprinting opt-out helper.
- `src/gt4py/next/ffront/stages.py` — `semantic_fingerprinter`.
- `src/gt4py/next/otf/workflow.py` — `CachedStep.cache_key`.
- `src/gt4py/next/config.py` — `BUILD_CACHE_VERSION_ID`.
- ADR [0011 - On The Fly Compilation](0011-On_The_Fly_Compilation.md) and
  [0017 - Toolchain Configuration](0017-Toolchain-Configuration.md) for the OTF
  toolchain and build cache this mechanism feeds.
