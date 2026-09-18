# Connectivities as types — implementation plan

**Status**: **APPROVED** at revision 6 (adversarial review rounds 1–4; round 4 verdict APPROVED)
**Target**: an 8-PR stack on `main`, *alternative to* GridTools/gt4py#2844
**Proposal**: `egparedes/connectivities-as-types` in GridTools/gt4py_knowledge (PR #32)
**Baseline tree**: `b3c53fa7e` (v1.2.2)

## 0. Scope and relation to #2844

The proposal and #2844 agree on *what a dimension is* (a class, its indices its
instances) and disagree on *what identity a dimension has*. #2844 chose
`(tag, kind)` value equality with an interning registry; the proposal chose
nominal type identity with the tag being the qualified Python name. That single
disagreement propagates into five mechanisms, so the two cannot both land.

This stack **re-cuts** #2844: it keeps the machinery independent of identity,
drops the machinery that exists only to support value identity, and then builds
the connectivity layer on top. **#2844 is closed, not merged** — that is what
makes this an alternative. Consequences: the new dimension ADR is **0028** (the
ADR directory on `main` ends at **0027**; 0028 exists only inside unmerged
#2844), and there is nothing to supersede.

### Taken from #2844, unchanged in substance

| Piece | Where in #2844 |
| --- | --- |
| `DimensionMeta` metaclass; `I + 1`, `I > 5`, `repr` living on it | `common.py` |
| `DimensionIndex` base: `__slots__ = ("value",)`, `kind` class keyword, `.dim` property | `common.py` |
| `type Dimension = type[DimensionIndex]` as a PEP 695 alias (so `Dimension("I")` raises rather than silently evaluating to `str`) | `common.py` |
| Metaclass `.value` property raising `AttributeError` that points at `.tag` | `common.py` |
| Deletion of `common.NamedIndex` (`.dim` / `.value` move onto the index instance) | `common.py` + ~40 call sites |
| Deletion of the dimension half of `mypy_plugin.py` (`_DimA`..`_AnyDim`); only the mixed-precision hooks remain | `type_system/mypy_plugin.py` |
| The mechanical migration of every declaration, incl. docs, workshop notebooks and `examples/` (which `test_examples` executes) | 131 files: 52 `src/`, 66 `tests/`, 13 docs |
| `xtyping.resolve_annotation` usage at `fbuiltins._type_conversion_helper` (already on `main` via #2841) | — |

### Dropped from #2844

| Piece | Why |
| --- | --- |
| `_DIMENSION_REGISTRY` interning | identity is the type; nothing to intern |
| `copyreg.pickle(DimensionMeta, _reduce_dimension)` — the **blanket** registration on all dimensions | verified: a module-level dimension class pickles by reference with no help (`pickle.loads(pickle.dumps(KDim)) is KDim`). **But a narrow `copyreg` on `StaggeredMeta` is still required** — see §1.5 |
| The `DimensionMeta`-vs-`DimensionMeta` branch of `__eq__` / `__ne__` | becomes `is`. **The `IntegralScalar` overloads (`I == 5` → `Domain`) are kept**, and therefore so is an explicit `__hash__` — see §1.0 |
| `DimensionIndex.__eq__` comparing `type(self) == type(other)` | becomes `type(self) is type(other)` |
| `common.dimension(tag, kind)` factory | replaced by `common.resolve(tag)`, which imports |
| `fingerprinting.py` `DimensionMeta` deconstructor keyed on `(tag, kind)` | under type identity a dimension *is* fingerprinted by qualified name, so the generic `type` deconstructor is correct — **for the lenient variant only**. The STRICT variant rejects `Staggered[KDim]`, which is not importable under its qualified name. Both in-tree fingerprinters are lenient (`ffront/stages.py:62`, `iterator/ir.py:26`), and `eve_utils.content_hash` (`compiled_program.py:420`) is pickle-based and so needs §1.5's `copyreg`. Record the STRICT caveat in the ADR |
| ADR 0028 as drafted in #2844 | never lands; this stack writes its own 0028 |

### Changed relative to #2844

| Piece | #2844 | This stack |
| --- | --- | --- |
| `tag` default | `cls.__name__`, settable in the class body | `f"{cls.__module__}.{cls.__qualname__}"`, a metaclass property; a class-body `tag = ...` is a `TypeError` |
| Rebuilding a dimension from a tag | `dimension(tag, kind)` (registry) | `resolve(tag)` (`import_module` + `qualname` walk), memoized |
| Declaration site requirement | none | module level, or unpicklable; `<locals>` heuristic in `__init_subclass__` |
| Backend name mangling | `tag` used directly | `codegen_name(tag)` + inverse, at ~19 enumerated sites in two name spaces (§1.3(b), (c)) |
| Staggered dimensions | `_Staggered` prefix through the interning factory | `Staggered[D]`, a real parametrized type — **required in PR 2**, not optional (§1.5) |

### Superseded

- **#2845** (`test[next]: adopt class-style dimension declarations`) is subsumed
  by PR 2: because `dimension()` is not user-facing, the minimal
  `I = gtx.dimension("I")` form does not exist and every declaration takes class
  form immediately. #2845's pyright coverage is folded in.
- The `FieldOffset`-as-frontend-identifier part of **ADR 0019**.
- **ADR 0026**'s `_Staggered` name prefix (PR 2).

## 1. Design questions closed before implementation

Everything in this section was verified by running it, not by reading. Probe
files are named; they become committed test material in the PR that needs them.

### 1.0 Metaclass mechanics that are easy to get wrong

**`__hash__` must be declared explicitly.** Python sets `__hash__ = None` on any
class body that defines `__eq__` without `__hash__` — metaclasses included. Since
the `I == 5` → `Domain` overload keeps `__eq__` on `DimensionMeta`, dropping
#2844's `__hash__` makes every dimension class *unhashable*:

```
>>> class M(type):
...     def __eq__(cls, o): return True
>>> M.__hash__ is None
True
>>> class C(metaclass=M): pass
>>> hash(C)
TypeError: unhashable type: 'M'
```

That would break `domain({I: 2})` (`common.py:672-690`),
`Counter[common.Dimension]` (`embedded/nd_array_field.py:314`),
`dict[Dimension, SymbolicRange]` (`iterator/ir_utils/domain_utils.py:136,152`),
`seen: dict[Dimension, Dimension]` (`common.py:1351`), and eve's validator
memoization on annotation objects (`eve/type_validation.py:599`) — so
`ts.DimensionType` would fail at *import*. **Fix: `__hash__ = type.__hash__`
explicitly on `DimensionMeta`,** and likewise on `ConnectivityMeta` if it ever
defines `__eq__`.

**A metaclass `__getitem__` shadows `__class_getitem__`.** `ConnectivityMeta`
needs `__getitem__` for `V2E[1]` (the single-neighbor shift handle that
`FieldOffset.__getitem__` provides today), but metaclass lookup takes precedence
over `Generic.__class_getitem__`, so a naive implementation makes
`NeighborConnectivity[V, E]` in a bases list fail with
`TypeError: tuple expected at most 1 argument, got 3`.

**Fix, verified clean under `mypy --strict` and pyright 1.1.414 on Python 3.12**
(`/tmp/probe_meta_getitem3.py`): dispatch on the argument type, delegating
non-`int` subscription back to `cls.__class_getitem__`:

```python
class ConnectivityMeta(type):
    __hash__ = type.__hash__
    @overload
    def __getitem__(cls, item: int) -> Connectivity: ...
    @overload
    def __getitem__(cls, item: Any) -> Any: ...
    def __getitem__(cls, item: Any) -> Any:
        # `numbers.Integral`, not `int`: `V2E[np.int32(1)]` must not fall through
        # to the type-parameter branch (it raises `TypeError: V2E is not a
        # generic class` there). `bool` is excluded so `V2E[True]` is an error
        # rather than silently neighbor 1.
        if isinstance(item, numbers.Integral) and not isinstance(item, bool):
            return _bound_single_neighbor(cls, int(item))
        # type-parameter subscription, e.g. `NeighborConnectivity[V, E]`
        return cast(Any, cls).__class_getitem__(item)
```

`cast(Any, cls)`, not `super()` — `__class_getitem__` is on the class, not on the
metaclass MRO; `super().__class_getitem__` raises `AttributeError`. With the cast
both checkers report zero errors and all uses work at runtime
(`NC[V, E]`, `class V2E(NC[V, E])`, `V2E[1]`, and `V2E.Local` as an annotation).
pyright accepts `NC[V, E]` in a **bases list**; in a *value* position it types it
`Any`, which is why the overloads above matter — without them `V2E[1]` is also
`Any` and the shift handle is untyped.

### 1.1 `NeighborConnectivity` is **not** a `Connectivity` (proposal Open Q6)

`common.Connectivity` is `Field[DimsT, IntegralScalar]` — a **data** protocol
(`common.py:990`; `ndarray`, `asnumpy`, `domain` are all on it). A declaration
class holds no data.

**Resolution.** Two distinct things, distinct hierarchies:

- `NeighborConnectivity` — a **declaration**. Not a `Connectivity`. It produces a
  `NeighborConnectivityType` via `__gt_type__()`, is the provider key, and is the
  handle written in DSL code (`a(V2E)`).
- `NeighborTable` / `NdArrayConnectivityField` — the **data**, unchanged, still
  `Connectivity` implementations.

This is the shape `FieldOffset` already has: it is *not* a `Connectivity` either,
and `premap` special-cases it at `nd_array_field.py:317-320`. So `a(V2E)`
continues to work by widening the same union — `Field.premap` and
`Field.__call__` are typed `Connectivity | fbuiltins.FieldOffset`
(`common.py:785, 791-794`) and become `Connectivity | type[NeighborConnectivity]`
in PR 4. `V2E` has **no instances**: `ConnectivityMeta.__call__` raises
`TypeError("… is a connectivity declaration and cannot be instantiated; bind a
table through the offset provider")`.

Consequence: the proposal's sketch line
`class NeighborConnectivity(Connectivity[MultiDimensionIndex[Origin, Local], Codomain], ...)`
is **wrong and dropped**. `MultiDimensionIndex` remains the *domain index type* of
the `NeighborTable` (PR 8). **The knowledge-repo note needs this correction.**

### 1.2 How `Local` reaches the base (proposal Open Q2)

`requires-python = '>=3.12'`, so a PEP 696 default type parameter (3.13) is not
available. Resolution: **metaclass discovery**, base carrying a `ClassVar`
annotation, subclass declaring the nested class explicitly:

```python
class NeighborConnectivity[Origin: DimensionIndex, Codomain: DimensionIndex](
    metaclass=ConnectivityMeta
):
    Local: ClassVar[type[LocalDimensionIndex]]      # annotation only, never assigned

class V2E(NeighborConnectivity[V, E], max_neighbors=6):
    class Local(LocalDimensionIndex): ...           # explicit, required
```

Verified under `mypy --strict --python-version 3.12` and `pyright --pythonversion 3.12`
(`/tmp/probe_local.py`):

| Variant | base declares | mypy | pyright |
| --- | --- | --- | --- |
| 1 | `Local: ClassVar[type[LocalDimensionIndex]]` | clean | clean |
| 2 | nothing | clean | clean |
| 3 | a real nested `class Local(LocalDimensionIndex)` | clean | **`reportIncompatibleVariableOverride`** |

In all three the intended negative case (`Field[V, A.Local]` vs
`Field[V, B.Local]`) is correctly an error. Variant 3 is rejected.

**Stated precisely — what variant 1 does and does not buy.** It does *not* make
`conn.Local` usable as a **type annotation** when `conn` is a generic
`type[NeighborConnectivity]`: both checkers reject that (mypy `name-defined`,
pyright `reportInvalidTypeForm`), and `T.Local` on a `TypeVar` is rejected too.
What variant 1 buys over variant 2 is only **value-level** access —
`reveal_type(conn.Local)` is `type[LocalDimensionIndex]` instead of an attribute
error — which is what library code in `common`, the backends and
`type_synthesizer` actually needs. Variant 1 is chosen for that, not for generic
annotations. Generic library code that must *name* a local dimension in a
signature uses `type[LocalDimensionIndex]`.

This extends the proposal's probe P2: a **generated** `Local` is unusable as an
annotation, but a base `ClassVar` *annotation* plus an explicitly declared nested
class is fine.

### 1.3 The IR keeps string tags; `resolve()` and `codegen_name()` are both required

`AxisLiteral.value: str` stays (making it carry the class is a separate IR
change, deferred past this stack). It now holds the **qualified** tag, and that
has two consequences the first draft of this plan underestimated.

**(a) `resolve(tag)` at every rebuild site**, memoized — `inference.py:464` calls
it once per `AxisLiteral` on the type-inference hot path:

| Site | Purpose |
| --- | --- |
| `iterator/ir_utils/domain_utils.py` | `AxisLiteral` → `Dimension` |
| `iterator/ir_utils/misc.py` | `AxisLiteral` → `Dimension` |
| `iterator/type_system/inference.py:464` | `AxisLiteral` → `ts.DimensionType` |
| `codegens/gtfn/itir_to_gtfn_ir.py` (×2) | staggered-name sniffing → replaced in PR 2 by `Staggered[D]` |
| `dace/lowering/gtir_to_sdfg_lambda.py:1155` | synthesizes the local dim from the *offset* tag: `Dimension(offset, LOCAL)`. **Must be fixed in PR 2, not deferred** — see below |
| `dace/sdfg_args.py` | axis name → `Dimension` |
| `runners/roundtrip.py` | emits `gtx.Dimension(...)` as *source text* → becomes an import |
| ~~`common.flip_staggered` (×2)~~ | **not** a `resolve()` site: `Staggered[D]` replaces it with an interning subscript, §1.5 |

`resolve` on a nested qualname was verified to work and round-trip
(`resolve("mymod.V2E.Local") is mymod.V2E.Local`), which matters because PR 4 keys
the provider on `V2E.Local.tag`. **One hazard to settle in PR 2**: a purely dotted
tag does not record *where* the module path ends and the qualname begins, so
`resolve` must try the longest importable prefix and walk the rest — O(depth)
import attempts, and in principle ambiguous if a module path and a class-attribute
chain collide. `pickle` avoids this by storing module and qualname *separately*.
Options: keep the pure dotted form (what the proposal asks for, ambiguity
tolerated and memoized away) or use an explicit separator such as
`"module:qualname"`. **Recommendation: keep the dotted form** — it is what makes
the tag "also a valid tag string for the IR", the collision requires a module and
an attribute chain to have the same spelling, and `resolve` can prefer the
*longest* importable prefix so a real module always wins. Record the residual in
the ADR.

**(b) `codegen_name(tag)` — dots are illegal in every generated identifier.**
`eve`'s `SymbolName`/`SymbolRef` are constrained by
`_SYMBOL_NAME_RE = ^[a-zA-Z_]\w*$` (`eve/concepts.py:23,26,32`), so a qualified
tag reaching `Sym(id=...)` is a *validation error*, not a cosmetic problem. The
first draft mentioned mangling only in the abstract and put the roundtrip change
in a later PR; both were wrong. All of these are **PR 2**:

| Site | What breaks without mangling |
| --- | --- |
| `codegens/gtfn/itir_to_gtfn_ir.py:170-195` | `TagDefinition(name=Sym(id=dim.value))` → `SymbolName` validation error |
| `codegens/gtfn/gtfn_module.py:97, 130-136` | `generated::{dim.value}_t`, plus `name.lower()` |
| `otf/binding/nanobind.py:197, 211` | C++ identifiers |
| `dace/lowering/gtir_to_sdfg_utils.py` `get_map_variable` | `i_{dim.value}_gtx_{kind}` → invalid DaCe symbol |
| `dace/sdfg_args.py:80` `_field_symbol` | invalid DaCe symbol |
| `dace/lowering/gtir_python_codegen.py:137-138` | `visit_AxisLiteral` returns the raw value |
| `runners/roundtrip.py:64, 177` | `AxisLiteral = as_fmt("{value}")`, and `{o.value} = gtx.Dimension(...)` emits `a.b.I = ...` → `SyntaxError` |

**(d) The mangling scheme, corrected.** Earlier drafts said "injective (escape
existing `__` before replacing `.`)", i.e. `_ -> __` then `. -> _`. **That is not
injective**: `.` becomes a single `_`, so `".."` and `"_"` both map to `"__"`.
Exhaustively tested over the alphabet `{a, ., _}` up to length 6
(`/tmp/probe_mangle.py`): **686 collisions in 1092 inputs.** Since a generated
identifier may only contain `[A-Za-z0-9_]`, `_` is the only available separator
and a *prefix escape* is required:

```python
def codegen_name(tag: Tag) -> str:
    return tag.replace("_", "_u").replace(".", "_d")

def from_codegen_name(name: str) -> Tag:
    return re.sub(r"_([ud])", lambda m: "_" if m.group(1) == "u" else ".", name)
```

Every `_` in the output is the first character of a two-character escape, so
decoding is unambiguous. Verified exhaustively over `{a, ., _, u, d}` up to
length 6 — **19530 inputs, 0 collisions, 0 round-trip failures**, every output a
valid identifier, including the adversarial `"_u"`, `"_d"` and `"a_ud.b"`
(`/tmp/probe_mangle2.py`). Cost: names grow (`mod.V2E.Local` →
`mod_dV2E_dLocal`), which is what gtfn's existing `TagDefinition.alias` mechanism
is for.

**An inverse is needed too**, wherever generated names are parsed *back* into
dimensions: `dace/sdfg_args.py:25, 60-72` matches `gt_conn_(\S+)` and feeds the
result to `has_offset`. `codegen_name` must therefore be injective *and* have a
`from_codegen_name` partner (escape `__` → `____` before `.` → `__`).

**A site that cannot be deferred: `gtir_to_sdfg_lambda.py:1155`.** It builds
`gtx_common.Dimension(offset, DimensionKind.LOCAL)` — a local dimension
synthesized from the **offset** tag, which in PR 2 is still a bare provider key
(`"V2E"`) that `resolve()` cannot import. Every DaCe unstructured shift passes
through it, so PR 2 is red on DaCe unless it is fixed there. The fix is local and
available: `conn_type` is already in scope (`:1134-1152`) and `:1135` already
asserts `conn_type.domain[1].kind == LOCAL`, so the line becomes
`offset_type = conn_type.domain[1]` (equivalently `conn_type.neighbor_dim`).
It is *necessary but not sufficient* for PR 1's `shift × tag≠localdim` DaCe cell:
that cell fails earlier, at `gtir_to_sdfg.py:842`
(`neighbor_table_types[dim.value]`, i.e. A4 on the connectivity *argument's* local
dim), before `:1155` is reached — and after the `:1155` fix, `:1371`/`:1455` would
reference `gt_conn_<localdim>` while `:1104`/`:722` declare `gt_conn_<offset>`. So
**the DaCe shift cell stays in the skip matrix until PR 4**, where the
single-string choice makes both agree. (An earlier draft said PR 2; that would
leave PR 2 red on that cell.)

**(c) The *offset* key is a second dotted name space, and it is mangled in PR 4,
not PR 2.** §1.3(b) covers *dimension* names only. When PR 4 makes the provider
key `cls.tag`, the **offset** string that flows through the IR
(`OffsetLiteral.value`, the provider key, `o` in the gtfn/DaCe connectivity
plumbing) becomes dotted too, and a different set of sites turns *it* into an
identifier. These are all **PR 4**:

| Site | What breaks |
| --- | --- |
| `codegens/gtfn/itir_to_gtfn_ir.py:184` | `TagDefinition(name=Sym(id=offset_name))` → `SymbolName` regex |
| `codegens/gtfn/itir_to_gtfn_ir.py:490` | `SymRef(id=o)` for each connectivity → `SymbolRef` regex |
| `codegens/gtfn/codegen.py:147-148` | `visit_OffsetLiteral` emits `node.value` raw into C++ |
| `codegens/gtfn/gtfn_module.py:118, 132, 136` | `GENERATED_CONNECTIVITY_PARAM_PREFIX + name.lower()`, `generated::{name}_t` |
| `dace/sdfg_args.py:56` | `connectivity_identifier(name)` → `gt_conn_a.b.V2E`, an invalid SDFG array name |
| `dace/sdfg_args.py:60`, `dace/workflow/bindings.py:200, 286` | `is_connectivity_identifier` / `_parse_gt_connectivities` — the **inverse** direction, so `from_codegen_name` has *several* live consumers, not one |
| `dace/workflow/translation.py:61`, `dace/sdfg_callable.py:103`, `dace/program.py:156` | `connectivity_identifier(offset)` again, on the argument-binding path |
| `dace/lowering/gtir_to_sdfg_lambda.py:1104, 1371, 1455, 1727`, `gtir_to_sdfg.py:722` | the same identifier, consumed in the lowering |
| `runners/roundtrip.py:63` | `OffsetLiteral = as_fmt("{value}")` — emits the offset tag *raw as Python source*, into the program **body**; mangling `:176` alone still leaves `NameError: name 'tests' is not defined` |
| `dace/sdfg_args.py:83-84` | `_field_symbol`: `assert m[1] in offset_provider_type` — a *second* `from_codegen_name` consumer besides `:70` |
| `dace/lowering/gtir_to_sdfg_lambda.py:1892` | `visit_OffsetLiteral` → `SymbolExpr(node.value, INDEX_DTYPE)`, i.e. a dotted string used as a DaCe symbolic expression |
| `runners/roundtrip.py:152, 176` | collects offset-literal strings, then `f'{o} = offset("{o}")'` → `a.b.V2E = offset(...)` → `SyntaxError` |

So `codegen_name` / `from_codegen_name` are introduced in PR 2 for dimensions and
**applied again in PR 4 for offsets**, at ~16 further sites. Two earlier claims
were wrong: that `from_codegen_name`'s only live consumer is in PR 2, and that the
DaCe surface is confined to `sdfg_args.py` and the lowering — the
argument-binding path (`workflow/translation.py`, `workflow/bindings.py`,
`sdfg_callable.py`, `program.py`) carries it too, in both directions.

Because of (b) and (c), the review shortcut "diff PR 2 against #2844, the delta is
only identity" is **false**: #2844 needed none of this. Reviewers should expect a real
mangling layer on top of the identity delta.

**`AxisLiteral.kind` becomes redundant** (the class carries it) — the `TODO` at
`iterator/ir.py:93`. Kept in PR 2, removed in PR 7, to keep PR 2's IR-expectation
churn to the `value` strings only.

### 1.4 `LocalDimensionIndex` subclasses `DimensionIndex`; `DimensionBaseIndex` is dropped

The proposal lists `DimensionBaseIndex` as a separate root with `DimensionIndex`
and `LocalDimensionIndex` as siblings. That does not survive contact with the
tree: `Dimension` is `type[DimensionIndex]`, eve validates a `type[X]`
annotation by `issubclass` (verified: a subclass passes, the base and an
unrelated class are both rejected), so sibling local dimensions would force
widening to `type[DimensionBaseIndex]` at `ts.DimensionType.dim`,
`ts.FieldType.dims`, `ConnectivityType.domain`, `Domain.__init__` and the `DimT`
/ `DimT_co` bounds — and would then accept local dimensions everywhere a primary
one is meant, which is the same looseness with extra ceremony.

**Resolution**: `class LocalDimensionIndex(DimensionIndex, kind=DimensionKind.LOCAL)`.
`DimensionBaseIndex` is not introduced at all — one concept fewer, which is the
proposal's own stated goal. Where primary-only is required the check is
`dim.kind is not DimensionKind.LOCAL`, exactly as today. This also removes
#2844's deferral note ("a `DimensionBase` root, deferred until the requirements
of non-user-declarable dimensions are known") as a thing that needs resolving.

**Verified**: all 38 sites in `src/` that discriminate a local dimension do so by
a **runtime `kind` check**, not by a static type distinction
(`transform_utils.py:65`, `type_deduction.py:460, 774`,
`custom_layout_allocators.py:171`, `past_to_itir.py:409`, `common.py:1168, 1336`,
`nd_array_field.py:972, 976`, `gtfn_module.py:91`, `embedded.py:922`,
`gtir_to_sdfg_types.py:76`, …). The tree already treats local dimensions as
`Dimension`s everywhere — `ConnectivityType.domain: tuple[Dimension, ...]`
includes the local one — so subclassing loses nothing it currently relies on, and
`Dims` (`tuple[Unpack[ShapeTs]]`, `common.py:57`) puts no bound on its members
either.

**What subclassing does cost**, and the mitigation: every `DimensionIndex`
*bound* now statically admits a local dimension —
`NeighborConnectivity[Origin: DimensionIndex, Codomain: DimensionIndex]`,
`Staggered[D: DimensionIndex]` and `MultiDimensionIndex[D: DimensionIndex, *Ls]`
would all accept `V2E.Local` as their primary parameter. Each therefore gets a
runtime `kind is not DimensionKind.LOCAL` check in `__init_subclass__` /
`__class_getitem__`, and `LocalDimensionIndex.__init_subclass__` rejects an
explicit `kind=` other than `LOCAL`. This is the same runtime-check discipline
the tree already uses; the static gap is the price of the concept removed.

**Deviation from the proposal; needs feeding back to the note.**

### 1.5 `Staggered[D]` is required in PR 2, not PR 7

`flip_staggered` builds `Dimension(f"_Staggered{name}")` from a string
(`common.py:1452-1457`) and `is_staggered` tests `dim.value.startswith(prefix)`
(`:1447-1449`). #2844 routes both through the interning factory. With the
registry gone there is **no importable `_Staggered<tag>` type**, and a
dynamically created class would get the tag
`gt4py.next.common._Staggered<name>`, so `is_staggered` is false and
`as_non_staggered` cannot recover the base dimension's module. Live dependents:
`test_staggered.py` (233 lines), `cases_utils.py:161`
(`KHalfDim = flip_staggered(KDim)`), gtfn `_add_staggered_aliases`
(`itir_to_gtfn_ir.py:203-215`), DaCe `get_map_variable`
(`gtir_to_sdfg_utils.py:52`), `type_synthesizer`, `test_common.py`,
`test_domain_utils.py`.

So PR 2 is **not green** without `Staggered[D]`. It is Cartesian-only and does
not depend on the connectivity layer, so it moves into PR 2.

**The obvious mechanism does not work.** A PEP 695 generic
`class Staggered[D: DimensionIndex](DimensionIndex)` makes `Staggered[KDim]` a
`typing._GenericAlias`, **not a class** (verified, `/tmp/probe_staggered.py`):

```
type(Staggered[KDim])            -> <class 'typing._GenericAlias'>
isinstance(Staggered[KDim], type)-> False
issubclass(Staggered[KDim], ...) -> TypeError: issubclass() arg 1 must be a class
Staggered[KDim].tag              -> '__main__.Staggered'      # KDim is gone
```

So it fails eve's `type[DimensionIndex]` validation and its tag cannot name the
base dimension — it is not a `Dimension` at all.

**The mechanism that does work** (verified, `/tmp/probe_staggered3.py`: runs
correctly and is **0 errors under both `mypy --strict` and pyright 1.1.414** on
3.12) is a metaclass `__getitem__` that *builds and interns a real class*, paired
with a `TYPE_CHECKING` declaration so checkers still see an ordinary generic:

```python
class StaggeredMeta(DimensionMeta):
    def __getitem__(cls, base: Dimension) -> Dimension:
        if base not in _staggered_cache:
            _staggered_cache[base] = StaggeredMeta(
                f"Staggered[{base.__name__}]",
                (cls,),          # NOT (cls, base) -- see below
                {"_tag": f"{cls.__module__}.{cls.__qualname__}[{base.tag}]",
                 "kind": base.kind, "base": base, "__slots__": ()},
            )
        return _staggered_cache[base]

if TYPE_CHECKING:
    class Staggered[D: DimensionIndex](DimensionIndex):
        base: ClassVar[Dimension]
else:
    class Staggered(DimensionIndex, metaclass=StaggeredMeta):
        __slots__ = ()
        base: ClassVar[Dimension]
```

Verified properties of `Staggered[KDim]`: it *is* a class;
`tag == "gt4py.next.common.Staggered[<KDim's qualified tag>]"`; `kind` is
inherited from the base; `issubclass(_, DimensionIndex)` and
`issubclass(_, Staggered)` hold; it is instantiable as an index; and
`Staggered[KDim] is Staggered[KDim]`, so identity is stable. `Staggered[KDim]` in
an annotation and inside `Field[Dims[Staggered[KDim]], float]` are both accepted
by both checkers.

- **Bases are `(cls,)`, not `(cls, base)`.** Inheriting from the base dimension
  would make `issubclass(Staggered[KDim], KDim)` true, i.e. `KHalfDim` would be
  accepted everywhere `KDim` is required. It is a *different* dimension; only
  `kind` is inherited, copied explicitly into the namespace.
- `is_staggered(dim)` becomes **`"base" in dim.__dict__`**, not
  `issubclass(dim, Staggered)`, and `as_non_staggered(dim)` becomes `dim.base`.
  Two runtime facts force this: `issubclass(Staggered, Staggered)` is true for
  the bare base, which has no `base`; and `Staggered[KDim]` is *subclassable*
  (`class KHalf2(Staggered[KDim])` yields a second, un-interned staggered-K type
  with tag `<module>.KHalf2`). `Staggered.__init_subclass__` therefore rejects
  any subclass the metaclass did not create, so the interned form is the only
  one. Still structural — no string sniffing.
- **The guards were verified, including the escape routes**
  (`/tmp/probe_staggered_guards.py`). All four are blocked:
  `class KHalf2(Staggered[KDim])`, `class X(Staggered)`, a direct
  `StaggeredMeta("Y", (Staggered,), {})`, and the double subscript
  `Staggered[KDim][KDim]`. The `copyreg` fallback round-trips the bare
  `Staggered` by reference, the parametrized class with identity preserved, and
  instances. Implementation note: gate `__init_subclass__` on a **namespace
  marker** the metaclass sets (`"_tag" in cls.__dict__`), not on a module-level
  "currently building" flag — the flag works but is not thread-safe, and
  compilation runs in worker processes and threads. Three further honest limits:
  the guard defends against **accidental** subclassing only — a deliberate
  `StaggeredMeta("Forged", (Staggered,), {...marker})` or `types.new_class` can
  still forge a same-`tag`, non-identical type (as it can for any class);
  `Staggered[Staggered[KDim]]` must be rejected explicitly by testing
  `"base" in base.__dict__` in `__getitem__`, or it nests and pickles happily; and
  a hand-built `copyreg` payload such as `(_make_staggered, (int,))` should raise
  a `TypeError` naming the offending base rather than an `AttributeError`.
- `resolve` gains the `<qualname>[<tag>]` grammar: it parses the brackets and
  evaluates `Staggered[resolve(inner)]`, which hits the same intern cache, so a
  staggered dimension round-trips through the IR to the *same* class object.
- **A narrow `copyreg` is required after all.** `Staggered[KDim]`'s
  `__qualname__` is `Staggered[KDim]`, which `pickle.save_global` cannot look up:
  `PicklingError: Can't pickle <class 'Staggered[KDim]'>: attribute lookup
  Staggered[KDim] on … failed` (verified, `/tmp/probe_staggered_pickle.py`). A
  `copyreg.pickle(StaggeredMeta, lambda cls: (_make_staggered, (cls.base,)))`
  fixes it *and preserves identity*, because the reconstructor goes back through
  the intern cache. **But it must guard the bare base**: `type(Staggered) is
  StaggeredMeta` too, so a reducer that unconditionally reads `cls.base` fails on
  `Staggered` itself with `AttributeError: type object 'Staggered' has no
  attribute 'base'` (verified — an earlier draft of this section claimed the
  registration "captures only parametrized dimensions", which is false). The
  reducer therefore falls back to by-reference pickling when
  `"base" not in cls.__dict__`. It never captures a plain dimension
  (`type(KDim) is DimensionMeta`). This is materially narrower than
  #2844's blanket registration on `DimensionMeta` — a parametrized type needs a
  reconstructor for the same reason `typing` aliases do — but §0's "`copyreg`
  dropped" row is only true of the blanket form, and the ADR must say so.
- **Two honest costs.** (i) `_staggered_cache` is a cache, and the proposal's
  headline is that the *name-keyed* registry goes away. The difference is real but must be
  stated: it is keyed by a *dimension class*, is internal, and is memoization of
  a type constructor (as `typing`'s own subscription cache is), not interning of
  user-authored name strings — nothing resolves a user string through it.
  (ii) the `TYPE_CHECKING` split means the static and runtime definitions can
  drift; a unit test must assert the runtime facts the static form does not
  express (real class, `issubclass` against `Staggered` but *not* against the
  base, tag shape, interning).
- Supersedes ADR 0026, recorded in the PR-2 ADR.

## 2. The PR stack

Branches follow the repo's stacked convention, `connectivities-as-types-<n>-<slug>`,
each based on its predecessor, all targeting `main`. PR titles are Conventional
Commits (squash-merge lands the title).

---

### PR 1 — `fix[next]: lower unstructured shifts with the offset's own tag`

**Independent of the rest of the stack; lands first, on its own merit.**

`foast_to_gtir._visit_shift` emits the **Python variable name** as the IR shift
tag (`foast_to_gtir.py:305` `offset_name.id`, `:331` `str(offset_name)`), because
`ts.OffsetType` does not carry the tag. Embedded execution keys on
`FieldOffset.value`. So the same program needs a *different* provider key
depending on the backend — confirmed by running it on v1.2.2:

```
MyOff = FieldOffset("TAGNAME", ...)
embedded:  {"TAGNAME": conn} OK ; {"MyOff": conn} -> KeyError 'TAGNAME'
roundtrip: {"MyOff": conn}   OK ; {"TAGNAME": conn} -> KeyError 'MyOff'
```

**Change**

- `ts.OffsetType` gains **`tag: Optional[Tag] = None`** — *not* a required field.
  `type_deduction.py:709` builds `ts.OffsetType(source=conn.codomain,
  target=(conn.domain_dim,))` from `IDim + 1`, a `CartesianConnectivity` that has
  no tag at all; making `tag` required breaks it.
- `FieldOffset.__gt_type__` fills it (`fbuiltins.py:485`).
- `type_deduction.py:464`, which rebuilds an `OffsetType` when `Off[1]` drops the
  local dimension, must **propagate** the tag.
- `foast_to_gtir._visit_shift`: the `Subscript` branch and the bare `Name` branch
  use `arg.type.tag`, asserting non-`None` (both are unstructured paths, where a
  tag always exists).

**Tests.** `tests/next_tests/regression_tests/ffront_tests/test_offset_dimensions_names.py`
today covers exactly `a(Off[1])` on `GTFN_CPU`. Extend to
{shift, `neighbor_sum`} × {embedded, roundtrip, gtfn, dace} × {tag≠varname,
tag≠local-dim-name}.

**The matrix is not uniform, and a blanket `xfail` will not do.** `xfail_strict = true`
(`pyproject.toml:323`), and measured behaviour on v1.2.2 is:

| case | embedded | roundtrip | gtfn | dace |
| --- | --- | --- | --- | --- |
| shift, tag≠varname | pass | pass | pass | pass |
| shift, tag≠localdim | pass | pass | pass | **fail** `KeyError` (`gtir_to_sdfg_lambda.py:1155` synthesizes the local dim from the tag) |
| `neighbor_sum`, tag≠localdim | **fail** | **pass** | **fail** | **fail** |

So the first draft's acceptance criterion ("shift cells pass on all four
backends") is unreachable before the backend work, and a strict blanket `xfail`
would XPASS on roundtrip. **Fix**: add a per-backend skip matrix entry in
`tests/next_tests/definitions.py` (a new `USES_*` marker) covering exactly the
failing cells, roundtrip excluded. **They are removed in two steps**: the
`shift × tag≠localdim` DaCe cell and the three `neighbor_sum × tag≠localdim` cells
all in **PR 4**, where the single-string choice makes A3/A4 vacuous — *not* in
PR 5, and *not* the DaCe cell in PR 2 (the `:1155` fix there is necessary but not
sufficient; see §1.3(a)). The gtfn `neighbor_sum`
failure is now confirmed **by running it**; the proposal had it only "by
reading".

**No CHANGELOG entry.** Verified against the history: `CHANGELOG.md` is touched
*only* by release PRs (`git log -- CHANGELOG.md` is release commits exclusively,
and nothing between `b3c53fa7e` and `upstream/main` touches it). The behaviour
change — which key a compiled backend requires when tag ≠ variable name — belongs
in the PR description, and reaches the changelog when the release PR is cut. Two
earlier drafts of this plan said otherwise, including for PR 6's breaking change.

**ICON4Py is unaffected by PR 1**: all 16 `FieldOffset` variable names equal
their tags (`model/common/src/icon4py/model/common/dimension.py:33-48`).

**Acceptance**: `nox -s test_next` green; every cell in the matrix either passes
or is covered by the documented skip matrix.

---

### PR 2 — `feat[next]: a concrete Dimension is a class, identified by its qualified name`

The #2844 core with the identity divergences of §0, **plus** the mangling layer
of §1.3(b) and `Staggered[D]` of §1.5 — both of which #2844 did not need and
without which this PR cannot be green. Large and largely mechanical.

**`src/gt4py/next/common.py`**

```python
class DimensionMeta(type):
    kind: DimensionKind
    __hash__ = type.__hash__            # §1.0 — mandatory, not optional
    @property
    def tag(cls) -> Tag: ...            # f"{cls.__module__}.{cls.__qualname__}"
    # operators as in #2844; __eq__/__ne__ keep only the IntegralScalar overload
    # (I == 5 -> Domain); the dim-vs-dim branch is `is`.

class DimensionIndex(metaclass=DimensionMeta):
    __slots__ = ("value",)
    kind: ClassVar[DimensionKind] = DimensionKind.HORIZONTAL
    def __init_subclass__(cls, /, kind=None, **kw): ...

# Staggered: an interning metaclass + TYPE_CHECKING split, NOT a PEP 695
# generic -- see §1.5, where the generic form is shown to be unworkable.

def resolve(tag: Tag) -> Dimension: ...          # memoized; <qualname>[<tag>] grammar
def codegen_name(tag: Tag) -> str: ...           # "_" -> "_u", "." -> "_d"  (§1.3(d))
def from_codegen_name(name: str) -> Tag: ...     # the inverse, `_([ud])` -> `_` / `.`

type Dimension = type[DimensionIndex]
```

- `tag` is a metaclass **property**, so it cannot drift from the type. This makes
  a class-body `tag = "C2E"` a **silent no-op** (verified: `C2EDim.tag` stays
  `"__main__.C2EDim"` even with `tag = "C2E"` in the body) — and that pattern is
  exactly what ICON4Py and #2845 use to rename. `__init_subclass__` therefore
  **raises** on `"tag" in cls.__dict__`, naming the class and pointing at the
  rename path.
- `__init_subclass__` also rejects `"<locals>" in cls.__qualname__`. Neither
  necessary nor sufficient (`type("Dyn", ...)` in a function passes; a `del`'d
  class passes) — the authoritative check stays pickle's own `save_global`.
- `resolve` raises a `ValueError` naming the tag and the failing import, per
  CODING_GUIDELINES.

**Removals**: `NamedIndex`; `_DimA`..`_AnyDim` and the dimension half of
`mypy_plugin.py`; `_DIMENSION_REGISTRY`; `copyreg`; the `fingerprinting.py`
deconstructor; `_STAGGERED_PREFIX` and its string sniffing.

**Migration**. Every `Dimension("X")` becomes `class X(DimensionIndex): ...` at
module level. Verified counts: 333 `Dimension("` declarations in `tests/`, of
which **133 are function-local across 15 files** and must move to module level;
a dimension *named* `"I"` is declared 46 times across **17** files (the first
draft said 45 files — that was the proposal's *`IDim` file* count, a different
number). Docs, workshop notebooks and `examples/` are included because
`test_examples` executes them; notebook *code* cells only, stored outputs
untouched (they hold recorded tracebacks that must keep naming the symbols that
produced them).

**IR expectation churn**: 36 `AxisLiteral` and 37 `OffsetLiteral` occurrences in
`tests/`, most already computed from `dim.value`. `test_pretty_roundtrip.py` and
the gtfn/DaCe snapshot tests hold the hardcoded names.

**Do the sweep with a codemod script, not agent fan-out.** A previous attempt at
agent fan-out on a large mechanical rewrite in this repo died mid-file on the
rate limit and left the tree inconsistent; a script did all 57 files uniformly.

**ADR 0028** (the directory ends at 0027): nominal identity; the module-level
declaration requirement; `Staggered[D]` superseding ADR 0026; that cache
fingerprints now shift when a declaration moves module (a consequence for ADR
0023, not a reversal); that `resolve()` imports modules named in the IR, which is
the same trust level as `pickle` loading a class by reference.

**Documented limitation**: interactive `__main__` (REPL, notebooks, `python -c`)
cannot be resolved. `spawn` compile workers re-execute the main *script* as
`__mp_main__`, so file-based `__main__` resolves provided the script has the
`if __name__ == "__main__":` guard the pool already requires.

**ICON4Py migration script is a PR-2 deliverable, not PR 6.** All 15 local
dimensions and `KDim`/`EdgeDim`/`CellDim`/`VertexDim` have variable name ≠ tag
(`EdgeDim = Dimension("Edge")`), so PR 2 changes every generated symbol and every
cache key downstream.

**Acceptance**: `nox -s test_next` on **3.12, 3.13 and 3.14** (the `typing`
subscription cache behaves differently per interpreter and this change moves
exactly that behaviour), then `test_eve`, `test_storage`, `test_cartesian`,
`test_examples`; `uv run mypy src/`; `uv run pyright`; `uv run tach check`;
`uv run pre-commit run -a`. One at a time, pytest capped at `-n 4`.

---

### PR 3 — `feat[next]: NeighborConnectivity declarations and local dimensions that know their owner`

**Purely additive**: new concepts next to `FieldOffset`, nothing removed, no
behaviour change, no test churn.

```python
class LocalDimensionIndex(DimensionIndex, kind=DimensionKind.LOCAL):   # §1.4
    owner: ClassVar[type[NeighborConnectivity] | None] = None
    max_neighbors: ClassVar[int | None] = None
    min_neighbors: ClassVar[int | None] = None
    def __init_subclass__(cls, *, size: int | None = None, **kw): ...

class ConnectivityMeta(type):        # §1.0 for __hash__ and __getitem__
    @property
    def tag(cls) -> Tag: ...
    def __call__(cls, *a, **kw) -> NoReturn: ...

class NeighborConnectivity[Origin: DimensionIndex, Codomain: DimensionIndex](
    metaclass=ConnectivityMeta
):
    Local: ClassVar[type[LocalDimensionIndex]]
    def __init_subclass__(cls, *, max_neighbors=None, min_neighbors=None, **kw): ...
```

- `__init_subclass__` asserts `"Local" in cls.__dict__` and that it subclasses
  `LocalDimensionIndex`, then sets `Local.owner = cls` and copies the counts. A
  missing `Local` is a `TypeError` at class creation naming the class.
- **Owner-less** locals: `class LsqUnk(LocalDimensionIndex, size=3)` — `owner is
  None`, `min == max == size`, never in the provider. ICON4Py's `LsqUnkDim` and
  `RBFDimension` need this: they index no table but need sparse storage and
  layout.
- Counts are **optional class keywords**, not type parameters (Python has no
  integer type parameters and nothing static needs the count). Declared ⇒ a
  constraint the table must satisfy. Undeclared ⇒ completed at bind time, from
  the table in the JIT flow or from the `NeighborConnectivityType` already passed
  through `connectivities=` (`ffront/decorator.py:188-208`) in the AOT flow.
  Not static-only because `fvm_nabla_setup.py:99` sizes `V2E` from the atlas
  mesh, and ICON skip-value presence is configuration-dependent (`icon.py:130` —
  pentagons have skip values on the icosahedron, not the torus).
- **Bind-time validation**, one function replacing constraints A6–A8: shape
  `(n, max_neighbors)`, integral dtype, skip values present iff
  `min_neighbors < max_neighbors`, `domain[0] is Origin`, `codomain is Codomain`.

**No provider bridge.** The first draft proposed a dual-keyed
(`Tag | type[NeighborConnectivity]`) provider here. Dropped: the provider is
accessed **directly, not through `get_offset`, at 19 sites in 12 `src/` files**
(`itir_to_gtfn_ir.py:181`, `gtfn_module.py:106`, `sdfg_args.py:63-72`,
`compiled_program.py`, `pass_manager.py`, …) despite the note at
`common.py:1174`, so a bridge would be both invasive and — since nothing would
exercise class keys — untested. Class-keyed providers land in one place, PR 4.

**Typing tests**: `typing_probe.py` / `probe_local.py` / `probe_meta_getitem3.py`
become real coverage — `typing_tests/test_next.yaml` cases for
`Field[Dims[V, V2E.Local], float]`, a `TypeVar` bound to `LocalDimensionIndex`,
the negative cross-connectivity case, and the `NC[V, E]`-in-bases case of §1.0;
plus the pyright variants.

**Acceptance**: full suite green with no behaviour change; new unit tests for
declaration errors, owner wiring, owner-less locals and bind-time validation.

---

### PR 4 — `feat[next]: declare connectivities as classes; FieldOffset derived from them`

**The ordering fix.** Revision 2 put class-keyed providers here and the
declaration migration in PR 6. That cannot be green: `Local.owner` only exists if
the user declared a `NeighborConnectivity` class, and a `FieldOffset` written the
old way (`FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))`) has no class
to point at — so neither the backend work nor a class-keyed provider has anything
to resolve. The declaration migration must come **first**, and the provider key
must stay a string until the backends are through.

- The **unstructured** `FieldOffset` is **derived**, not authored:
  `FieldOffset.from_connectivity(V2E)` (or `V2E.__gt_offset__()`), which fills
  `source = Codomain`, `target = (Origin, V2E.Local)` and — critically —
  **`value = V2E.Local.tag`, the *local dimension's* tag, not `V2E.tag`.**
- **Why the local dimension's tag and not the connectivity's.** An earlier draft
  used `V2E.tag` and claimed PR 4 was green. It is not: A3 and A4 key the provider
  on the **local dimension's** name, at `nd_array_field.py:983`
  (`get_offset(provider, axis.value)`, whose in-tree comment is literally
  `# assumes offset and local dimension have same name`), `unroll_reduce.py:47`
  (`arg.type.offset_type.value`), `gtfn_module.py:95`, `gtir_to_sdfg.py:581, 842`,
  `iterator/embedded.py:954, 1519`, and
  `gtir_to_sdfg_lambda.py:1371, 1455` (`connectivity_identifier(offset_type.value)`).
  Today the `V2EDim = Dimension("V2E")` convention makes that string equal to the
  offset tag; PR 4 deletes the convention tree-wide, while the `owner` lookup that
  replaces it is PR 5. With `value = V2E.tag` every reduction and every sparse-field
  argument would break on embedded, gtfn and DaCe simultaneously — the round-1
  matrix row (`neighbor_sum`, tag≠localdim: embedded/gtfn/DaCe fail) would become
  the tree's universal state.
  Choosing `V2E.Local.tag` instead makes **all four** of A1, A3, A4 and A5 vacuous
  at once, because there is then exactly *one* string and the class produces it.
  It also makes the #1789 branch at `itir_to_gtfn_ir.py:181-190`
  (`if offset_name != connectivity_type.neighbor_dim.value`) dead already in PR 4.
  This is preferable to the alternatives — fusing PR 5 into PR 4, or a transient
  `owner`-based fallback inside `get_offset` — because it needs no scaffolding:
  the string is simply picked correctly, and PR 5 then removes the dependence on a
  string at all.
- **The Cartesian `FieldOffset` constructor stays in PR 4.** An earlier draft said
  "every declaration becomes a class", which is wrong: `Ioff`, `Koff` and
  `EdgeOffset` (`cases_utils.py:163-169`, e.g.
  `Ioff = gtx.FieldOffset("Ioff", source=IDim, target=(IDim,))`) and ICON4Py's
  `Koff`/`KHalfOff` (`dimension.py:47-48`) are single-target and have no
  `NeighborConnectivity` to derive from, and their only remaining consumer —
  `as_offset` — does not change until PR 6. Restricting the public constructor to
  the single-target form keeps PR 4 green; it disappears with `as_offset` in
  PR 6.
- **Providers stay keyed on `Tag`**, now `V2E.Local.tag`. Nothing about the key
  *mechanism* changes yet, so the 19 direct-access sites are untouched. A1, A3, A4
  and A5 are all dead at this point — there is one string, and the class produces
  it.
- `ts.OffsetType` → `ConnectivityType`, produced by the class
  (`type_specifications.py:74` TODO). `type_info.py:637, 858` gate `a(V2E)`
  deduction on `ts.OffsetType` and follow. `Field.premap` / `Field.__call__`
  unions widen to `Connectivity | type[NeighborConnectivity]` (`common.py:785,
  791-794`, `1313-1316`).
- **Test-tree migration lands here**: `toy_connectivity.py`, `cases_utils.py`,
  `fvm_nabla_setup.py` are the fixture modules everything imports; 37
  `FieldOffset` sites, 42 `DimensionKind.LOCAL` sites. String provider keys keep
  working because they are `cls.tag` — but the tags are now *qualified*, so the
  111 ITIR-level string-offset occurrences in 15 files (`im.shift("V2E")`,
  `neighbors("…")`, `OffsetLiteral(value="…")`, string-keyed providers) are
  rewritten to `V2E.Local.tag` here rather than in PR 6.
- **ICON4Py**: this is the release-visible declaration change. The migration
  script written in PR 2 is extended.

---

### PR 5 — `refactor[next]: backends resolve connectivities through the local dimension's owner`

Where A3, A4 and A5 dissolve and PR 1's skip-matrix entries are removed. Green
while providers are still string-keyed, because a backend goes
`local_dim.owner` → `owner.tag` → the existing lookup: the *identity* question is
answered by the owner pointer, and the key is still a string.

**The owner-less case must be handled, not assumed away.** At PR 5 `_CONST_DIM`
is still a plain `DimensionIndex(kind=LOCAL)` (it becomes `ConstList` only in
PR 7), and `LsqUnk`-style local axes have `owner is None` by design. Every
converted site reads `getattr(dim, "owner", None)` and falls through when it is
`None`. Two sites already guard by accident — DaCe compares against `_CONST_DIM`
first (`gtir_to_sdfg_lambda.py:1314`) and `unroll_reduce` filters
`offset_type is None` — but `gtfn_module.py:91-98` and `nd_array_field.py:981`
have **no** guard, and ICON4Py never exercises the case
(`test_icon.py:220`), so the gap would not show up downstream.

`unroll_reduce.py:47` (reads `arg.type.offset_type`, which is the local
`Dimension` — now a `LocalDimensionIndex` carrying `owner`, which is exactly the
back-pointer it lacked), `gtfn_module.py:95, 118, 132`, `itir_to_gtfn_ir.py`
(including the `#1789` `offset_name != neighbor_dim.value` branch at `:181-190`,
which becomes dead and goes), `gtir_to_sdfg.py:581`,
`gtir_to_sdfg_lambda.py:766-770, 1155` (the `Dimension(offset, LOCAL)` synthesis
goes), `nd_array_field.py:981-985` (and its
`# assumes offset and local dimension have same name` comment),
`iterator/embedded.py`, and `runners/roundtrip.py`.

---

### PR 6 — `feat[next]!: class-keyed offset providers; remove FieldOffset and the string offset API`

The only breaking PR, and now the only one that touches the provider key.

- `OffsetProvider*` become
  `Mapping[type[NeighborConnectivity], NeighborTable]`; `get_offset` keys on the
  class. **Note the `.owner` hop**: because PR 4 made the IR offset tag the *local
  dimension's* tag, `resolve(OffsetLiteral.value)` yields a `LocalDimensionIndex`,
  not the connectivity — so the class-key lookup is
  `resolve(tag).owner`. (The alternative is to switch the IR tag to `V2E.tag` in
  this PR; the `.owner` hop is cheaper and keeps the IR stable.) The **19 direct-access sites in 12 `src/` files**
  (`itir_to_gtfn_ir.py:181`, `gtfn_module.py:106`, `sdfg_args.py:63-72`,
  `compiled_program.py`, `pass_manager.py`, …) are converted here, and
  `common.py:1174`'s "all accesses should go through `get_offset`" either becomes
  true or the note goes. `hash_offset_provider_items_by_id` and the
  `fingerprinting` dict handling already tolerate class keys once §1.0's
  `__hash__` is in place.
- **Removals**: `FieldOffset` entirely (both forms); `runtime.Offset` as its base
  (`fbuiltins.py:467-470` TODO); `iterator/runtime.offset("...")` (12 sites in 6
  files, plus `tracing.py:161-162`); the `V2EDim`-next-to-`V2E` convention;
  `embedded/context.py` string plumbing; the `gt4py.next.__init__` exports at
  `:47, 140`.
- **`as_offset` changes in the same PR.** It is why the Cartesian `FieldOffset`
  form cannot go alone: `ffront/experimental.py:17` +
  `type_deduction.py:956-967` require one. New signature
  `as_offset(KDim, field)`. Used in 5 test modules, the `Ioff`/`Koff`/`EdgeOffset`
  fixtures at `cases_utils.py:163-169`, and **40 non-test call sites in
  ICON4Py**.
- `transform_utils.py:50-77` and `past_to_itir.py:77` deduce grid type from the
  provider and follow.
- **Accepted double churn**: the ~26 `offset_provider={...}` literals are rewritten
  twice — `{V2E.Local.tag: t}` in PR 4, `{V2E: t}` here. The alternative is fusing PR 4
  and PR 6, which loses the green boundary. The ITIR string sites do *not* churn
  twice: `im.shift(V2E.Local.tag)` written in PR 4 stays correct.

**ADR 0029**: the connectivities-as-types record — `FieldOffset` removed, the
class-keyed provider, superseding the `FieldOffset` part of ADR 0019.

**Breaking-change communication**: the PR title carries the Conventional Commits
`!` marker and the ADR records the removal; the changelog entry is written by the
release PR, not here (see PR 1). No deprecation window — an explicit decision:
ICON4Py's provider keys are bare names, so no import-based shim could have
resolved them.

**Acceptance**: full suite; `test_fvm_nabla` and `test_icon_like_scan` are the
integration canaries. A before/after run of one gtfn and one DaCe program
checking generated-code equivalence modulo names.

### PR 7 — `refactor[next]: ConstList replaces _CONST_DIM; AxisLiteral drops kind`

- `_CONST_DIM` (`iterator/embedded.py:220` and
  `dace/lowering/gtir_to_sdfg_lambda.py` — two separate declarations, each
  internally consistent; 14 references in total) becomes the owner-less
  `ConstList(LocalDimensionIndex, size=1)`, generalizing the magic name from size
  1 to size *n*.
- `AxisLiteral.kind` removed (`iterator/ir.py:93` TODO), now that every tag
  resolves to a class carrying its kind.

---

### PR 8 — `refactor[next]: MultiDimensionIndex and typed embedded positions`

- `MultiDimensionIndex[D: DimensionIndex, *Ls]` as the index type of a sparse
  position and the domain index of a `NeighborTable`. `*Ls` is unconstrained
  because `TypeVarTuple` cannot carry a bound; `__init_subclass__` checks at
  runtime what the checker cannot.
- `iterator/embedded.py` positions keyed by dimension types instead of name
  strings (`embedded.py:574-576`, `597-616`, `941-950`); `SparseTag` removed.
  Constraint A9 dissolves.
- Nothing else depends on this; it is last for that reason.

---

## 3. Constraint ledger

| # | Constraint | Retired by |
| --- | --- | --- |
| A1 | `FieldOffset.value` == provider key | PR 4 (one declaration produces both) |
| — | *all four string-equality constraints below become vacuous in PR 4*, because the class emits a single string (`V2E.Local.tag`); PR 5 removes the dependence on a string at all | PR 4 / PR 5 |
| A2 | Python variable name == provider key | **PR 1** |
| A3 | local dim name == provider key (reductions) | PR 4 vacuous; mechanism removed in PR 5 (`Local.owner`) |
| A4 | local dim name == provider key (sparse args) | PR 4 vacuous; mechanism removed in PR 5 (`Local.owner`) |
| A5 | `FieldOffset.value` == local dim name | PR 4 (one declaration) |
| A6 | `target[-1]` == connectivity `neighbor_dim` | PR 3 (bind-time check) |
| A7 | `FieldOffset.source` == `codomain` | PR 3 (bind-time check) |
| A8 | `target[0]` == `domain[0]` | PR 3 (bind-time check) |
| A9 | dim name is the iterator-position dict key | PR 8 |
| A10 | dim name round-trips through `AxisLiteral` | structural; PR 2 qualifies it, PR 7 drops `kind` |
| F5/F8/F9 | codegen name formats | PR 2 (`codegen_name` + inverse) |
| S6 | `as_offset` needs a Cartesian `FieldOffset` | PR 6 |
| — | string-keyed provider | PR 6 |

## 4. Risks

1. **PR 2 size, and it is not purely mechanical.** ~150 files of migration *plus*
    a name-mangling layer and `Staggered[D]`. It cannot be reviewed as "the
    #2844 diff plus identity". Mitigation: land the mangling layer and
    `Staggered[D]` as reviewable commits *within* the PR, ordered before the
    sweep, so the mechanical part is a separate commit.
2. **Function-local dummy dimensions.** 133 declarations in 15 test files must
    move to module level. Under nominal identity, two same-named locals that were
    silently the same dimension become distinct — each resulting failure is a
    real finding, not churn.
3. **`typing` subscription caching.** Under `(name, kind)` equality,
    `Field[Dims[I]] is Field[Dims[I2]]` aliases for two distinct same-named
    classes — a known residual of the #2844 design, and an argument *for* this
    stack. The cache behaves differently per interpreter: verify on 3.12, 3.13
    and 3.14 **via nox**, not `uv run pytest`.
4. **Fingerprint/cache invalidation.** Moving a declaration between modules now
    invalidates compiled artifacts. Intended; CHANGELOG + ADR line.
5. **Naming not yet converged** with `havogt/dependent-local-dimensions`
    (`Origin`/`Codomain` vs `source_dim`/`neighbor_dim`; `Local` vs `Dim`;
    `min_neighbors` vs `has_skip_values`). PR 3 fixes public names. Converge
    before PR 3 is *opened*.
6. **`V2E.Local` vs `Local[V2E]`.** The chain proposals' encodings subscript
    `Local`, and a `TypeVar` cannot be subscripted for a nested attribute. Their
    semantics are unaffected; their static encoding needs rewriting to `C.Local`
    plus a protocol for the generic hop-stack case. Knowledge-repo concern, not a
    gt4py blocker.
7. **CSCS GPU CI is flaky and opaque.** All jobs failing at the same second means
    infrastructure; `cscs-ci run default` as a PR comment reruns it. #2844's CI is
    green except that job.
8. **Two mangling passes, two PRs.** `codegen_name` is applied to dimension names
    in PR 2 and to offset names in PR 4, at ~19 sites total, several of which
    (`Sym`/`SymRef` construction, DaCe array names) fail *loudly* and several of
    which (C++ emission, `name.lower()`) fail only in the generated artifact.
    Both PRs need a test that a qualified tag survives a real gtfn and a real
    DaCe compile, not just lowering.
9. **`resolve()` on the inference hot path** (`inference.py:464`, once per
    `AxisLiteral`). Must be memoized from the start, and the memo must be keyed
    so a reloaded module does not return a stale class.

## 4b. Work the earlier drafts did not mention

- **Public exports.** `gt4py.next.__init__` must export `NeighborConnectivity`,
  `LocalDimensionIndex`, `Staggered` and `resolve` (PR 2 for the dimension half,
  PR 3 for the connectivity half), and drop `FieldOffset` / `offset` at `:47, 140`
  in PR 6.
- **`type_translation.from_value(V2E)`** works only because the
  `hasattr(value, "__gt_type__")` branch at `type_translation.py:328` is tested
  *before* the `DimensionMeta` branch. That ordering is load-bearing under this
  design and currently untested — PR 3 adds a unit test pinning it.
- **`pyright` is not yet a dependency.** `uv run pyright` appears throughout §5
  but pyright is absent from `pyproject.toml` on `main`; #2845 is what adds it.
  PR 2 must explicitly fold in #2845's `typing_exports` / pyright dependency-group
  change, or §5's pyright step is not runnable.
- **`test_examples` belongs to PR 4 too.** §5 lists it for PR 2 and PR 6; the
  docs and notebooks use `FieldOffset`, so PR 4's declaration migration touches
  them and must run it.

## 5. Verification

Per PR, in this order, **one at a time** on the shared machine, pytest capped at
`-n 4`:

```
uv run pre-commit run -a          # ruff, mypy, tach, license headers
uv run pyright                    # static checks the mypy plugin no longer fakes
uv run nox -s "test_next-3.12(...)"        # then 3.13, 3.14 for PR 2
uv run nox -s test_eve test_storage test_cartesian test_examples   # PR 2, PR 6
```

Test-first where behaviour changes, per AGENTS.md: PR 1's regression matrix, PR
3's declaration-error and bind-validation units, and PR 4's provider-key tests
are written before the implementation they cover.

## 6. Feedback owed to the knowledge-repo note

- `NeighborConnectivity` is not a `Connectivity`; the sketch's base line is wrong
  (§1.1) — this closes Open Q6.
- `DimensionBaseIndex` should be dropped; `LocalDimensionIndex` subclasses
  `DimensionIndex` (§1.4).
- Open Q2 is closed: metaclass discovery, base `ClassVar` *annotation*, explicit
  nested class — with the precise limit of what that buys statically (§1.2).
- `Staggered[D]` is not a late step; it is a precondition for removing the
  name-keyed registry (§1.5) — and it cannot be a PEP 695 generic, needs an
  identity-keyed intern cache, and needs a narrow `copyreg`. The note's claim
  that `copyreg` disappears entirely is therefore too strong.
- The note's "five name spaces" analysis should record that under qualified tags
  there are **two** dotted name spaces reaching codegen — dimension tags and
  offset tags — each needing its own mangling pass (§1.3(b) and (c)).
- The note's §Staging step 2 ("`NeighborConnectivity` … object-keyed provider;
  `FieldOffset` and string keys removed outright") bundles three changes that
  must be separated to stay green: the declaration migration has to precede the
  backend work (because `Local.owner` only exists once classes are declared), and
  the provider *key* has to stay a string until the backends resolve through the
  owner. See PR 4/5/6.
- The staging in the note's §Staging (steps 0–8) is superseded by §2 here; in
  particular step 4 ("backends, one file at a time") cannot follow step 2, since
  the provider key change and the backend lookups are separable but the mangling
  layer is needed at the *dimension* step.

## 7. Open, non-blocking

- Naming convergence (risk 5).
- Whether a same-`__name__` collision warning is useful or noise (note Q5).
- Whether interactive `__main__` should be detected with a fallback to in-process
  compilation and a warning, or merely documented (note Q3). Plan assumes
  documented.
- `AxisLiteral.dim` instead of `AxisLiteral.value` — a follow-up after PR 8.
