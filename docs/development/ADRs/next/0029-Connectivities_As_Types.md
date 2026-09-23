---
tags: []
---

# Connectivities as Types

- **Status**: proposed
- **Authors**: Enrique González Paredes (@egparedes)
- **Created**: 2026-09-21
- **Updated**: 2026-09-22

A neighbor connectivity is declared as a **class**, and its local dimension as a
class **nested** in it:

```python
class V2E(gtx.NeighborConnectivity[Vertex, Edge], max_neighbors=6, min_neighbors=5):
    class Local(gtx.LocalDimensionIndex): ...


@gtx.field_operator
def f(a: Field[Dims[Edge], float]) -> Field[Dims[Vertex], float]:
    return neighbor_sum(a(V2E), axis=V2E.Local) + a(V2E[0])
```

The declaration is written in DSL code, owns its local dimension, and states the
constraints a neighbor table bound to it has to satisfy. It holds no data. It builds on [ADR 0028](0028-Dimensions_As_Nominal_Types.md): the
connectivity, like a dimension, is identified by its type, and `V2E.Local` is an
ordinary dimension class with the tag `<module>.V2E.Local`.

## Context

An unstructured connectivity used to be spelled by four independently authored
names that had to agree, none of them checked against the others: the
`FieldOffset` tag, the Python variable it was bound to, the local dimension's
name and the offset-provider key. The `V2EDim = Dimension("V2E")` convention made
all four equal, which hid which one each execution path actually used; the
regression tests in `test_offset_dimensions_names.py` break the convention one
name at a time. Nothing tied a local dimension to the table it indexes, so the
backends recovered that link by string equality, and the table's shape, codomain
and skip values were never checked against the `FieldOffset` declaration.

## Decision

### The declaration

- `NeighborConnectivity[Origin, Codomain]` is a PEP 695 generic whose subclasses
  are declarations: for each `Origin` element, a list of `Codomain` neighbors. Its
  metaclass, `ConnectivityMeta`, forbids instantiation.
- The local dimension is the nested class `Local`, a subclass of
  `LocalDimensionIndex`. Declaring it is required, and `NeighborConnectivity`
  sets `Local.owner` to the connectivity when the class is created. A local
  dimension can have at most one owner; a declaration redefined under the same
  name (a re-run notebook cell) takes ownership over again, and for a local
  dimension adopted rather than nested, the first declaration wins.
- A local dimension with no table, such as the coefficient axis of a fixed-size
  stencil, is declared on its own: `class LsqCoeff(LocalDimensionIndex, size=3)`.
  Its `owner` is `None`. A declaration can also *adopt* such a module-level local
  dimension, written `Local: TypeAlias = LsqCoeff`, which then keeps its own tag.
- A connectivity can *share* another one's local dimension,
  `Local: TypeAlias = C2E.Local`.
  This is the flattened sparse pattern, e.g. cell-to-cell-edge (`C2CE: Cell -> CellEdge`) indexing the same neighbor axis as `C2E`, so that its results
  combine with `C2E`-shaped sparse fields. The owner stays `C2E`, and the
  neighbor counts and skip-value structure are the owner's.
- `max_neighbors` and `min_neighbors` are optional class keywords, not type
  parameters: Python has no integer type parameters, and nothing static needs
  the count. A declared count is a constraint on the bound table; an undeclared
  one is taken from the table. `min_neighbors < max_neighbors` means that the
  table must use skip values.
- `common.check_neighbor_table(V2E, table)` checks a table, or just its type
  (which is all an ahead-of-time compilation has), against the declaration:
  the domain is `(Origin, V2E.Local)`, the codomain is `Codomain`, the dtype is
  integral, and the neighbor counts and skip values agree. Skip values are
  checked on the table's type: a table with a `skip_value` counts as having skip
  values whether or not an entry uses it. Programs run the check on the tables
  they are given, see below.

### `NeighborConnectivity` is not a `Connectivity`

`common.Connectivity` is a *data* protocol (`ndarray`, `domain`, `asnumpy`); a
declaration holds no data. The neighbor table stays a `Connectivity`
implementation, and the declaration is only the type the table is checked
against. `Field.premap` and `Field.__call__` accept either, as they already
accepted a `FieldOffset`, which is not a `Connectivity` either.

### `LocalDimensionIndex` subclasses `DimensionIndex`

A separate root would force every `type[DimensionIndex]` annotation in the tree
(`ts.FieldType.dims`, `Domain`, `ConnectivityType.domain`, ...) to widen, and
would then accept local dimensions wherever a primary one is meant anyway. The
tree already distinguishes local dimensions by a runtime `kind` check, so it
keeps doing so; generic constructors whose parameter must be a primary dimension
(`NeighborConnectivity[Origin, Codomain]`, `Staggered[D]`) check it at runtime.

### `Local` is not annotated anywhere

Neither `NeighborConnectivity` nor `ConnectivityMeta` annotates `Local`, and
that is load-bearing: an annotation makes a declaration's `Local` a *variable*
for the checkers, so `Field[Dims[Vertex, V2E.Local], float]` is rejected by
pyright ("Variable not allowed in type expression") for a nested `Local`, and by
mypy ("not valid as a type") for an adopted or shared one. A real nested `Local`
on the base is not an option either: pyright reports an incompatible override in
every declaration. With no annotation, all three spellings are types for both
checkers, which `typing_tests/pyright_probes.py` pins for pyright and
`typing_tests/test_next.yaml` for mypy.

The cost is that `conn.Local` is not an attribute the checkers know for a
*generic* `conn`. Library code reads it through `common.local_dimension_of(conn)`
instead, and code that has to name a local dimension generically uses a
`TypeVar` bound to `LocalDimensionIndex`. Writing an adopted or shared local as
`Local: TypeAlias = ...` (rather than a plain assignment) is what keeps mypy
treating it as a type.

### Frontend integration

A declaration is typed like the `FieldOffset` it replaces: `V2E.__gt_type__()`
is the `ts.OffsetType` of the derived offset `(Codomain -> (Origin, Local))`,
whose tag is the connectivity's `offset_tag`:

- **the local dimension's tag**, `V2E.Local.tag`, for the connectivity that
  declares it. This is the single string that shifts, neighbor reductions and
  sparse arguments already use to find the table in the offset provider, so
  existing backends need no change.
- **its own tag**, `C2CE.tag`, for a connectivity that shares another one's local
  dimension, since the local dimension's tag already names the owner's table.
  Shifts find the table by that tag. Reductions and sparse arguments know only
  the local dimension, and take its neighbor count and skip values from a table
  over it (`common.connectivity_key_over`): the owner's if bound, else the
  sharer with the smallest tag. Connectivities sharing a local dimension must
  therefore have the same neighbor *structure* — the same count, and a skip value
  at the same positions — which is what sharing a neighbor axis means;
  `check_offset_provider` enforces it for the tables it is given.

`V2E.Local` inside DSL code types as that local dimension. The other frontend
touch points treat a declaration as the offset it replaces: grid-type deduction
(`transform_utils`, `past_to_itir`) counts it as unstructured, and embedded
`premap` accepts it. `V2E[i]` subscripts the
metaclass, which forwards type-parameter subscription (`NeighborConnectivity[V, E]`) to `__class_getitem__`, since a metaclass `__getitem__` shadows it.

### Offset providers are keyed by the declaration

Users bind tables to declarations:

```python
program(..., offset_provider={V2E: v2e_table, C2E: c2e_table})
```

Every entry point of a program normalizes such a provider to the form the IR
uses: each declaration is replaced by its `offset_tag`. Everything below the
entry points — lowering, the backends, compiled-program caching — therefore keeps
seeing a provider keyed by strings, which is also what hand-written IR uses.

The frontend entry points (`Program.__call__`, `FieldOperator.__call__`,
`compile`, `CompilationOptions.connectivities`) are *strict*: a string key must
be a tag, i.e. a qualified name, and a bare name such as `"V2E"` is the removed
`FieldOffset` spelling, rejected with a message pointing here. The IR-level hooks
(`embedded.context.update`, the iterator `fendef`, DaCe's `get_sdfg_conn_args`)
accept any string, because a hand-written program names its offsets itself.

Tables are checked against their declarations (`check_offset_provider`) at every
entry point, but the result is remembered per set of bound tables, so repeated
calls cost one hash. Reading the tables — comparing the skip-value positions of
two connectivities that share a local dimension — is done only where a program is
compiled, not on the call path. A tag that
names no declared connectivity, as in hand-written IR, is not checked.

### `FieldOffset` is removed

`FieldOffset` and its export are gone. An unstructured connectivity is a
`NeighborConnectivity`; a Cartesian shift is `Dim + i`, which the DSL already
had; and `as_offset` takes the dimension to shift along, `as_offset(KDim, k_offsets)`, instead of a Cartesian `FieldOffset`. `scripts/python/migrate_connectivities.py`
rewrites declarations and Cartesian offset uses, and reports the provider keys
and other sites it cannot rewrite from the source alone.

## Consequences

- An unstructured connectivity is spelled once. The provider key, the offset tag
  and the local dimension are all derived from the declaration.
- A table bound to a connectivity can be checked against its declaration.
- `V2E.Local` in DSL code is resolved from the offset type, because the type of
  `V2E` is not the class.
- A declaration is fingerprinted by its name *and* its declared dimensions and
  counts, so redefining it under the same name (e.g. re-running a notebook
  cell) does not reuse artifacts compiled for the old declaration.
- `FieldOffset` is removed, and offset providers are keyed by declarations: a
  breaking change for every unstructured program, eased by the migration script.

## Alternatives considered

- **The local dimension generated by the metaclass**, e.g. `V2E.Local` created
  from `V2E`'s name. Type checkers cannot see a generated class, so it could not
  be used in `Field[Dims[Vertex, V2E.Local], ...]`.
- **Neighbor counts as type parameters.** Python has no integer type parameters,
  and a `Literal[6]` argument would add a type parameter nothing statically uses.
- **`NeighborConnectivity` as a `Connectivity` subclass.** Mixes the
  declaration with the data protocol; see above.
