---
tags: []
---

# Connectivities as Types

- **Status**: proposed
- **Authors**: Enrique González Paredes (@egparedes)
- **Created**: 2026-09-21
- **Updated**: 2026-09-21

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
  dimension can have at most one owner.
- A local dimension with no table, such as the coefficient axis of a fixed-size
  stencil, is declared on its own: `class LsqCoeff(LocalDimensionIndex, size=3)`.
  Its `owner` is `None`. A declaration can also *adopt* such a module-level local
  dimension (`Local = LsqCoeff`), which then keeps its own tag.
- A connectivity can *share* another one's local dimension, `Local = C2E.Local`.
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
  values whether or not an entry uses it. The check is explicit for as long as
  offset providers are keyed by tag strings, since nothing then connects a
  provider entry to a declaration; it becomes automatic with class-keyed
  providers.

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

### How the base reaches `Local`

The base class declares `Local: ClassVar[type[LocalDimensionIndex]]` as an
annotation only. A real nested class on the base would be an incompatible
override for pyright in every declaration. The annotation makes `conn.Local` a
value of type `type[LocalDimensionIndex]` for library code taking any
connectivity; it does not make `conn.Local` usable as an *annotation* when
`conn` is generic, which no checker allows. Code that needs to name a local
dimension generically uses a `TypeVar` bound to `LocalDimensionIndex`.

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
  Shifts find the table by that tag; reductions and sparse arguments still find
  the owner's table by the local dimension's tag, for its neighbor structure, so
  the owner has to be bound too.
  `V2E.Local` inside DSL code types as that local dimension, and
  `FieldOffset.Local` names the same thing on a legacy offset, so the spelling
  works for both. The other frontend touch points treat the class like the
  `FieldOffset` it derives: grid-type deduction (`transform_utils`, `past_to_itir`)
  counts it as unstructured, and embedded `premap` accepts it. `V2E[i]` subscripts the
  metaclass, which forwards type-parameter subscription (`NeighborConnectivity[V, E]`) to `__class_getitem__`, since a metaclass `__getitem__` shadows it.

## Consequences

- An unstructured connectivity is spelled once. The provider key, the offset tag
  and the local dimension are all derived from the declaration.
- A table bound to a connectivity can be checked against its declaration.
- `V2E.Local` in DSL code is resolved from the offset type, because the type of
  `V2E` is not the class.
- A declaration is fingerprinted by its name *and* its declared dimensions and
  counts, so redefining it under the same name (e.g. re-running a notebook
  cell) does not reuse artifacts compiled for the old declaration.
- `FieldOffset` remains during migration; a `FieldOffset` and a
  `NeighborConnectivity` sharing a local dimension are interchangeable.

## Alternatives considered

- **The local dimension generated by the metaclass**, e.g. `V2E.Local` created
  from `V2E`'s name. Type checkers cannot see a generated class, so it could not
  be used in `Field[Dims[Vertex, V2E.Local], ...]`.
- **Neighbor counts as type parameters.** Python has no integer type parameters,
  and a `Literal[6]` argument would add a type parameter nothing statically uses.
- **`NeighborConnectivity` as a `Connectivity` subclass.** Mixes the
  declaration with the data protocol; see above.
