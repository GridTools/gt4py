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
constraints the neighbor table bound at call time has to satisfy. It holds no
data. It builds on [ADR 0028](0028-Dimensions_As_Nominal_Types.md): the
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
  Its `owner` is `None`.
- `max_neighbors` and `min_neighbors` are optional class keywords, not type
  parameters: Python has no integer type parameters, and nothing static needs
  the count. A declared count is a constraint on the bound table; an undeclared
  one is taken from the table. `min_neighbors < max_neighbors` means that the
  table must use skip values.
- `common.check_neighbor_table(V2E, table)` checks a table, or just its type
  (which is all an ahead-of-time compilation has), against the declaration:
  the domain is `(Origin, V2E.Local)`, the codomain is `Codomain`, the dtype is
  integral, and the neighbor counts and skip values agree.

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
whose tag is **the local dimension's tag**, `V2E.Local.tag`. This is the single
string that shifts, neighbor reductions and sparse arguments already use to find
the table in the offset provider, so existing backends need no change.
`V2E.Local` inside DSL code types as that local dimension. `V2E[i]` subscripts the
metaclass, which forwards type-parameter subscription (`NeighborConnectivity[V, E]`) to `__class_getitem__`, since a metaclass `__getitem__` shadows it.

## Consequences

- An unstructured connectivity is spelled once. The provider key, the offset tag
  and the local dimension are all derived from the declaration.
- A table bound to a connectivity can be checked against its declaration.
- The frontend needs one special case: `V2E.Local` is resolved from the offset
  type, because the type of `V2E` is not the class.
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
