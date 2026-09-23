# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Client code that has to type-check under *pyright*, checked by `nox -s test_typing_exports`.

The cases in `test_next.yaml` run under mypy only, and the two checkers disagree about what
counts as a type: an annotated `Local` on a connectivity or its metaclass makes every
declaration's local dimension a *variable* for pyright, so `Field[Dims[V, V2E.Local], float]`
is rejected there while mypy accepts it (see ADR 0029). Everything here must be error-free.
"""

from __future__ import annotations

import typing

from gt4py import next as gtx


class Vertex(gtx.DimensionIndex): ...


class Edge(gtx.DimensionIndex): ...


class Cell(gtx.DimensionIndex): ...


class CellEdge(gtx.DimensionIndex): ...


class KDim(gtx.DimensionIndex, kind=gtx.DimensionKind.VERTICAL): ...


class V2E(gtx.NeighborConnectivity[Vertex, Edge], max_neighbors=6, min_neighbors=5):
    class Local(gtx.LocalDimensionIndex): ...


class C2E(gtx.NeighborConnectivity[Cell, Edge]):
    class Local(gtx.LocalDimensionIndex): ...


#: A flattened sparse pattern sharing `C2E`'s neighbor axis.
class C2CE(gtx.NeighborConnectivity[Cell, CellEdge]):
    Local: typing.TypeAlias = C2E.Local


class LsqCoeff(gtx.LocalDimensionIndex, size=3): ...


#: A declaration adopting a local dimension declared at module level.
class V2EAdopted(gtx.NeighborConnectivity[Vertex, Edge]):
    Local: typing.TypeAlias = LsqCoeff


def nested_local(sparse: gtx.Field[gtx.Dims[Vertex, V2E.Local], gtx.float64]) -> None: ...


def shared_local(sparse: gtx.Field[gtx.Dims[Cell, C2CE.Local], gtx.float64]) -> None: ...


def adopted_local(sparse: gtx.Field[gtx.Dims[Vertex, V2EAdopted.Local], gtx.float64]) -> None: ...


def a_shared_local_is_its_owners(
    owned: gtx.Field[gtx.Dims[Cell, C2E.Local], gtx.float64],
    shared: gtx.Field[gtx.Dims[Cell, C2CE.Local], gtx.float64],
) -> None:
    shared_local(owned)  # the two spellings are one type
    shared_local(shared)


def an_adopted_local_is_the_adopted_one(
    coefficients: gtx.Field[gtx.Dims[Vertex, LsqCoeff], gtx.float64],
) -> None:
    adopted_local(coefficients)


L = typing.TypeVar("L", bound=gtx.LocalDimensionIndex)


def local_of(connectivity: type[gtx.NeighborConnectivity]) -> type[gtx.LocalDimensionIndex]:
    # generic code names a local dimension through the accessor, not through `conn.Local`
    return gtx.local_dimension_of(connectivity)


def first_neighbor(sparse: gtx.Field[gtx.Dims[Vertex, L], gtx.float64]) -> type[L]:
    raise NotImplementedError


def generic_local(sparse: gtx.Field[gtx.Dims[Vertex, V2E.Local], gtx.float64]) -> None:
    typing.assert_type(first_neighbor(sparse), type[V2E.Local])


@gtx.field_operator
def reduce_over_a_local_dimension(
    a: gtx.Field[gtx.Dims[Edge], gtx.float64],
) -> gtx.Field[gtx.Dims[Vertex], gtx.float64]:
    return gtx.neighbor_sum(a(V2E), axis=V2E.Local)


@gtx.field_operator
def shift_by_a_dimension(
    a: gtx.Field[gtx.Dims[KDim], gtx.float64],
) -> gtx.Field[gtx.Dims[KDim], gtx.float64]:
    return a(KDim + 1)
