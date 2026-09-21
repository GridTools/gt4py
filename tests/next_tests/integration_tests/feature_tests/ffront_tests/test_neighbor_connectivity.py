# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""A `NeighborConnectivity` declaration used directly in DSL code, on every backend."""

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import Dims, Field, common, constructors, neighbor_sum

from next_tests import definitions as test_defs
from next_tests.integration_tests import cases, cases_utils
from next_tests.integration_tests.cases_utils import (  # noqa: F401 [unused-import] # fixture
    exec_alloc_descriptor,
)


class V(gtx.DimensionIndex): ...


class E(gtx.DimensionIndex): ...


class V2E(gtx.NeighborConnectivity[V, E], max_neighbors=4, min_neighbors=4):
    class Local(gtx.LocalDimensionIndex): ...


#: A second connectivity over the same neighbor axis, bound to a different table.
class V2EShared(gtx.NeighborConnectivity[V, E]):
    Local = V2E.Local


@pytest.fixture
def case(exec_alloc_descriptor):
    mesh = cases_utils.simple_mesh(exec_alloc_descriptor.allocator)
    v2e_arr = mesh.offset_provider[cases_utils.V2EDim.tag].asnumpy()
    table = constructors.as_connectivity(
        domain={V: v2e_arr.shape[0], V2E.Local: v2e_arr.shape[1]},
        codomain=E,
        data=v2e_arr,
        skip_value=None,
        allocator=exec_alloc_descriptor.allocator,
    )
    common.check_neighbor_table(V2E, table)
    shared_table = constructors.as_connectivity(
        domain={V: v2e_arr.shape[0], V2E.Local: v2e_arr.shape[1]},
        codomain=E,
        data=np.ascontiguousarray(v2e_arr[:, ::-1]),
        skip_value=None,
        allocator=exec_alloc_descriptor.allocator,
    )
    common.check_neighbor_table(V2EShared, shared_table)
    return cases.Case(
        (
            None
            if isinstance(exec_alloc_descriptor, test_defs.EmbeddedDummyBackend)
            else exec_alloc_descriptor
        ),
        # NOTE: still keyed on the local dimension's tag; class keys come with the removal of
        # `FieldOffset`.
        offset_provider={V2E.offset_tag: table, V2EShared.offset_tag: shared_table},
        default_sizes={V: mesh.num_vertices, E: mesh.num_edges, V2E.Local: v2e_arr.shape[1]},
        grid_type=common.GridType.UNSTRUCTURED,
        allocator=exec_alloc_descriptor.allocator,
    )


def _table(case: cases.Case, connectivity=V2E) -> np.ndarray:
    return case.offset_provider[connectivity.offset_tag].asnumpy()


@pytest.mark.uses_unstructured_shift
def test_shift(case):
    @gtx.field_operator
    def testee(a: Field[Dims[E], float]) -> Field[Dims[V], float]:
        return a(V2E[1])

    cases.verify_with_default_data(case, testee, lambda a: a[_table(case)[:, 1]])


@pytest.mark.uses_unstructured_shift
def test_reduction(case):
    @gtx.field_operator
    def testee(a: Field[Dims[E], float]) -> Field[Dims[V], float]:
        return neighbor_sum(a(V2E), axis=V2E.Local)

    cases.verify_with_default_data(case, testee, lambda a: np.sum(a[_table(case)], axis=1))


@pytest.mark.uses_unstructured_shift
def test_sparse_argument(case):
    @gtx.field_operator
    def testee(
        s: Field[Dims[V, V2E.Local], float], a: Field[Dims[E], float]
    ) -> Field[Dims[V], float]:
        return neighbor_sum(s * a(V2E), axis=V2E.Local)

    cases.verify_with_default_data(case, testee, lambda s, a: np.sum(s * a[_table(case)], axis=1))


@pytest.mark.uses_unstructured_shift
def test_program(case):
    @gtx.field_operator
    def shift_by_one(a: Field[Dims[E], float]) -> Field[Dims[V], float]:
        return a(V2E[0])

    @gtx.program
    def testee(a: Field[Dims[E], float], out: Field[Dims[V], float]):
        shift_by_one(a, out=out)

    cases.verify_with_default_data(case, testee, lambda a: a[_table(case)[:, 0]])


@pytest.mark.uses_unstructured_shift
@pytest.mark.uses_offset_tag_differing_from_local_dim
def test_shift_through_a_shared_local_dimension(case):
    @gtx.field_operator
    def testee(a: Field[Dims[E], float]) -> Field[Dims[V], float]:
        return a(V2EShared[1])

    cases.verify_with_default_data(case, testee, lambda a: a[_table(case, V2EShared)[:, 1]])


@pytest.mark.uses_unstructured_shift
@pytest.mark.uses_offset_tag_differing_from_local_dim
@pytest.mark.uses_offset_tag_differing_from_local_dim_in_reduction
def test_reduction_through_a_shared_local_dimension(case):
    @gtx.field_operator
    def testee(
        s: Field[Dims[V, V2E.Local], float], a: Field[Dims[E], float]
    ) -> Field[Dims[V], float]:
        # combines a sparse field on the shared axis with each connectivity's neighbors
        return neighbor_sum(s * a(V2EShared) - a(V2E), axis=V2E.Local)

    cases.verify_with_default_data(
        case,
        testee,
        lambda s, a: np.sum(s * a[_table(case, V2EShared)] - a[_table(case)], axis=1),
    )
