# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for the names under which one connectivity is used.

With `FieldOffset`, using a connectivity required four independently authored strings to
agree -- the offset tag, the Python variable it was bound to, the local dimension's name and
the offset-provider key -- and each execution path silently depended on a different subset of
them. A `NeighborConnectivity` declaration produces all of them (ADR 0029), so what is left to
pin is that the *Python* name a declaration is reached through does not matter.
"""

import numpy as np
import pytest

from gt4py import next as gtx
from gt4py.next import Dims, Field, common, constructors, neighbor_sum

from next_tests import definitions as test_defs
from next_tests.integration_tests import cases
from next_tests.integration_tests import cases_utils
from next_tests.integration_tests.cases_utils import (  # noqa: F401 [unused-import] # fixture
    exec_alloc_descriptor,
)


class V(gtx.DimensionIndex): ...


class E(gtx.DimensionIndex): ...


class V2E(gtx.NeighborConnectivity[V, E]):
    class Local(gtx.LocalDimensionIndex): ...


#: The declaration, reached through a different Python name.
off_a = V2E
#: Its local dimension, likewise.
Neigh = V2E.Local


@pytest.fixture
def case(exec_alloc_descriptor) -> cases.Case:
    mesh = cases_utils.simple_mesh(exec_alloc_descriptor.allocator)
    # NOTE: `.asnumpy()`, not `.ndarray`: under a GPU allocator the latter is a device
    # array, and `simple_mesh` builds the table from NumPy anyway.
    v2e_arr = mesh.offset_provider[cases_utils.V2E].asnumpy()
    return cases.Case(
        (
            None
            if isinstance(exec_alloc_descriptor, test_defs.EmbeddedDummyBackend)
            else exec_alloc_descriptor
        ),
        offset_provider={
            off_a: constructors.as_connectivity(
                domain={V: v2e_arr.shape[0], Neigh: v2e_arr.shape[1]},
                codomain=E,
                data=v2e_arr,
                skip_value=None,
                allocator=exec_alloc_descriptor.allocator,
            )
        },
        default_sizes={V: mesh.num_vertices, E: mesh.num_edges},
        grid_type=common.GridType.UNSTRUCTURED,
        allocator=exec_alloc_descriptor.allocator,
    )


def _neighbor_table(case: cases.Case) -> np.ndarray:
    return case.offset_provider[V2E].asnumpy()


def test_shift_through_an_alias(case):
    @gtx.field_operator
    def foo(a: Field[Dims[E], float]) -> Field[Dims[V], float]:
        return a(off_a[1])

    cases.verify_with_default_data(case, foo, lambda a: a[_neighbor_table(case)[:, 1]])


def test_reduction_through_an_alias(case):
    @gtx.field_operator
    def foo(a: Field[Dims[E], float]) -> Field[Dims[V], float]:
        return neighbor_sum(a(off_a), axis=Neigh)

    cases.verify_with_default_data(case, foo, lambda a: np.sum(a[_neighbor_table(case)], axis=1))


def test_reduction_over_the_nested_name(case):
    @gtx.field_operator
    def foo(a: Field[Dims[E], float]) -> Field[Dims[V], float]:
        return neighbor_sum(a(V2E), axis=off_a.Local)

    cases.verify_with_default_data(case, foo, lambda a: np.sum(a[_neighbor_table(case)], axis=1))
