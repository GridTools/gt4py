# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py import next as gtx
from gt4py.next import Dims, Field, common, constructors

from next_tests import definitions as test_defs
from next_tests.integration_tests import cases
from next_tests.integration_tests.feature_tests.ffront_tests import ffront_test_utils


V = gtx.Dimension("V")
E = gtx.Dimension("E")
Neigh = gtx.Dimension("Neigh", kind=common.DimensionKind.LOCAL)
Off = gtx.FieldOffset("Off", source=E, target=(V, Neigh))


@pytest.fixture
def case():
    exec_alloc_descriptor = test_defs.ProgramBackendId.GTFN_CPU.load()
    mesh = ffront_test_utils.simple_mesh(exec_alloc_descriptor.allocator)
    v2e_arr = mesh.offset_provider["V2E"].ndarray
    return cases.Case(
        exec_alloc_descriptor,
        offset_provider={
            "Off": constructors.as_connectivity(
                domain={V: v2e_arr.shape[0], Neigh: 4},
                codomain=E,
                data=v2e_arr,
                skip_value=None,
                allocator=exec_alloc_descriptor.allocator,
            ),
        },
        default_sizes={
            V: mesh.num_vertices,
            E: mesh.num_edges,
        },
        grid_type=common.GridType.UNSTRUCTURED,
        allocator=exec_alloc_descriptor.allocator,
    )


def test_offset_dimension_name_differ(case):
    """
    Ensure that gtfn works with offset name that differs from the name of the local dimension.

    If the value of the `NeighborConnectivityType.neighbor_dim` did not match the `FieldOffset` value,
    gtfn would silently ignore the neighbor index, see https://github.com/GridTools/gridtools/pull/1814.
    """

    @gtx.field_operator
    def foo(a: Field[Dims[E], float]) -> Field[Dims[V], float]:
        return a(Off[1])

    cases.verify_with_default_data(
        case, foo, lambda a: a[case.offset_provider["Off"].ndarray[:, 1]]
    )
