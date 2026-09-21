# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for the four independently authored names of one connectivity.

Using a single connectivity requires four strings to agree, none of which is
checked against the others at declaration time:

    N1  the `FieldOffset` tag              `FieldOffset("V2E", ...)`
    N2  the Python variable it is bound to  `V2E = FieldOffset(...)`
    N3  the local dimension's name          `Dimension("V2E", kind=LOCAL)`
    N4  the offset-provider key             `offset_provider={"V2E": ...}`

The `V2EDim = Dimension("V2E")` convention makes all four equal, which hides
which one each execution path actually uses. These tests break the convention
deliberately, one name at a time, so the real requirement is visible.
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


#: N1 == N3 == N4, but N2 differs: the tag is `TaggedOffDim.tag`, the variable is `off_a`.
class TaggedOffDim(gtx.LocalDimensionIndex): ...


off_a = gtx.FieldOffset(TaggedOffDim.tag, source=E, target=(V, TaggedOffDim))


#: N1 == N2 == N4, but N3 differs: the local dimension is `Neigh`, the tag is `OffB`.
class Neigh(gtx.LocalDimensionIndex): ...


OffB = gtx.FieldOffset("OffB", source=E, target=(V, Neigh))


def _case(exec_alloc_descriptor, tag: str, local_dim: common.Dimension) -> cases.Case:
    """
    A `Case` whose offset provider holds exactly one connectivity, keyed on its tag.

    One entry per `Case` on purpose: DaCe walks every provider entry while building the
    SDFG, and looks a connectivity up by its *local dimension's* name
    (`gtir_to_sdfg.py`, constraint A4). A second, non-conforming entry would therefore
    fail a program that does not even use it, and the cell under test would be measuring
    the wrong thing.
    """
    mesh = cases_utils.simple_mesh(exec_alloc_descriptor.allocator)
    # NOTE: `.asnumpy()`, not `.ndarray`: under a GPU allocator the latter is a device
    # array, and `simple_mesh` builds the table from NumPy anyway.
    v2e_arr = mesh.offset_provider[cases_utils.V2EDim.tag].asnumpy()
    return cases.Case(
        (
            None
            if isinstance(exec_alloc_descriptor, test_defs.EmbeddedDummyBackend)
            else exec_alloc_descriptor
        ),
        offset_provider={
            tag: constructors.as_connectivity(
                domain={V: v2e_arr.shape[0], local_dim: v2e_arr.shape[1]},
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


@pytest.fixture
def case_tag_vs_variable_name(exec_alloc_descriptor):
    return _case(exec_alloc_descriptor, TaggedOffDim.tag, TaggedOffDim)


@pytest.fixture
def case_tag_vs_local_dim(exec_alloc_descriptor):
    return _case(exec_alloc_descriptor, "OffB", Neigh)


def _neighbor_table(case: cases.Case, tag: str) -> np.ndarray:
    return case.offset_provider[tag].asnumpy()


# --- N2: the tag differs from the Python variable name ----------------------------
# Lowering used to emit the *variable* name as the IR shift tag, so embedded and
# compiled execution of the same program needed different provider keys.


def test_shift_tag_differs_from_variable_name(case_tag_vs_variable_name):
    @gtx.field_operator
    def foo(a: Field[Dims[E], float]) -> Field[Dims[V], float]:
        return a(off_a[1])

    cases.verify_with_default_data(
        case_tag_vs_variable_name,
        foo,
        lambda a: a[_neighbor_table(case_tag_vs_variable_name, TaggedOffDim.tag)[:, 1]],
    )


def test_reduction_tag_differs_from_variable_name(case_tag_vs_variable_name):
    @gtx.field_operator
    def foo(a: Field[Dims[E], float]) -> Field[Dims[V], float]:
        return neighbor_sum(a(off_a), axis=TaggedOffDim)

    cases.verify_with_default_data(
        case_tag_vs_variable_name,
        foo,
        lambda a: np.sum(a[_neighbor_table(case_tag_vs_variable_name, TaggedOffDim.tag)], axis=1),
    )


# --- N3: the tag differs from the local dimension's name --------------------------
# Lifted for the gtfn shift path by #1789; still required elsewhere, which is what
# the markers below record.


@pytest.mark.uses_offset_tag_differing_from_local_dim
def test_shift_tag_differs_from_local_dim_name(case_tag_vs_local_dim):
    """
    Ensure a shift works with an offset tag that differs from the local dimension's name.

    If `NeighborConnectivityType.neighbor_dim` did not match the `FieldOffset` value,
    gtfn would silently ignore the neighbor index, see
    https://github.com/GridTools/gridtools/pull/1814.
    """

    @gtx.field_operator
    def foo(a: Field[Dims[E], float]) -> Field[Dims[V], float]:
        return a(OffB[1])

    cases.verify_with_default_data(
        case_tag_vs_local_dim,
        foo,
        lambda a: a[_neighbor_table(case_tag_vs_local_dim, "OffB")[:, 1]],
    )


@pytest.mark.uses_offset_tag_differing_from_local_dim_in_reduction
def test_reduction_tag_differs_from_local_dim_name(case_tag_vs_local_dim):
    @gtx.field_operator
    def foo(a: Field[Dims[E], float]) -> Field[Dims[V], float]:
        return neighbor_sum(a(OffB), axis=Neigh)

    cases.verify_with_default_data(
        case_tag_vs_local_dim,
        foo,
        lambda a: np.sum(a[_neighbor_table(case_tag_vs_local_dim, "OffB")], axis=1),
    )
