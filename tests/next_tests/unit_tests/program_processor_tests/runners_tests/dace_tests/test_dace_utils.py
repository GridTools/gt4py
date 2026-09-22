# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test utility functions of the dace backend module."""

import dace
import pytest

from gt4py.next.program_processors.runners.dace.lowering import gtir_to_sdfg_utils


def test_safe_replace_symbolic():
    assert gtir_to_sdfg_utils.safe_replace_symbolic(
        dace.symbolic.pystr_to_symbolic("x*x + y"), symbol_mapping={"x": "y", "y": "x"}
    ) == dace.symbolic.pystr_to_symbolic("y*y + x")


def test_local_dimension_size():
    import numpy as np

    from gt4py._core import definitions as core_defs
    from gt4py.next import common
    from gt4py.next.program_processors.runners.dace import sdfg_args

    from next_tests.toy_connectivity import V2E, V2EDim, Vertex, Edge

    def conn_type(max_neighbors: int) -> common.NeighborConnectivityType:
        return common.NeighborConnectivityType(
            domain=(Vertex, V2EDim),
            codomain=Edge,
            skip_value=None,
            dtype=core_defs.dtype(np.int32),
            max_neighbors=max_neighbors,
        )

    sharer_tag = "some.module.V2EShared"
    table_types = {V2E.offset_tag: conn_type(4), sharer_tag: conn_type(4)}
    # a field finds the size in the table keyed by the local dimension
    assert sdfg_args.local_dimension_size("a_field", V2EDim, table_types) == 4
    # a connectivity array has its own
    conn_array = sdfg_args.connectivity_identifier(sharer_tag)
    assert sdfg_args.local_dimension_size(conn_array, V2EDim, {sharer_tag: conn_type(4)}) == 4
    # a field over a local dimension bound only through a sharing connectivity
    assert sdfg_args.local_dimension_size("a_field", V2EDim, {sharer_tag: conn_type(4)}) == 4
    with pytest.raises(KeyError, match="No connectivity over the local dimension"):
        sdfg_args.local_dimension_size("a_field", V2EDim, {})
