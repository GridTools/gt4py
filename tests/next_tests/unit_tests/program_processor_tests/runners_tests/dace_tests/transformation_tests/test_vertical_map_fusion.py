# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util


def test_vertical_map_fusion():
    N = 80
    sdfg = dace.SDFG(util.unique_name("simple"))
    A, _ = sdfg.add_array("A", [N], dtype=dace.float64)
    B, _ = sdfg.add_array("B", [N], dtype=dace.float64)
    C, _ = sdfg.add_scalar("C", dace.float64)
    tmp, _ = sdfg.add_temp_transient([N], dtype=dace.float64)

    st = sdfg.add_state()
    out_node = st.add_access(B)
    tmp_node = st.add_access(tmp)

    st.add_mapped_tasklet(
        "map1",
        map_ranges=dict(_i="0:N"),
        code="__out = __inp + 1",
        inputs={
            "__inp": dace.Memlet(data=A, subset="__i"),
        },
        outputs={
            "__out": dace.Memlet(data=tmp, subset="__i"),
        },
        output_nodes={tmp_node},
        external_edges=True,
    )

    st.add_mapped_tasklet(
        "map2",
        map_ranges=dict(_i="1:N"),
        code="__out = __inp - 1",
        inputs={
            "__inp": dace.Memlet(data=tmp, subset="__i"),
        },
        outputs={
            "__out": dace.Memlet(data=B, subset="__i"),
        },
        input_nodes={tmp_node},
        output_nodes={out_node},
        external_edges=True,
    )

    st.add_nedge(st.add_access(C), out_node, dace.Memlet(data=C, subset="0", other_subset="0"))

    sdfg.validate()
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 2

    ret = gtx_transformations.gt_vertical_map_fusion(
        sdfg=sdfg,
        run_simplify=False,
        validate=True,
        validate_all=True,
    )
    assert ret == 1
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 3
