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
from dace import subsets as dace_subsets

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
    A_node = st.add_access(A)
    B_node = st.add_access(B)
    C_node = st.add_access(C)
    tmp_node = st.add_access(tmp)

    st.add_mapped_tasklet(
        "map1",
        map_ranges={"__i": f"0:{N}"},
        code="__out = __inp + 1",
        inputs={
            "__inp": dace.Memlet(data=A, subset="__i"),
        },
        outputs={
            "__out": dace.Memlet(data=tmp, subset="__i"),
        },
        input_nodes={A_node},
        output_nodes={tmp_node},
        external_edges=True,
    )

    st.add_mapped_tasklet(
        "map2",
        map_ranges={"__i": f"1:{N}"},
        code="__out = __inp + 0.5",
        inputs={
            "__inp": dace.Memlet(data=tmp, subset="__i"),
        },
        outputs={
            "__out": dace.Memlet(data=B, subset="__i"),
        },
        input_nodes={tmp_node},
        output_nodes={B_node},
        external_edges=True,
    )

    st.add_nedge(C_node, B_node, dace.Memlet(data=C, subset="0", other_subset="0"))

    sdfg.validate()
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 2

    res, ref = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    ret = gtx_transformations.gt_vertical_map_fusion(
        sdfg=sdfg,
        run_simplify=True,
        consolidate_edges_only_if_not_extending=False,
        validate=True,
        validate_all=True,
    )

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)

    # It will apply `VerticalSplitMapRange` on the first Map, then run
    #  `SplitAccessNode`  and finally call MapFusion.
    assert ret == 3
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 1

    map_entry = next(node for node in st.nodes() if isinstance(node, dace_nodes.MapEntry))
    assert map_entry.map.range == dace_subsets.Range.from_string(f"1:{N}")

    # `A`, `B` and `C` as well as a transient inside the map.
    ac_nodes = util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert len(ac_nodes) == 4
    assert all(g_node in ac_nodes for g_node in [A_node, B_node, C_node])
    transient_node = next(iter(ac for ac in ac_nodes if ac.desc(sdfg).transient))
    assert st.scope_dict()[transient_node] is map_entry
