# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import pytest

dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util


def test_complex_copies_global_access_node():
    N = 64
    K = 80
    sdfg = dace.SDFG(gtx_transformations.utils.unique_name("vertically_implicit_solver_like_sdfg"))
    A, _ = sdfg.add_array("A", [N, K + 1], dtype=dace.float64)
    B, _ = sdfg.add_array("B", [N, K + 1], dtype=dace.float64)
    tmp0, _ = sdfg.add_temp_transient([N], dtype=dace.float64)
    tmp1, _ = sdfg.add_temp_transient([N, K + 1], dtype=dace.float64)
    tmp2, _ = sdfg.add_temp_transient([N, K + 1], dtype=dace.float64)

    st = sdfg.add_state()
    A_node = st.add_access(A)
    A_node_copy = copy.deepcopy(A_node)
    B_node = st.add_access(B)
    tmp0_node = st.add_access(tmp0)
    tmp1_node = st.add_access(tmp1)
    tmp2_node = st.add_access(tmp2)

    st.add_nedge(
        tmp0_node,
        A_node,
        dace.Memlet(data=tmp0, subset=f"0:{N}", other_subset=f"0:{N}, {K}:{K + 1}"),
    )

    st.add_mapped_tasklet(
        "map0",
        map_ranges={"__i": f"0:{N}"},
        code="__out = 0.0",
        inputs={},
        outputs={
            "__out": dace.Memlet(data=tmp0, subset="__i"),
        },
        input_nodes={},
        output_nodes={tmp0_node},
        external_edges=True,
    )

    st.add_mapped_tasklet(
        "map1",
        map_ranges={"__i": f"0:{N}", "__j": f"0:{K}"},
        code="__out = 0.42",
        inputs={},
        outputs={
            "__out": dace.Memlet(data=tmp1, subset="__i, __j"),
        },
        input_nodes={},
        output_nodes={tmp1_node},
        external_edges=True,
    )

    st.add_nedge(
        A_node,
        tmp1_node,
        dace.Memlet(data=A, subset=f"0:{N}, {K}:{K + 1}", other_subset=f"0:{N}, {K}:{K + 1}"),
    )
    st.add_nedge(
        tmp1_node,
        tmp2_node,
        dace.Memlet(data=tmp1, subset=f"0:{N}, 0:1", other_subset=f"0:{N}, 0:1"),
    )
    st.add_nedge(
        tmp1_node,
        tmp2_node,
        dace.Memlet(data=tmp1, subset=f"0:{N}, {K // 2}:{K}", other_subset=f"0:{N}, {K // 2}:{K}"),
    )
    st.add_nedge(
        tmp1_node,
        tmp2_node,
        dace.Memlet(data=tmp1, subset=f"0:{N}, {K}:{K + 1}", other_subset=f"0:{N}, {K}:{K + 1}"),
    )

    st.add_mapped_tasklet(
        "map2",
        map_ranges={"__i": f"0:{N}", "__j": f"1:{K // 2}"},
        code="__out = __inp + 0.5",
        inputs={
            "__inp": dace.Memlet(data=tmp1, subset="__i, __j"),
        },
        outputs={
            "__out": dace.Memlet(data=tmp2, subset="__i, __j"),
        },
        input_nodes={tmp1_node},
        output_nodes={tmp2_node},
        external_edges=True,
    )

    st.add_nedge(
        tmp2_node,
        A_node_copy,
        dace.Memlet(data=tmp2, subset=f"0:{N}, 0:1", other_subset=f"0:{N}, 0:1"),
    )
    st.add_nedge(
        tmp2_node,
        A_node_copy,
        dace.Memlet(data=tmp2, subset=f"0:{N}, 1:{K // 2}", other_subset=f"0:{N}, 1:{K // 2}"),
    )
    st.add_nedge(
        tmp2_node,
        A_node_copy,
        dace.Memlet(data=tmp2, subset=f"0:{N}, {K // 2}:{K}", other_subset=f"0:{N}, {K // 2}:{K}"),
    )

    st.add_mapped_tasklet(
        "map3",
        map_ranges={"__i": f"0:{N}", "__j": f"0:{K}"},
        code="__out = __inp + 0.5",
        inputs={
            "__inp": dace.Memlet(data=tmp2, subset="__i, __j"),
        },
        outputs={
            "__out": dace.Memlet(data=B, subset="__i, __j"),
        },
        input_nodes={tmp2_node},
        output_nodes={B_node},
        external_edges=True,
    )

    sdfg.validate()
    ac_nodes = util.count_nodes(sdfg, dace_nodes.AccessNode, return_nodes=True)
    assert len(ac_nodes) == 6
    assert len([node for node in ac_nodes if node.data == "A"]) == 2
    assert len([node for node in ac_nodes if "tmp" in node.data]) == 3

    edges = list(sdfg.all_edges_recursive())
    assert len(edges) == 22

    res, ref = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    ret = sdfg.apply_transformations_repeated(
        gtx_transformations.RemoveAccessNodeCopies(),
        validate=True,
        validate_all=True,
    )

    gtx_transformations.gt_simplify(
        sdfg,
        validate=True,
        validate_all=True,
    )

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)

    assert ret == 1
    ac_nodes = util.count_nodes(sdfg, dace_nodes.AccessNode, return_nodes=True)
    assert len(ac_nodes) == 3
    assert len([node for node in ac_nodes if node.data == "A"]) == 2
    assert len([node for node in ac_nodes if "tmp0" in node.data]) == 0  # tmp0 removed

    reduced_edges = list(sdfg.all_edges_recursive())
    assert len(reduced_edges) == 14
