# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import numpy as np
import pytest

dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes
from dace import subsets as dace_subsets

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util


def serial_map_sdfg(N, extra_intermediate_edge=False):
    sdfg = dace.SDFG(
        util.unique_name("serial_map" if extra_intermediate_edge else "serial_map_extra_edge")
    )
    A, _ = sdfg.add_array("A", [N], dtype=dace.float64)
    B, _ = sdfg.add_array("B", [N], dtype=dace.float64)
    C, _ = sdfg.add_scalar("C", dace.float64)
    if extra_intermediate_edge:
        D, _ = sdfg.add_array("D", [2], dtype=dace.float64)
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

    if extra_intermediate_edge:
        D_node = st.add_access(D)
        assert N >= 2
        st.add_nedge(
            tmp_node, D_node, dace.Memlet(data=tmp, subset=f"{N}-2:{N}", other_subset="0:2")
        )

    sdfg.validate()

    return sdfg, A_node, B_node, C_node


def test_vertical_map_fusion():
    N = 80
    sdfg, A_node, B_node, C_node = serial_map_sdfg(N, extra_intermediate_edge=False)
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 2

    res, ref = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    ret = gtx_transformations.gt_vertical_map_split_fusion(
        sdfg=sdfg,
        run_simplify=True,
        run_map_fusion=False,
        fuse_map_fragments=True,
        consolidate_edges_only_if_not_extending=False,
        validate=True,
        validate_all=True,
    )

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)

    assert ret == 1
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 1

    assert len(sdfg.states()) == 1
    st = next(iter(sdfg.states()))

    map_entry = next(node for node in st.nodes() if isinstance(node, dace_nodes.MapEntry))
    assert map_entry.map.range == dace_subsets.Range.from_string(f"1:{N}")

    # `A`, `B` and `C` as well as a transient inside the map.
    ac_nodes = util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert len(ac_nodes) == 4
    assert all(g_node in ac_nodes for g_node in [A_node, B_node, C_node])
    transient_node = next(iter(ac for ac in ac_nodes if ac.desc(sdfg).transient))
    assert st.scope_dict()[transient_node] is map_entry


def test_vertical_map_fusion_disabled():
    N = 80
    sdfg, _, _, _ = serial_map_sdfg(N, extra_intermediate_edge=True)
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 2

    res, ref = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    ret = gtx_transformations.gt_vertical_map_split_fusion(
        sdfg=sdfg,
        run_simplify=True,
        run_map_fusion=True,
        fuse_map_fragments=True,
        consolidate_edges_only_if_not_extending=False,
        validate=True,
        validate_all=True,
    )

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)

    # Check that vertical map split doesn't happen if part of the intermediate
    # access node is used outside the maps.
    assert ret == 0
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 2


@pytest.mark.parametrize("run_map_fusion", [True, False])
def test_vertical_map_fusion_with_neighbor_access(run_map_fusion: bool):
    N = 80
    sdfg = dace.SDFG(util.unique_name("simple"))
    A, _ = sdfg.add_array("A", shape=(N,), dtype=dace.float64, strides=(1,))
    B, _ = sdfg.add_array("B", shape=(N,), dtype=dace.float64, strides=(1,))
    C, _ = sdfg.add_array("C", shape=(N,), dtype=dace.float64, strides=(1,))
    D, _ = sdfg.add_array("D", shape=(N,), dtype=dace.float64, strides=(1,))
    E, _ = sdfg.add_array("E", shape=(N,), dtype=dace.float64, strides=(1,))
    F, _ = sdfg.add_array("F", shape=(N,), dtype=dace.float64, strides=(1,))
    E2C, _ = sdfg.add_array("gt_conn_E2C", shape=(N, 2), dtype=dace.int32, strides=(2, 1))
    tmp, _ = sdfg.add_temp_transient(shape=(N,), dtype=dace.float64, strides=(1,))
    tmp2, _ = sdfg.add_temp_transient(shape=(N,), dtype=dace.float64, strides=(1,))
    b_out, _ = sdfg.add_array("b_out", shape=(2,), dtype=dace.float64, transient=True, strides=(1,))
    d_out, _ = sdfg.add_array("d_out", shape=(2,), dtype=dace.float64, transient=True, strides=(1,))
    t, _ = sdfg.add_scalar("t", dace.float64, transient=True)
    t2, _ = sdfg.add_scalar("t2", dace.float64, transient=True)

    st = sdfg.add_state()
    A_node = st.add_access(A)
    B_node = st.add_access(B)
    C_node = st.add_access(C)
    D_node = st.add_access(D)
    E_node = st.add_access(E)
    F_node = st.add_access(F)
    E2C_node = st.add_access(E2C)
    tmp_node = st.add_access(tmp)
    tmp2_node = st.add_access(tmp2)
    b_out_node = st.add_access(b_out)
    d_out_node = st.add_access("d_out")
    t_node = st.add_access("t")
    t2_node = st.add_access("t2")

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

    mentry, mexit = st.add_map(
        "neighbor_map",
        ndrange={"__i": f"0:{N}"},
    )
    for name in ["tmp", "gt_conn_E2C"]:
        mentry.add_scope_connectors(name)
    st.add_edge(tmp_node, None, mentry, "IN_tmp", dace.Memlet(data=tmp, subset="0:80"))
    st.add_edge(E2C_node, None, mentry, "IN_gt_conn_E2C", dace.Memlet(data=E2C, subset="0:80, 0:2"))
    st.add_edge(mexit, "OUT_B", B_node, None, dace.Memlet(data=B, subset="0:80"))

    mnentry, mnexit = st.add_map(
        "reduction",
        ndrange={"__j": "0:2"},
    )
    st.add_edge(mentry, "OUT_tmp", mnentry, "IN_tmp", dace.Memlet(data=tmp, subset="0:80"))
    st.add_edge(
        mentry,
        "OUT_gt_conn_E2C",
        mnentry,
        "IN_gt_conn_E2C",
        dace.Memlet(data=E2C, subset="__i, 0:2"),
    )
    for name in ["tmp", "gt_conn_E2C"]:
        mnentry.add_scope_connectors(name)
    reduction_tasklet = st.add_tasklet(
        "reduction_tasklet",
        inputs={"__tmp", "__gt_conn_E2C"},
        outputs={"__b_out"},
        code="__b_out = __tmp[__gt_conn_E2C]",
    )
    st.add_edge(
        mnentry, "OUT_tmp", reduction_tasklet, "__tmp", dace.Memlet(data=tmp, subset="0:80")
    )
    st.add_edge(
        mnentry,
        "OUT_gt_conn_E2C",
        reduction_tasklet,
        "__gt_conn_E2C",
        dace.Memlet(data=E2C, subset="__i, __j"),
    )
    st.add_edge(
        reduction_tasklet, "__b_out", mnexit, "IN_b_out", dace.Memlet(data=b_out, subset="__j")
    )
    st.add_edge(mnexit, "OUT_b_out", b_out_node, None, dace.Memlet(data=b_out, subset="0:2"))
    red = st.add_reduce(
        wcr="lambda a, b: a + b",
        axes=None,
        identity=0.0,
    )
    red.add_in_connector("IN_b_out")
    red.add_out_connector("OUT_t")
    st.add_edge(b_out_node, None, red, "IN_b_out", dace.Memlet(data=b_out, subset="0:2"))
    st.add_edge(red, "OUT_t", t_node, None, dace.Memlet(data=t, subset="0"))
    st.add_edge(t_node, None, mexit, "IN_B", dace.Memlet(data=B, subset="__i"))
    mnexit.add_in_connector("IN_b_out")
    mnexit.add_out_connector("OUT_b_out")
    mexit.add_in_connector("IN_B")
    mexit.add_out_connector("OUT_B")

    st.add_mapped_tasklet(
        "map3",
        map_ranges={"__i": f"0:{N}"},
        code="__out = __inp + 2",
        inputs={
            "__inp": dace.Memlet(data=C, subset="__i"),
        },
        outputs={
            "__out": dace.Memlet(data=tmp2, subset="__i"),
        },
        input_nodes={C_node},
        output_nodes={tmp2_node},
        external_edges=True,
    )

    mentry2, mexit2 = st.add_map(
        "neighbor_map2",
        ndrange={"__i": f"0:{N}"},
    )
    for name in ["tmp2", "F", "gt_conn_E2C"]:
        mentry2.add_scope_connectors(name)
    st.add_edge(tmp2_node, None, mentry2, "IN_tmp2", dace.Memlet(data=tmp2, subset="0:80"))
    st.add_edge(
        E2C_node, None, mentry2, "IN_gt_conn_E2C", dace.Memlet(data=E2C, subset="0:80, 0:2")
    )
    st.add_edge(F_node, None, mentry2, "IN_F", dace.Memlet(data=F, subset="0:80"))
    st.add_edge(mexit2, "OUT_D", D_node, None, dace.Memlet(data=D, subset="0:80"))

    mnentry2, mnexit2 = st.add_map(
        "reduction2",
        ndrange={"__j": "0:2"},
    )
    st.add_edge(mentry2, "OUT_F", mnentry2, "IN_F", dace.Memlet(data=F, subset="0:80"))
    st.add_edge(
        mentry2,
        "OUT_gt_conn_E2C",
        mnentry2,
        "IN_gt_conn_E2C",
        dace.Memlet(data=E2C, subset="__i, 0:2"),
    )
    for name in ["F", "gt_conn_E2C"]:
        mnentry2.add_scope_connectors(name)
    reduction_tasklet = st.add_tasklet(
        "reduction_tasklet2",
        inputs={"__F", "__gt_conn_E2C"},
        outputs={"__d_out"},
        code="__d_out = __F[__gt_conn_E2C]",
    )
    st.add_edge(mnentry2, "OUT_F", reduction_tasklet, "__F", dace.Memlet(data=F, subset="0:80"))
    st.add_edge(
        mnentry2,
        "OUT_gt_conn_E2C",
        reduction_tasklet,
        "__gt_conn_E2C",
        dace.Memlet(data=E2C, subset="__i, __j"),
    )
    st.add_edge(
        reduction_tasklet, "__d_out", mnexit2, "IN_d_out", dace.Memlet(data=d_out, subset="__j")
    )
    st.add_edge(mnexit2, "OUT_d_out", d_out_node, None, dace.Memlet(data=d_out, subset="0:2"))
    mnexit2.add_in_connector("IN_d_out")
    mnexit2.add_out_connector("OUT_d_out")
    red2 = st.add_reduce(
        wcr="lambda a, b: a + b",
        axes=None,
        identity=0.0,
    )
    red2.add_in_connector("IN_d_out")
    red2.add_out_connector("OUT_t2")
    st.add_edge(d_out_node, None, red2, "IN_d_out", dace.Memlet(data=d_out, subset="0:2"))
    st.add_edge(red2, "OUT_t2", t2_node, None, dace.Memlet(data=t2, subset="0"))
    tasklet2 = st.add_tasklet(
        "tasklet2",
        inputs={"__inp1", "__inp2"},
        outputs={"__out"},
        code="__out = __inp1 + __inp2",
    )
    st.add_edge(t2_node, None, tasklet2, "__inp1", dace.Memlet(data=t2, subset="0"))
    st.add_edge(mentry2, "OUT_tmp2", tasklet2, "__inp2", dace.Memlet(data=tmp2, subset="__i"))
    st.add_edge(tasklet2, "__out", mexit2, "IN_D", dace.Memlet(data=D, subset="__i"))
    mexit2.add_in_connector("IN_D")
    mexit2.add_out_connector("OUT_D")

    st.add_mapped_tasklet(
        "map5",
        map_ranges={"__i": f"0:{N}"},
        code="__out = __inp1 + __inp2",
        inputs={
            "__inp1": dace.Memlet(data=B, subset="__i"),
            "__inp2": dace.Memlet(data=D, subset="__i"),
        },
        outputs={
            "__out": dace.Memlet(data=E, subset="__i"),
        },
        input_nodes={B_node, D_node},
        output_nodes={E_node},
        external_edges=True,
    )

    sdfg.validate()
    initial_map_entries_nb = util.count_nodes(sdfg, dace_nodes.MapEntry)
    assert initial_map_entries_nb == 7

    res, ref = util.make_sdfg_args(sdfg)
    ref["gt_conn_E2C"] = np.random.randint(0, N, ref["gt_conn_E2C"].shape, dtype=np.int32)
    res["gt_conn_E2C"] = copy.deepcopy(ref["gt_conn_E2C"])
    util.compile_and_run_sdfg(sdfg, **ref)

    ret = gtx_transformations.gt_vertical_map_split_fusion(
        sdfg=sdfg,
        run_simplify=True,
        run_map_fusion=run_map_fusion,
        fuse_map_fragments=True,
        consolidate_edges_only_if_not_extending=False,
        validate=True,
        validate_all=True,
    )

    util.compile_and_run_sdfg(sdfg, **res)
    # TODO(iomaganaris): Enable assertion for the result. Currently, the assertion fails on MacOS
    # with random neighbor indexes in E2C.
    assert util.compare_sdfg_res(ref=ref, res=res)

    if run_map_fusion:
        # Although no map could be split, map fusion was activated and was able to
        #  fuse 3 Maps.
        assert ret == 3
        assert util.count_nodes(sdfg, dace_nodes.MapEntry) == initial_map_entries_nb - ret

    else:
        # `VerticalSplitMapRange` cannot be applied on the map that has neighbor access
        # to the temporary field.
        assert ret == 0
        assert util.count_nodes(sdfg, dace_nodes.MapEntry) == initial_map_entries_nb
