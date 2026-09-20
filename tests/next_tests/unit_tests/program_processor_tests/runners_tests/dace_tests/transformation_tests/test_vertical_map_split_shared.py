# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy

import dace
import numpy as np
import pytest
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations
from gt4py.next.program_processors.runners.dace.transformations import map_fusion_extended


def _make_sdfg(*, upper_consumer=True, later_state=False):
    sdfg = dace.SDFG("shared_output_bands")
    for name in ("input", "theta", "rho", "wind"):
        sdfg.add_array(name, [6, 12], dace.float64)
    state = sdfg.add_state()
    access = {name: state.add_access(name) for name in sdfg.arrays}
    _, producer, producer_exit = state.add_mapped_tasklet(
        "producer",
        {"i": "1:5", "k": "0:12"},
        {"a": dace.Memlet("input[i,k]")},
        "t = a + 1; r = a * 2",
        {"t": dace.Memlet("theta[i,k]"), "r": dace.Memlet("rho[i,k]")},
        input_nodes={access["input"]},
        output_nodes={access["theta"], access["rho"]},
        external_edges=True,
    )
    consumers = []
    for name, bounds, factor in [("lower", "0:3", 3), ("upper", "3:12", 4)]:
        if name == "upper" and not upper_consumer:
            continue
        _, entry, _ = state.add_mapped_tasklet(
            name,
            {"i": "1:5", "k": bounds},
            {"t": dace.Memlet("theta[i,k]")},
            f"w = t * {factor}",
            {"w": dace.Memlet("wind[i,k]")},
            input_nodes={access["theta"]},
            output_nodes={access["wind"]},
            external_edges=True,
        )
        consumers.append(entry)
    if later_state:
        sdfg.add_array("later", [6, 12], dace.float64)
        following = sdfg.add_state_after(state)
        following.add_mapped_tasklet(
            "read_external_output",
            {"i": "0:6", "k": "0:12"},
            {"t": dace.Memlet("theta[i,k]")},
            "out = t + 7",
            {"out": dace.Memlet("later[i,k]")},
            external_edges=True,
        )
    sdfg.validate()
    return sdfg, state, access, producer, producer_exit, consumers


@pytest.mark.parametrize("allow_shared_data", [False, True])
@pytest.mark.parametrize("fuse_map_fragments", [False, True])
@pytest.mark.parametrize("upper_consumer", [False, True])
def test_shared_output_split_preserves_values(
    tmp_path, allow_shared_data, fuse_map_fragments, upper_consumer
):
    sdfg, state, _, _, _, _ = _make_sdfg(upper_consumer=upper_consumer, later_state=True)
    external_descriptors = {name: desc.to_json() for name, desc in sdfg.arrays.items()}
    gtx_transformations.gt_vertical_map_split_fusion(
        sdfg,
        run_simplify=False,
        run_map_fusion=not fuse_map_fragments,
        consolidate_edges_only_if_not_extending=True,
        fuse_map_fragments=fuse_map_fragments,
        allow_shared_data=allow_shared_data,
        validate=True,
        validate_all=True,
    )
    maps = [node for node in state.nodes() if isinstance(node, dace_nodes.MapEntry)]
    assert len(maps) == (2 if allow_shared_data else 2 + upper_consumer)
    if allow_shared_data:
        assert {str(node.map.range) for node in maps} == {"1:5, 0:3", "1:5, 3:12"}
    for name, desc in external_descriptors.items():
        assert sdfg.arrays[name].to_json() == desc

    gtx_transformations.gt_simplify(sdfg, validate=True, validate_all=True)
    rng = np.random.default_rng(42)
    args = {name: rng.random((6, 12)) for name in external_descriptors}
    expected = copy.deepcopy(args)
    expected["theta"][1:5] = args["input"][1:5] + 1
    expected["rho"][1:5] = args["input"][1:5] * 2
    expected["wind"][1:5, :3] = expected["theta"][1:5, :3] * 3
    if upper_consumer:
        expected["wind"][1:5, 3:] = expected["theta"][1:5, 3:] * 4
    expected["later"][:] = expected["theta"] + 7
    sdfg.build_folder = str(tmp_path / "build")
    sdfg.name += f"_{allow_shared_data}_{fuse_map_fragments}_{upper_consumer}"
    compiled = sdfg.compile()
    compiled(**args)
    for name in expected:
        np.testing.assert_allclose(args[name], expected[name], rtol=1e-14, atol=0)


@pytest.mark.parametrize(
    "unsafe",
    [
        "shifted_read",
        "crossing_consumer",
        "overlapping_writes",
        "alias",
        "other_access",
        "dynamic",
        "reduction",
    ],
)
def test_shared_output_split_rejects_unsafe_partition(unsafe):
    sdfg, state, access, _, producer_exit, consumers = _make_sdfg()
    theta = access["theta"]
    if unsafe == "shifted_read":
        outer = next(edge for edge in state.out_edges(theta) if edge.dst is consumers[0])
        outer.data.subset = dace.subsets.Range.from_string("1:5, 1:4")
        for edge in state.out_edges_by_connector(consumers[0], "OUT_" + outer.dst_conn[3:]):
            edge.data.subset = dace.subsets.Range.from_string("i, k+1")
    elif unsafe == "crossing_consumer":
        sdfg.add_array("extra", [6, 12], dace.float64)
        state.add_mapped_tasklet(
            "crossing",
            {"i": "1:5", "k": "2:4"},
            {"t": dace.Memlet("theta[i,k]")},
            "out = t",
            {"out": dace.Memlet("extra[i,k]")},
            input_nodes={theta},
            external_edges=True,
        )
    elif unsafe == "overlapping_writes":
        state.add_mapped_tasklet(
            "overlap",
            {"i": "1:5", "k": "0:1"},
            {},
            "out = 0",
            {"out": dace.Memlet("theta[i,k]")},
            output_nodes={theta},
            external_edges=True,
        )
    elif unsafe == "alias":
        sdfg.arrays["theta"].may_alias = True
    elif unsafe == "other_access":
        state.add_access("theta")
    elif unsafe == "dynamic":
        next(iter(state.out_edges(theta))).data.dynamic = True
    elif unsafe == "reduction":
        next(iter(state.in_edges(theta))).data.wcr = "lambda a, b: a + b"
    before = sdfg.to_json()
    assert not map_fusion_extended.VerticalSplitMapRange.can_be_applied_to(
        sdfg,
        options={"allow_shared_data": True},
        first_map_exit=producer_exit,
        access_node=theta,
        second_map_entry=consumers[0],
    )
    assert sdfg.to_json() == before


def test_shared_output_split_respects_callback():
    sdfg, _, _, _, _, _ = _make_sdfg()
    before = sdfg.to_json()
    assert (
        gtx_transformations.gt_vertical_map_split_fusion(
            sdfg,
            run_simplify=False,
            run_map_fusion=False,
            consolidate_edges_only_if_not_extending=True,
            fuse_map_fragments=True,
            allow_shared_data=True,
            check_split_callback=lambda *args: False,
        )
        == 0
    )
    assert sdfg.to_json() == before


def test_shared_output_split_can_be_selected_by_optimizer_hook():
    sdfg, state, _, _, _, _ = _make_sdfg()
    selected = []

    def select_theta(transformation, first_map, second_map, graph, sdfg):
        transformation.allow_shared_data = transformation.access_node.data == "theta"
        if transformation.allow_shared_data:
            selected.append(transformation.access_node.data)
        return True

    gtx_transformations.gt_auto_optimize(
        sdfg,
        gpu=False,
        optimization_hooks={
            gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowVerticalSplitCallBack: select_theta
        },
        validate=True,
        validate_all=True,
    )
    assert selected
    assert len([node for node in state.nodes() if isinstance(node, dace_nodes.MapEntry)]) == 2
