# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

dace = pytest.importorskip("dace")
import copy
from dace.sdfg import nodes as dace_nodes

from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace import (
    gtir_to_sdfg_utils as gtx_sdfg_utils,
    transformations as gtx_transformations,
)

from . import util


def _make_sdfg() -> tuple[dace.SDFG, dace.SDFGState]:
    sdfg = dace.SDFG(util.unique_name("test"))
    state = sdfg.add_state(is_start_block=True)

    for name in "abcde":
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=name in "bc",
        )
    a, b, c, d, e = (state.add_access(name) for name in "abcde")

    state.add_mapped_tasklet(
        "smap1",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = math.sin(__in)",
        outputs={"__out": dace.Memlet("b[__i]")},
        input_nodes={a},
        output_nodes={b},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "smap2",
        map_ranges={"__j": "0:10"},
        inputs={"__in": dace.Memlet("b[__j]")},
        code="__out = math.cos(__in)",
        outputs={"__out": dace.Memlet("c[__j]")},
        input_nodes={b},
        output_nodes={c},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "pmap3",
        map_ranges={"__j": "0:10"},
        inputs={"__in": dace.Memlet("c[__j]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("d[__j]")},
        input_nodes={c},
        output_nodes={d},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "pmap4",
        map_ranges={"__j": "0:10"},
        inputs={"__in": dace.Memlet("c[__j]")},
        code="__out = __in + 2.0",
        outputs={"__out": dace.Memlet("e[__j]")},
        input_nodes={c},
        output_nodes={e},
        external_edges=True,
    )
    sdfg.validate()

    return sdfg, state


def test_auto_optimizer_without_callbacks():
    sdfg, _ = _make_sdfg()
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 4

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    gtx_transformations.gt_auto_optimize(sdfg, gpu=False)

    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 1

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref, res)


@pytest.mark.parametrize("disable_splitting", [True, False])
def test_auto_optimizer_with_callbacks(disable_splitting: bool):
    sdfg, _ = _make_sdfg()

    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 4

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    # Number of times a callback was called.
    call_count = {}

    def _increase_count(cb):
        call_count[cb] = call_count.get(cb, 0) + 1

    # We do not allow any horizontal map fusions, which means that the last two Maps
    #  can not be fused.
    def horizontal_map_fusion_hook_that_rejects_all_atteps(
        this: gtx_transformations.MapFusionHorizontal,
        first_map_entry: dace_nodes.MapEntry,
        second_map_entry: dace_nodes.MapEntry,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> bool:
        # Because this hook runs before the transformation checks if the match is
        #  applicable we have to manually check if the two Maps are parallel.
        if gtx_transformations.utils.is_reachable(
            start=first_map_entry, target=second_map_entry, state=graph
        ):
            return False
        if gtx_transformations.utils.is_reachable(
            start=second_map_entry, target=first_map_entry, state=graph
        ):
            return False

        # The only candidates for the fusion are `pmap3` and `pmap4` so we ensure that
        #  only they are involved.
        for map_entry in [first_map_entry, second_map_entry]:
            assert any(
                map_entry.map.label.startswith(eligible_map_name)
                for eligible_map_name in ["pmap3", "pmap4"]
            ), f"Found non expected Map name: {map_entry.map.label}"
        _increase_count(
            gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowMapFusionHorizontalCallBack
        )
        return False

    # In vertical Map fusion we reject all fusing where `c` is an intermediate.
    #  It is important here, that `pmap3` and `pmap4` can be vertically fused into
    #  Map `smap2`.
    def vertical_map_fusion_hook_that_ensures_that_transient_c_survives(
        this: gtx_transformations.MapFusionHorizontal,
        first_map_exit: dace_nodes.MapExit,
        second_map_entry: dace_nodes.MapEntry,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> bool:
        # NOTE: This check runs before the transformation, i.e. `this`, has made sure
        #   that the transformation can apply in the first place.
        _increase_count(
            gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowMapFusionVerticalCallBack
        )
        if any(
            oedge.dst.data == "c"
            for oedge in graph.out_edges(first_map_exit)
            if isinstance(oedge.dst, dace_nodes.AccessNode)
        ):
            return False
        return True

    # This hook is called before the top level dataflow optimization starts, thus
    #  it should still have all Maps.
    def pre_top_level_data_flow_optimization_stage(sdfg: dace.SDFG) -> None:
        assert sdfg.number_of_nodes() == 1
        assert len(list(sdfg.nodes())) == 1
        assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 4

        state = next(iter(sdfg.states()))
        access_nodes = util.count_nodes(state, dace_nodes.AccessNode, True)
        assert len(access_nodes) == 5
        assert {ac.data for ac in access_nodes} == set("abcde")
        _increase_count(gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowPre)

    # This hook is called during the optimization loop of the top level data flow
    #  optimization. There is nothing useful we can do with it now, thus we only
    #  do some checks.
    def top_level_data_flow_optimization_stage_step(sdfg: dace.SDFG) -> None:
        assert any(ac.data == "c" for ac in util.count_nodes(sdfg, dace_nodes.AccessNode, True))
        _increase_count(gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowStep)

    # This hook is called at the end of the top level dataflow optimization stage.
    #  We will use this to perform some checks.
    def post_top_level_data_flow_optimization_stage(sdfg: dace.SDFG) -> None:
        assert sdfg.number_of_nodes() == 1
        assert len(list(sdfg.nodes())) == 1
        assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 3

        state = next(iter(sdfg.states()))
        access_nodes = util.count_nodes(state, dace_nodes.AccessNode, True)
        top_level_data = {ac.data for ac in access_nodes if state.scope_dict()[ac] is None}
        assert top_level_data == set("acde")

        remaining_map_entries = util.count_nodes(state, dace_nodes.MapEntry, True)
        assert len(remaining_map_entries) == 3

        found_pmap3 = False
        found_pmap4 = False
        found_smap = False
        for map_entry in remaining_map_entries:
            if map_entry.map.label.startswith("pmap3"):
                assert not found_pmap3
                found_pmap3 = True
            if map_entry.map.label.startswith("pmap4"):
                assert not found_pmap4
                found_pmap4 = True
            if map_entry.map.label.startswith("smap"):
                assert not found_smap
                found_smap = True

        assert found_pmap3 and found_pmap4 and found_smap
        _increase_count(gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowPost)

    callbacks = {
        gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowMapFusionHorizontalCallBack: horizontal_map_fusion_hook_that_rejects_all_atteps,
        gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowMapFusionVerticalCallBack: vertical_map_fusion_hook_that_ensures_that_transient_c_survives,
        gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowPre: pre_top_level_data_flow_optimization_stage,
        gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowStep: top_level_data_flow_optimization_stage_step,
        gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowPost: post_top_level_data_flow_optimization_stage,
    }

    # Apply the auto optimization, with the hooks.
    #  Setting `disable_splitting` to `False` is used to see if there is an uncontrolled
    #  MapFusion somewhere.
    gtx_transformations.gt_auto_optimize(
        sdfg,
        gpu=False,
        optimization_hooks=callbacks,
        disable_splitting=disable_splitting,
    )

    # Note that some results are not very predictable, hence `> 0`.
    assert (
        call_count[gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowMapFusionHorizontalCallBack]
        > 0
    )
    assert (
        call_count[gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowMapFusionVerticalCallBack]
        > 0
    )
    assert call_count[gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowPre] == 1
    assert call_count[gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowStep] > 0
    assert call_count[gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowPost] == 1

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref, res)
