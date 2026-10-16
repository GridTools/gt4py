# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dace
import pytest
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations

from . import util


GATHER_CODE = "__out = __in1 + __in2"


def _tasklets_with_prefix(state: dace.SDFGState, prefix: str) -> list[dace_nodes.Tasklet]:
    """All tasklets of `state` whose label starts with `prefix`."""
    return [
        node
        for node in state.nodes()
        if isinstance(node, dace_nodes.Tasklet) and node.label.startswith(prefix)
    ]


def _data_read_by_tasklets(state: dace.SDFGState, prefix: str) -> dict[str, set[str]]:
    """The names of the arrays read by the tasklets with the given label prefix."""
    return {
        tasklet.label: {edge.data.data for edge in state.in_edges(tasklet)}
        for tasklet in _tasklets_with_prefix(state, prefix)
    }


def _make_simple_duplicate() -> tuple[dace.SDFG, dace.SDFGState]:
    """Two sibling Maps that compute the same thing into different transients.

    This resembles the pattern the GT4Py frontend generates for an identical
    sub expression that is used by two different consumers: after inlining
    the nested SDFGs, the duplicates are sibling Map scopes inside the same
    state, differing only in the names of the transient arrays they write.
    """
    sdfg = dace.SDFG(util.unique_name("redundant_computation"))
    state = sdfg.add_state(is_start_block=True)

    for name in ["a", "b", "out1", "out2"]:
        sdfg.add_array(name, shape=(10,), dtype=dace.float64, transient=False)
    for name in ["t1", "t2"]:
        sdfg.add_array(name, shape=(10,), dtype=dace.float64, transient=True)

    a = state.add_access("a")
    b = state.add_access("b")
    t1 = state.add_access("t1")
    t2 = state.add_access("t2")

    state.add_mapped_tasklet(
        "gather1",
        map_ranges={"__i": "0:10"},
        inputs={"__in1": dace.Memlet("a[__i]"), "__in2": dace.Memlet("b[__i]")},
        code=GATHER_CODE,
        outputs={"__out": dace.Memlet("t1[__i]")},
        input_nodes={a, b},
        output_nodes={t1},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "gather2",
        map_ranges={"__i": "0:10"},
        inputs={"__in1": dace.Memlet("a[__i]"), "__in2": dace.Memlet("b[__i]")},
        code=GATHER_CODE,
        outputs={"__out": dace.Memlet("t2[__i]")},
        input_nodes={a, b},
        output_nodes={t2},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "consumer1",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("t1[__i]")},
        code="__out = __in * 2.0",
        outputs={"__out": dace.Memlet("out1[__i]")},
        input_nodes={t1},
        output_nodes={state.add_access("out1")},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "consumer2",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("t2[__i]")},
        code="__out = __in + 5.0",
        outputs={"__out": dace.Memlet("out2[__i]")},
        input_nodes={t2},
        output_nodes={state.add_access("out2")},
        external_edges=True,
    )

    sdfg.validate()
    return sdfg, state


def _make_duplicate_chain() -> tuple[dace.SDFG, dace.SDFGState]:
    """Two chains of Maps that only become duplicates after the first merge.

    The `gather` Maps are duplicates of each other, while the `multiply` Maps
    only become duplicates after the outputs of the gathers have been unified.
    The transformation has to iterate to a fixpoint to remove both levels.
    """
    sdfg = dace.SDFG(util.unique_name("redundant_computation_chain"))
    state = sdfg.add_state(is_start_block=True)

    for name in ["a", "b", "out1", "out2"]:
        sdfg.add_array(name, shape=(10,), dtype=dace.float64, transient=False)
    for name in ["g1", "g2", "m1", "m2"]:
        sdfg.add_array(name, shape=(10,), dtype=dace.float64, transient=True)

    a = state.add_access("a")
    b = state.add_access("b")
    g1 = state.add_access("g1")
    g2 = state.add_access("g2")

    state.add_mapped_tasklet(
        "gather1",
        map_ranges={"__i": "0:10"},
        inputs={"__in1": dace.Memlet("a[__i]"), "__in2": dace.Memlet("b[__i]")},
        code=GATHER_CODE,
        outputs={"__out": dace.Memlet("g1[__i]")},
        input_nodes={a, b},
        output_nodes={g1},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "multiply1",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("g1[__i]")},
        code="__out = __in * 3.0",
        outputs={"__out": dace.Memlet("m1[__i]")},
        input_nodes={g1},
        output_nodes={state.add_access("m1")},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "gather2",
        map_ranges={"__i": "0:10"},
        inputs={"__in1": dace.Memlet("a[__i]"), "__in2": dace.Memlet("b[__i]")},
        code=GATHER_CODE,
        outputs={"__out": dace.Memlet("g2[__i]")},
        input_nodes={a, b},
        output_nodes={g2},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "multiply2",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("g2[__i]")},
        code="__out = __in * 3.0",
        outputs={"__out": dace.Memlet("m2[__i]")},
        input_nodes={g2},
        output_nodes={state.add_access("m2")},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "consumer1",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("m1[__i]")},
        code="__out = __in * 2.0",
        outputs={"__out": dace.Memlet("out1[__i]")},
        input_nodes={state.add_access("m1")},
        output_nodes={state.add_access("out1")},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "consumer2",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("m2[__i]")},
        code="__out = __in + 5.0",
        outputs={"__out": dace.Memlet("out2[__i]")},
        input_nodes={state.add_access("m2")},
        output_nodes={state.add_access("out2")},
        external_edges=True,
    )

    sdfg.validate()
    return sdfg, state


def _make_no_elimination_variant(variant: str) -> tuple[dace.SDFG, dace.SDFGState]:
    """A `gather2` Map that must not be merged with `gather1`.

    The `variant` determines why the two scopes are not equivalent:
    - `"different_tasklet_code"`: the tasklets compute different things.
    - `"different_input_data"`: the scopes read different input arrays.
    - `"different_map_ranges"`: the scopes iterate over different ranges.
    - `"non_transient_output"`: `gather2` writes to a global array.
    - `"two_output_arrays"`: `gather2` writes to two arrays.
    - `"second_writer"`: the output of `gather2` is overwritten by another scope.
    """
    sdfg = dace.SDFG(util.unique_name(f"redundant_computation_{variant}"))
    state = sdfg.add_state(is_start_block=True)

    global_names = ["a", "b", "c", "out1", "out2"]
    transient_names = ["t1", "t2"]
    if variant == "two_output_arrays":
        transient_names.append("t3")

    for name in global_names:
        sdfg.add_array(name, shape=(10,), dtype=dace.float64, transient=False)
    for name in transient_names:
        sdfg.add_array(name, shape=(10,), dtype=dace.float64, transient=True)

    a = state.add_access("a")
    b = state.add_access("b")
    t1 = state.add_access("t1")

    gather2_inputs = {"__in1": dace.Memlet("a[__i]"), "__in2": dace.Memlet("b[__i]")}
    gather2_code = GATHER_CODE
    gather2_range = "0:10"
    gather2_outputs = {"__out": dace.Memlet("t2[__i]")}
    has_consumer2 = True

    if variant == "different_tasklet_code":
        gather2_code = "__out = __in1 - __in2"
    elif variant == "different_input_data":
        gather2_inputs = {"__in1": dace.Memlet("a[__i]"), "__in2": dace.Memlet("c[__i]")}
    elif variant == "different_map_ranges":
        gather2_range = "0:5"
    elif variant == "non_transient_output":
        gather2_outputs = {"__out": dace.Memlet("out2[__i]")}
        has_consumer2 = False
    elif variant == "two_output_arrays":
        gather2_outputs = {"__out": dace.Memlet("t2[__i]"), "__out2": dace.Memlet("t3[__i]")}
        gather2_code = f"{GATHER_CODE}\n__out2 = __in1 + __in2"

    state.add_mapped_tasklet(
        "gather1",
        map_ranges={"__i": "0:10"},
        inputs={"__in1": dace.Memlet("a[__i]"), "__in2": dace.Memlet("b[__i]")},
        code=GATHER_CODE,
        outputs={"__out": dace.Memlet("t1[__i]")},
        input_nodes={a, b},
        output_nodes={t1},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "consumer1",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("t1[__i]")},
        code="__out = __in * 2.0",
        outputs={"__out": dace.Memlet("out1[__i]")},
        input_nodes={t1},
        output_nodes={state.add_access("out1")},
        external_edges=True,
    )

    if variant == "second_writer":
        # `gather2` writes `t2` and a second scope overwrites it afterwards,
        # such that no scope exclusively owns the output array.
        t2_first_write = state.add_access("t2")
        t2_second_write = state.add_access("t2")
        _, _, gather2_exit = state.add_mapped_tasklet(
            "gather2",
            map_ranges={"__i": "0:10"},
            inputs={"__in1": dace.Memlet("a[__i]"), "__in2": dace.Memlet("b[__i]")},
            code=GATHER_CODE,
            outputs={"__out": dace.Memlet("t2[__i]")},
            input_nodes={a, b},
            output_nodes={t2_first_write},
            external_edges=True,
        )
        _, second_writer_entry, _ = state.add_mapped_tasklet(
            "second_writer",
            map_ranges={"__i": "0:10"},
            inputs={"__in": dace.Memlet("a[__i]")},
            code="__out = __in * 7.0",
            outputs={"__out": dace.Memlet("t2[__i]")},
            input_nodes={a},
            output_nodes={t2_second_write},
            external_edges=True,
        )
        state.add_nedge(gather2_exit, second_writer_entry, dace.Memlet())
        state.add_mapped_tasklet(
            "consumer2",
            map_ranges={"__i": "0:10"},
            inputs={"__in": dace.Memlet("t2[__i]")},
            code="__out = __in + 5.0",
            outputs={"__out": dace.Memlet("out2[__i]")},
            input_nodes={t2_second_write},
            output_nodes={state.add_access("out2")},
            external_edges=True,
        )
    else:
        if variant == "two_output_arrays":
            output_node_names = ["t2", "t3"]
        elif variant == "non_transient_output":
            output_node_names = ["out2"]
        else:
            output_node_names = ["t2"]
        gather2_input_nodes = {a, b}
        if variant == "different_input_data":
            gather2_input_nodes = {a, state.add_access("c")}
        state.add_mapped_tasklet(
            "gather2",
            map_ranges={"__i": gather2_range},
            inputs=gather2_inputs,
            code=gather2_code,
            outputs=gather2_outputs,
            input_nodes=gather2_input_nodes,
            output_nodes={state.add_access(name) for name in output_node_names},
            external_edges=True,
        )
        if has_consumer2:
            state.add_mapped_tasklet(
                "consumer2",
                map_ranges={"__i": gather2_range},
                inputs={"__in": dace.Memlet("t2[__i]")},
                code="__out = __in + 5.0",
                outputs={"__out": dace.Memlet("out2[__i]")},
                input_nodes={state.add_access("t2")},
                output_nodes={state.add_access("out2")},
                external_edges=True,
            )

    sdfg.validate()
    return sdfg, state


def test_eliminate_duplicate():
    sdfg, state = _make_simple_duplicate()

    assert len(_tasklets_with_prefix(state, "gather")) == 2
    assert {"t1", "t2"} <= set(sdfg.arrays.keys())

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    nb_removed = gtx_transformations.gt_eliminate_redundant_computations(sdfg=sdfg)

    assert nb_removed == 1
    # The duplicate gather scope was removed and only the canonical transient
    #  remains, both consumers read from it.
    assert len(_tasklets_with_prefix(state, "gather")) == 1
    assert set(sdfg.arrays.keys()) == {"a", "b", "out1", "out2", "t1"}
    assert _data_read_by_tasklets(state, "consumer") == {
        "consumer1": {"t1"},
        "consumer2": {"t1"},
    }

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)
    sdfg.validate()


def test_eliminate_duplicate_chain():
    sdfg, state = _make_duplicate_chain()

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    nb_removed = gtx_transformations.gt_eliminate_redundant_computations(sdfg=sdfg)

    # Both levels of the chain, the gathers and the multiplies, were removed.
    assert nb_removed == 2
    assert len(_tasklets_with_prefix(state, "gather")) == 1
    assert len(_tasklets_with_prefix(state, "multiply")) == 1
    assert set(sdfg.arrays.keys()) == {"a", "b", "out1", "out2", "g1", "m1"}
    assert _data_read_by_tasklets(state, "consumer") == {
        "consumer1": {"m1"},
        "consumer2": {"m1"},
    }

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)
    sdfg.validate()


@pytest.mark.parametrize(
    "variant",
    [
        "different_tasklet_code",
        "different_input_data",
        "different_map_ranges",
        "non_transient_output",
        "two_output_arrays",
        "second_writer",
    ],
)
def test_no_elimination(variant: str):
    sdfg, state = _make_no_elimination_variant(variant)

    nb_tasklets_before = util.count_nodes(state, dace_nodes.Tasklet)
    arrays_before = set(sdfg.arrays.keys())

    nb_removed = gtx_transformations.gt_eliminate_redundant_computations(sdfg=sdfg)

    assert nb_removed == 0
    assert util.count_nodes(state, dace_nodes.Tasklet) == nb_tasklets_before
    assert set(sdfg.arrays.keys()) == arrays_before
    sdfg.validate()


def test_gt_simplify_runs_elimination():
    sdfg, state = _make_simple_duplicate()

    result = gtx_transformations.gt_simplify(sdfg=sdfg, validate=True)

    assert result is not None
    assert result["RedundantComputationElimination"] == 1
    assert len(_tasklets_with_prefix(state, "gather")) == 1
    sdfg.validate()


def test_gt_simplify_skip_elimination():
    sdfg, state = _make_simple_duplicate()

    skip = gtx_transformations.constants.GT_SIMPLIFY_DEFAULT_SKIP_SET | {
        "RedundantComputationElimination"
    }
    result = gtx_transformations.gt_simplify(sdfg=sdfg, validate=True, skip=skip)

    assert "RedundantComputationElimination" not in (result or {})
    assert len(_tasklets_with_prefix(state, "gather")) == 2
    sdfg.validate()
