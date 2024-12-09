# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from typing import Any, Final, Iterable, Optional, TypeAlias, Union, Literal

dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)

from . import util


def _check_shapes(
    sdfg: dace.SDFG,
    expected_shape: tuple[str, ...],
    to_string: bool = True,
) -> bool:
    return all(
        tuple((str(s) if to_string else s) for s in desc.shape) == expected_shape
        for desc in sdfg.arrays.values()
    )


def _check_maps(
    sdfg: dace.SDFG,
    expected_end: str,
) -> bool:
    map_entries: list[dace_nodes.MapEntry] = util.count_nodes(
        graph=sdfg,
        node_type=dace_nodes.MapEntry,
        return_nodes=True,
    )
    return all(
        str(map_entry.map.range.ranges[0][1] + 1) == expected_end for map_entry in map_entries
    )


def _check_tasklets(
    sdfg: dace.SDFG,
    expected_symbols: Optional[set[str]] = None,
    forbidden_symbols: Optional[set[str]] = None,
) -> bool:
    assert not ((expected_symbols is None) and (forbidden_symbols is None))
    expected_symbols = expected_symbols or set()
    forbidden_symbols = forbidden_symbols or set()

    tasklets: list[dace_nodes.Tasklet] = util.count_nodes(
        graph=sdfg,
        node_type=dace_nodes.Tasklet,
        return_nodes=True,
    )
    if not all(expected_symbols.issubset(tasklet.free_symbols) for tasklet in tasklets):
        return False
    if not all(forbidden_symbols.isdisjoint(tasklet.free_symbols) for tasklet in tasklets):
        return False
    return True


def make_multi_state_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG(util.unique_name("multi_state_sdfg"))
    state = sdfg.add_state("stateS", is_start_block=True)
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_symbol("lim_area", dace.bool_)
    for name in "AB":
        sdfg.add_array(name, shape=("N",), dtype=dace.float64, transient=False)

    stateT = sdfg.add_state("stateT", is_start_block=False)
    stateT.add_mapped_tasklet(
        "Tcomp",
        map_ranges={"__i": "0:N"},
        inputs={"__in": dace.Memlet("A[__i]")},
        code="__out = (__in +  2 * N) if lim_area else (__in - 3 * N)",
        outputs={"__out": dace.Memlet("B[__i]")},
        external_edges=True,
    )

    stateF = sdfg.add_state("stateF", is_start_block=False)
    stateF.add_mapped_tasklet(
        "Fcomp",
        map_ranges={"__i": "0:N"},
        inputs={"__in": dace.Memlet("A[__i]")},
        code="__out = (__in +  3 * N) if lim_area else (__in - 4 * N)",
        outputs={"__out": dace.Memlet("B[__i]")},
        external_edges=True,
    )

    stateJ = sdfg.add_state("stateJ", is_start_block=False)
    sdfg.add_edge(state, stateT, dace.InterstateEdge(condition="lim_area"))
    sdfg.add_edge(state, stateF, dace.InterstateEdge(condition="not lim_area"))
    sdfg.add_edge(stateT, stateJ, dace.InterstateEdge())
    sdfg.add_edge(stateF, stateJ, dace.InterstateEdge())
    sdfg.validate()
    return sdfg


def make_single_state_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG(util.unique_name("single_state_sdfg"))
    state = sdfg.add_state(is_start_block=True)
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_symbol("lim_area", dace.bool_)
    for name in "AB":
        sdfg.add_array(name, shape=("N",), dtype=dace.float64, transient=False)

    state.add_mapped_tasklet(
        "PreComp",
        map_ranges={"__i": "0:N"},
        inputs={"__in": dace.Memlet("A[__i]")},
        code="__out = (__in + N) if lim_area else (__in - N)",
        outputs={"__out": dace.Memlet("B[__i]")},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg


def make_single_state_with_two_maps_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG(util.unique_name("single_state_sdfg"))
    state = sdfg.add_state(is_start_block=True)
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_symbol("lim_area", dace.bool_)
    for name in "ABT":
        sdfg.add_array(name, shape=("N",), dtype=dace.float64, transient=False)
    sdfg.arrays["T"].transient = True

    T = state.add_access("T")

    state.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i": "0:N"},
        inputs={"__in": dace.Memlet("A[__i]")},
        code="__out = (__in + N) if lim_area else (__in - N)",
        outputs={"__out": dace.Memlet("T[__i]")},
        output_nodes={T},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "comp2",
        map_ranges={"__i": "0:N"},
        inputs={"__in": dace.Memlet("T[__i]")},
        code="__out = (__in + 7 * N) if lim_area else (__in - 4 * N)",
        outputs={"__out": dace.Memlet("B[__i]")},
        input_nodes={T},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg


def make_wrapped_sdfg(
    single_state: bool,
) -> tuple[dace.SDFG, dace_nodes.NestedSDFG]:
    sdfg = dace.SDFG(util.unique_name("wrapped_sdfg"))
    state = sdfg.add_state("wrap_state", is_start_block=True)
    sdfg.add_symbol("lim_area", dace.bool_)
    sdfg.add_symbol("N", dace.int64)
    for name in "AB":
        sdfg.add_array(name, shape=("N",), dtype=dace.float64, transient=False)

    inner_sdfg = make_single_state_sdfg() if single_state else make_multi_state_sdfg()
    nsdfg = state.add_nested_sdfg(
        sdfg=inner_sdfg,
        parent=sdfg,
        inputs={"A"},
        outputs={"B"},
        symbol_mapping={"lim_area": "lim_area", "N": "N"},
    )
    state.add_edge(
        state.add_access("A"), None, nsdfg, "A", dace.Memlet.from_array("A", sdfg.arrays["A"])
    )
    state.add_edge(
        nsdfg, "B", state.add_access("B"), None, dace.Memlet.from_array("B", sdfg.arrays["B"])
    )
    sdfg.validate()
    return sdfg, nsdfg


def test_nested_sdfg_with_single_state():
    sdfg, nested_sdfg = make_wrapped_sdfg(single_state=True)
    assert _check_shapes(sdfg, ("N",))
    assert _check_shapes(nested_sdfg.sdfg, ("N",))
    assert _check_maps(nested_sdfg.sdfg, "N")
    assert _check_tasklets(nested_sdfg.sdfg, expected_symbols={"N", "lim_area"})

    repl = {"N": 10, "lim_area": True}
    gtx_transformations.gt_substitute_compiletime_symbols(sdfg, repl)

    assert _check_shapes(sdfg, (10,), to_string=False)
    assert _check_shapes(nested_sdfg.sdfg, ("10",))
    assert _check_maps(nested_sdfg.sdfg, "10")
    assert _check_tasklets(nested_sdfg.sdfg, forbidden_symbols={"N", "lim_area"})
    assert len(nested_sdfg.symbol_mapping) == 0


def test_nested_sdfg_with_multiple_states():
    sdfg, nested_sdfg = make_wrapped_sdfg(single_state=False)
    assert _check_shapes(sdfg, ("N",))
    assert _check_shapes(nested_sdfg.sdfg, ("N",))
    assert _check_maps(nested_sdfg.sdfg, "N")
    assert _check_tasklets(nested_sdfg.sdfg, expected_symbols={"N", "lim_area"})

    repl = {"N": 10, "lim_area": True}
    gtx_transformations.gt_substitute_compiletime_symbols(sdfg, repl)

    # Due to a bug in DaCe, see `gtx_transformations.gt_substitute_compiletime_symbols()`
    #  we can not inspect the nested SDFG, since the function has to call simplify.
    #  For that reason we currently check if the nested SDFG was inlineed and the
    #  whole thing has collapsed to a single state with a map.
    # TODO(phimuell): Reactivate after the bug has been fixed.
    # assert _check_shapes(nested_sdfg.sdfg, ("10",))
    # assert _check_maps(nested_sdfg.sdfg, "10")
    # assert _check_tasklets(nested_sdfg.sdfg, forbidden_symbols={"N", "lim_area"})
    # assert len(nested_sdfg.symbol_mapping) == 0
    # assert _check_shapes(sdfg, (10,), to_string=False)

    assert sdfg.number_of_nodes() == 1
    assert util.count_nodes(sdfg, node_type=dace_nodes.NestedSDFG) == 0
    assert _check_shapes(sdfg, ("10",))
    assert _check_maps(sdfg, "10")
    assert _check_tasklets(sdfg, forbidden_symbols={"N", "lim_area"})


def test_single_state_top_sdfg():
    # This test works because everything is inside a single state.
    sdfg = make_single_state_sdfg()
    assert sdfg.number_of_nodes() == 1

    assert _check_maps(sdfg, "N")
    assert _check_shapes(sdfg, ("N",))
    assert _check_tasklets(sdfg, expected_symbols={"N", "lim_area"})

    repl = {"N": 10, "lim_area": True}
    gtx_transformations.gt_substitute_compiletime_symbols(sdfg, repl)

    assert _check_maps(sdfg, "10")
    assert _check_shapes(sdfg, (10,), to_string=False)
    assert _check_tasklets(sdfg, forbidden_symbols={"N", "lim_area"})


def test_single_state_with_two_maps():
    # This test works because everything is inside a single state.
    sdfg = make_single_state_with_two_maps_sdfg()
    assert sdfg.number_of_nodes() == 1

    assert _check_maps(sdfg, "N")
    assert _check_shapes(sdfg, ("N",))
    assert _check_tasklets(sdfg, expected_symbols={"N", "lim_area"})

    repl = {"N": 10, "lim_area": True}
    gtx_transformations.gt_substitute_compiletime_symbols(sdfg, repl)

    assert _check_maps(sdfg, "10")
    assert _check_shapes(sdfg, (10,), to_string=False)
    assert _check_tasklets(sdfg, forbidden_symbols={"N", "lim_area"})


def test_multi_state_top_sdfg():
    sdfg = make_multi_state_sdfg()
    assert sdfg.number_of_nodes() == 4

    start_state: dace.SDFGState = sdfg.start_state
    assert start_state.label == "stateS"
    assert all("lim_area" in edge.data.free_symbols for edge in sdfg.out_edges(start_state))

    assert _check_maps(sdfg, "N")
    assert _check_shapes(sdfg, ("N",))
    assert _check_tasklets(sdfg, expected_symbols={"N", "lim_area"})

    repl = {"N": 10, "lim_area": True}
    gtx_transformations.gt_substitute_compiletime_symbols(sdfg, repl)

    assert _check_maps(sdfg, "10")
    assert _check_shapes(sdfg, (10,), to_string=False)
    assert _check_tasklets(sdfg, forbidden_symbols={"N", "lim_area"})

    # Because of the bug in DaCe, see `gtx_transformations.gt_substitute_compiletime_symbols()`
    #  we can not inspect the condition on the edges, because simplify has been called.
    #  Thus for the time being we will just test if we are left with one state instead.
    # TODO(phimuell): reactivate once the bug has been solved.
    # assert not any("lim_area" in edge.data.free_symbols for edge in sdfg.out_edges(start_state))
    assert sdfg.number_of_nodes() == 1


def test_single_state_nested_with_top_map():
    sdfg, nested_sdfg = make_wrapped_sdfg(single_state=True)
    assert sdfg.number_of_nodes() == 1
    state: dace.SDFGState = list(sdfg.states())[0]

    sdfg.add_datadesc("new_input", sdfg.arrays["A"].clone())
    sdfg.arrays["A"].transient = True
    A: dace_nodes.AccessNode = next(
        iter(dnode for dnode in state.data_nodes() if dnode.data == "A")
    )
    state.add_mapped_tasklet(
        "compOutside",
        map_ranges={"__i": "0:N"},
        inputs={"__in": dace.Memlet("new_input[__i]")},
        code="__out = (__in + 10 * N) if lim_area else (__in - 14 * N)",
        outputs={"__out": dace.Memlet("A[__i]")},
        output_nodes={A},
        external_edges=True,
    )
    sdfg.validate()

    assert _check_maps(sdfg, "N")
    assert _check_shapes(sdfg, ("N",))
    assert _check_tasklets(sdfg, expected_symbols={"N", "lim_area"})
    assert _check_shapes(nested_sdfg.sdfg, ("N",))
    assert _check_maps(nested_sdfg.sdfg, "N")
    assert _check_tasklets(nested_sdfg.sdfg, expected_symbols={"N", "lim_area"})

    repl = {"N": 10, "lim_area": True}
    gtx_transformations.gt_substitute_compiletime_symbols(sdfg, repl)

    assert _check_maps(sdfg, "10")
    assert _check_shapes(sdfg, (10,), to_string=False)
    assert _check_tasklets(sdfg, forbidden_symbols={"N", "lim_area"})
    assert _check_shapes(nested_sdfg.sdfg, ("10",))
    assert _check_maps(nested_sdfg.sdfg, "10")
    assert _check_tasklets(nested_sdfg.sdfg, forbidden_symbols={"N", "lim_area"})
    assert len(nested_sdfg.symbol_mapping) == 0


@pytest.mark.xfail(reason="AccessNode replacement can not be done yet.")
def test_replace_access_node():
    sdfg = dace.SDFG(util.unique_name("replaced_access_node"))
    state = sdfg.add_state(is_start_block=True)
    sdfg.add_symbol("N", dace.int64)
    for name in "AB":
        sdfg.add_array(name, shape=("N",), dtype=dace.float64, transient=False)
    sdfg.add_scalar("S", dtype=dace.float64, transient=False)

    tsklt, me, mx = state.add_mapped_tasklet(
        "computation",
        map_ranges={"__i0": "0:N"},
        inputs={
            "__in1": dace.Memlet("A[__i0]"),
            "__in2": dace.Memlet("S[0]"),
        },
        code="__out = __in1 + __in2",
        outputs={"__out": dace.Memlet("B[__i0]")},
        external_edges=True,
    )
    sdfg.validate()

    repl = {"N": 10, "S": 10}
    gtx_transformations.gt_substitute_compiletime_symbols(sdfg, repl)

    assert len(list(dnode for dnode in state.data_nodes() if dnode.data == "S")) == 0
    assert _check_maps(sdfg, "10")
    assert _check_shapes(sdfg, (10,), to_string=False)
    assert _check_tasklets(sdfg, forbidden_symbols={"N", "lim_area"})
