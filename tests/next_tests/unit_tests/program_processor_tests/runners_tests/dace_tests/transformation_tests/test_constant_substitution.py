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

from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)

from . import util


import dace


def _make_nested_sdfg_test_constant_sub() -> dace.SDFG:
    sdfg = dace.SDFG(util.unique_name("nested"))
    N = dace.symbol(sdfg.add_symbol("N", dace.int32))
    One = dace.symbol(sdfg.add_symbol("One", dace.int32))
    for name in "ABC":
        sdfg.add_array(
            name=name,
            dtype=dace.float64,
            shape=(N, N),
            strides=(N, One),
            transient=False,
        )
    state = sdfg.add_state(is_start_block=True)
    state.add_mapped_tasklet(
        "computation",
        map_ranges={"__i0": "0:N", "__i1": "0:N"},
        inputs={
            "__in0": dace.Memlet("A[__i0, __i1]"),
            "__in1": dace.Memlet("B[__i0, __i1]"),
        },
        code="__out = __in0 + __in1",
        outputs={"__out": dace.Memlet("C[__i0, __i1]")},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg


def _make_sdfg_test_constant_sub() -> tuple[dace.SDFG, dace.nodes.NestedSDFG]:
    sdfg = dace.SDFG(util.unique_name("outer_sdfg"))
    N = dace.symbol(sdfg.add_symbol("N", dace.int32))
    One = dace.symbol(sdfg.add_symbol("One", dace.int32))
    for name in "ABCD":
        sdfg.add_array(
            name=name,
            dtype=dace.float64,
            shape=(N, N),
            strides=(N, One),
            transient=False,
        )
    sdfg.arrays["C"].transient = True

    first_state: dace.SDFGState = sdfg.add_state(is_start_block=True)
    nested_sdfg: dace.SDFG = _make_nested_sdfg_test_constant_sub()
    nsdfg = first_state.add_nested_sdfg(
        nested_sdfg,
        parent=sdfg,
        inputs={"A", "B"},
        outputs={"C"},
        symbol_mapping={"One": "One", "N": "N"},
    )
    first_state.add_edge(
        first_state.add_access("A"),
        None,
        nsdfg,
        "A",
        dace.Memlet("A[0:N, 0:N]"),
    )
    first_state.add_edge(
        first_state.add_access("B"),
        None,
        nsdfg,
        "B",
        dace.Memlet("B[0:N, 0:N]"),
    )
    first_state.add_edge(
        nsdfg,
        "C",
        first_state.add_access("C"),
        None,
        dace.Memlet("C[0:N, 0:N]"),
    )

    second_state: dace.SDFGState = sdfg.add_state_after(first_state)
    second_state.add_mapped_tasklet(
        "outer_computation",
        map_ranges={"__i0": "0:N", "__i1": "0:N"},
        inputs={
            "__in0": dace.Memlet("A[__i0, __i1]"),
            "__in1": dace.Memlet("C[__i0, __i1]"),
        },
        code="__out = __in0 * __in1",
        outputs={"__out": dace.Memlet("D[__i0, __i1]")},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg, nsdfg


def test_constant_substitution():
    sdfg, nsdfg = _make_sdfg_test_constant_sub()

    # Ensure that `One` is present.
    assert len(sdfg.symbols) == 2
    assert len(nsdfg.sdfg.symbols) == 2
    assert len(nsdfg.symbol_mapping) == 2
    assert "One" in sdfg.symbols
    assert "One" in nsdfg.sdfg.symbols
    assert "One" in nsdfg.symbol_mapping
    assert "One" == str(nsdfg.symbol_mapping["One"])
    assert all(str(desc.strides[1]) == "One" for desc in sdfg.arrays.values())
    assert all(str(desc.strides[1]) == "One" for desc in nsdfg.sdfg.arrays.values())
    assert all(str(desc.strides[0]) == "N" for desc in sdfg.arrays.values())
    assert all(str(desc.strides[0]) == "N" for desc in nsdfg.sdfg.arrays.values())
    assert "One" in sdfg.used_symbols(True)

    # Now replace `One` with 1
    gtx_transformations.gt_substitute_compiletime_symbols(sdfg, {"One": 1, "N": 10})

    assert len(sdfg.symbols) == 1
    assert len(nsdfg.sdfg.symbols) == 1
    assert len(nsdfg.symbol_mapping) == 1
    assert "One" not in sdfg.symbols
    assert "One" not in nsdfg.sdfg.symbols
    assert "One" not in nsdfg.symbol_mapping
    assert all(desc.strides[1] == 1 and len(desc.strides) == 2 for desc in sdfg.arrays.values())
    assert all(
        desc.strides[1] == 1 and len(desc.strides) == 2 for desc in nsdfg.sdfg.arrays.values()
    )
    assert all(str(desc.strides[0]) == "N" for desc in sdfg.arrays.values())
    assert all(str(desc.strides[0]) == "N" for desc in nsdfg.sdfg.arrays.values())
    assert "One" not in sdfg.used_symbols(True)


def _make_not_wrapped_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG(util.unique_name("not_wrapped_sdfg"))
    state = sdfg.add_state(is_start_block=True)

    sdfg.add_symbol("N", dace.int64)
    sdfg.add_symbol("lim_area", dace.bool_)

    for name in "AB":
        sdfg.add_array(name, shape=("N",), dtype=dace.float64, transient=False)

    state.add_mapped_tasklet(
        "PreComp",
        map_ranges={"__i": "0:N"},
        inputs={"__in": dace.Memlet("A[__i]")},
        code="__out = __in + N",
        outputs={"__out": dace.Memlet("B[__i]")},
        external_edges=True,
    )

    stateT = sdfg.add_state(is_start_block=False)
    stateT.add_mapped_tasklet(
        "Tcomp",
        map_ranges={"__i": "0:N"},
        inputs={"__in": dace.Memlet("A[__i]")},
        code="__out = __in + N",
        outputs={"__out": dace.Memlet("B[__i]")},
        external_edges=True,
    )

    stateF = sdfg.add_state(is_start_block=False)
    stateF.add_mapped_tasklet(
        "Fcomp",
        map_ranges={"__i": "0:N"},
        inputs={"__in": dace.Memlet("A[__i]")},
        code="__out = __in + 2 * N",
        outputs={"__out": dace.Memlet("B[__i]")},
        external_edges=True,
    )

    stateJ = sdfg.add_state(is_start_block=False)
    sdfg.add_edge(state, stateT, dace.InterstateEdge(condition="lim_area"))
    sdfg.add_edge(state, stateF, dace.InterstateEdge(condition="not lim_area"))
    sdfg.add_edge(stateT, stateJ, dace.InterstateEdge())
    sdfg.add_edge(stateF, stateJ, dace.InterstateEdge())
    sdfg.validate()
    return sdfg


def _make_wrapped_sdfg() -> tuple[dace.SDFG, dace_nodes.NestedSDFG]:
    sdfg = dace.SDFG(util.unique_name("wrapped_sdfg"))
    state = sdfg.add_state("WRAPPED", is_start_block=True)
    sdfg.add_symbol("lim_area", dace.bool_)
    sdfg.add_symbol("N", dace.bool_)
    for name in "AB":
        sdfg.add_array(name, shape=("N",), dtype=dace.float64, transient=False)

    nsdfg = state.add_nested_sdfg(
        sdfg=_make_not_wrapped_sdfg(),
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


def test_constant_substitution_not_wrapped_sdfg():
    sdfg: dace.SDFG = _make_not_wrapped_sdfg()
    assert sdfg.number_of_nodes() > 1
    assert sdfg.free_symbols == {"N", "lim_area"}
    map_entries_old: list[dace_nodes.MapEntry] = util.count_nodes(
        sdfg,
        node_type=dace_nodes.MapEntry,
        return_nodes=True,
    )
    assert any(str(map_entry.map.range[0][1] + 1) == "N" for map_entry in map_entries_old)
    assert all(
        node.desc(sdfg).shape == ("N",)
        for node in sdfg.all_nodes_recursive()
        if isinstance(node, dace_nodes.AccessNode)
    )

    gtx_transformations.gt_substitute_compiletime_symbols(
        sdfg,
        repl={"N": 10, "lim_area": True},
        validate=True,
        validate_all=True,
    )
    assert sdfg.number_of_nodes() == 4
    assert len(sdfg.free_symbols) == 0
    map_entries: list[dace_nodes.MapEntry] = util.count_nodes(
        sdfg,
        node_type=dace_nodes.MapEntry,
        return_nodes=True,
    )
    assert len(map_entries) == 2
    assert any(str(map_entry.map.range[0][1]) == "9" for map_entry in map_entries)
    assert all(
        node.desc(sdfg).shape == (10,)
        for node in sdfg.all_nodes_recursive()
        if isinstance(node, dace_nodes.AccessNode)
    )


def test_constant_substitution_wrapped_sdfg():
    sdfg, nsdfg = _make_wrapped_sdfg()
    assert sdfg.number_of_nodes() == 1
    assert sdfg.free_symbols == {"N", "lim_area"}
    assert util.count_nodes(sdfg, dace_nodes.NestedSDFG) == 1

    map_entries_old: list[dace_nodes.MapEntry] = util.count_nodes(
        nsdfg.sdfg,
        dace_nodes.MapEntry,
        return_nodes=True,
    )
    assert any(str(map_entry.map.range[0][1] + 1) == "N" for map_entry in map_entries_old)
    assert all(
        node.desc(nsdfg.sdfg).shape == ("N",)
        for node in sdfg.all_nodes_recursive()
        if isinstance(node, dace_nodes.AccessNode)
    )

    sdfg.view()
    gtx_transformations.gt_substitute_compiletime_symbols(
        sdfg,
        repl={"N": "10", "lim_area": 1},
        validate=True,
        validate_all=True,
    )

    assert sdfg.number_of_nodes() == 1
    assert util.count_nodes(sdfg, dace_nodes.NestedSDFG) == 1
    assert len(sdfg.free_symbols) == 0
    for nsdfg in sdfg.all_sdfgs_recursive():
        assert all(
            node.desc(nsdfg).shape == (10,)
            for node in sdfg.all_nodes_recursive()
            if isinstance(node, dace_nodes.AccessNode)
        )

    map_entries: list[dace_nodes.MapEntry] = util.count_nodes(
        nsdfg.sdfg,
        dace_nodes.MapEntry,
        return_nodes=True,
    )
    assert len(map_entries) == 2
    assert any(str(map_entry.map.range[0][1]) == "9" for map_entry in map_entries)
