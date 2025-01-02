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


def test_constant_substitution():
    sdfg, nsdfg = _make_sdfg()

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
    gtx_transformations.gt_substitute_compiletime_symbols(sdfg, {"One": 1})

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


def _make_nested_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG("nested")
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


def _make_sdfg() -> tuple[dace.SDFG, dace.nodes.NestedSDFG]:
    sdfg = dace.SDFG("outer_sdfg")
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
    nested_sdfg: dace.SDFG = _make_nested_sdfg()
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
