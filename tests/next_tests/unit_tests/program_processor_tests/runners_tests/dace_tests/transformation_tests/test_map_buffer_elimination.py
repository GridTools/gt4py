# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np
import copy

dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util

import dace


def _make_test_sdfg(
    output_name: str = "G",
    input_name: str = "G",
    tmp_name: str = "T",
    array_size: int | str = 10,
    tmp_size: int | str | None = None,
    map_range: tuple[int | str, int | str] | None = None,
    tmp_to_glob_memlet: str | None = None,
    in_offset: str | None = None,
    out_offset: str | None = None,
) -> dace.SDFG:
    if isinstance(array_size, str):
        array_size = sdfg.add_symbol(array_size, dace.int32, find_new_name=True)
    if tmp_size is None:
        tmp_size = array_size
    if map_range is None:
        map_range = (0, array_size)
    if tmp_to_glob_memlet is None:
        tmp_to_glob_memlet = f"{tmp_name}[0:{array_size}] -> [0:{array_size}]"
    elif tmp_to_glob_memlet[0] == "[":
        tmp_to_glob_memlet = tmp_name + tmp_to_glob_memlet
    if in_offset is None:
        in_offset = "0"
    if out_offset is None:
        out_offset = in_offset

    sdfg = dace.SDFG(util.unique_name("map_buffer"))
    state = sdfg.add_state(is_start_block=True)
    names = {input_name, tmp_name, output_name}
    for name in names:
        sdfg.add_array(
            name,
            shape=((array_size,) if name != tmp_name else (tmp_size,)),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays[tmp_name].transient = True

    input_ac = state.add_access(input_name)
    tmp_ac = state.add_access(tmp_name)
    output_ac = state.add_access(output_name)

    state.add_mapped_tasklet(
        "computation",
        map_ranges={"__i0": f"{map_range[0]}:{map_range[1]}"},
        inputs={"__in1": dace.Memlet(data=input_ac.data, subset=f"__i0 + {in_offset}")},
        code="__out = __in1 + 10.0",
        outputs={"__out": dace.Memlet(data=tmp_ac.data, subset=f"__i0 + {out_offset}")},
        input_nodes={input_ac},
        output_nodes={tmp_ac},
        external_edges=True,
    )
    state.add_edge(
        tmp_ac,
        None,
        output_ac,
        None,
        dace.Memlet(tmp_to_glob_memlet),
    )
    sdfg.validate()
    return sdfg


def _perform_test(
    sdfg: dace.SDFG,
    xform: gtx_transformations.GT4PyMapBufferElimination,
    exp_count: int,
    array_size: int = 10,
) -> None:
    ref = {
        name: np.array(np.random.rand(array_size), dtype=np.float64, copy=True)
        for name, desc in sdfg.arrays.items()
        if not desc.transient
    }
    if "array_size" in sdfg.symbols:
        ref["array_size"] = array_size

    res = copy.deepcopy(ref)
    sdfg(**ref)

    count = sdfg.apply_transformations_repeated([xform], validate=True, validate_all=True)
    assert count == exp_count, f"Expected {exp_count} applications, but got {count}"

    if count == 0:
        return

    sdfg(**res)
    assert all(np.allclose(ref[name], res[name]) for name in ref.keys()), f"Failed for '{name}'."


def test_map_buffer_elimination_simple():
    sdfg = _make_test_sdfg()
    _perform_test(
        sdfg,
        gtx_transformations.GT4PyMapBufferElimination(assume_pointwise=True),
        exp_count=1,
    )


def test_map_buffer_elimination_simple_2():
    sdfg = _make_test_sdfg()
    _perform_test(
        sdfg,
        gtx_transformations.GT4PyMapBufferElimination(assume_pointwise=False),
        exp_count=0,
    )


def test_map_buffer_elimination_simple_3():
    sdfg = _make_test_sdfg(input_name="A", output_name="O")
    _perform_test(
        sdfg,
        gtx_transformations.GT4PyMapBufferElimination(assume_pointwise=False),
        exp_count=1,
    )


def test_map_buffer_elimination_offset_1():
    sdfg = _make_test_sdfg(
        map_range=(2, 8),
        tmp_to_glob_memlet="[2:8] -> [2:8]",
        input_name="A",
        output_name="O",
    )
    _perform_test(
        sdfg,
        gtx_transformations.GT4PyMapBufferElimination(assume_pointwise=False),
        exp_count=1,
    )


def test_map_buffer_elimination_offset_2():
    sdfg = _make_test_sdfg(
        map_range=(2, 8),
        in_offset="-2",
        out_offset="-2",
        tmp_to_glob_memlet="[0:6] -> [0:6]",
        input_name="A",
        output_name="O",
    )
    _perform_test(
        sdfg,
        gtx_transformations.GT4PyMapBufferElimination(assume_pointwise=False),
        exp_count=1,
    )


def test_map_buffer_elimination_offset_3():
    sdfg = _make_test_sdfg(
        map_range=(2, 8),
        in_offset="-2",
        out_offset="-2",
        tmp_to_glob_memlet="[0:6] -> [2:8]",
        input_name="A",
        output_name="O",
    )
    _perform_test(
        sdfg,
        gtx_transformations.GT4PyMapBufferElimination(assume_pointwise=False),
        exp_count=1,
    )


def test_map_buffer_elimination_offset_4():
    sdfg = _make_test_sdfg(
        map_range=(2, 8),
        in_offset="-2",
        out_offset="-2",
        tmp_to_glob_memlet="[1:7] -> [2:8]",
        input_name="A",
        output_name="O",
    )
    _perform_test(
        sdfg,
        gtx_transformations.GT4PyMapBufferElimination(assume_pointwise=False),
        exp_count=0,
    )


def test_map_buffer_elimination_offset_5():
    sdfg = _make_test_sdfg(
        map_range=(2, 8),
        tmp_size=6,
        in_offset="0",
        out_offset="-2",
        tmp_to_glob_memlet="[0:6] -> [2:8]",
        input_name="A",
        output_name="O",
    )
    _perform_test(
        sdfg,
        gtx_transformations.GT4PyMapBufferElimination(assume_pointwise=False),
        exp_count=1,
    )


def test_map_buffer_elimination_not_apply():
    """Indirect accessing, because of this the double buffer is needed."""
    sdfg = dace.SDFG(util.unique_name("map_buffer"))
    state = sdfg.add_state(is_start_block=True)

    names = ["A", "tmp", "idx"]
    for name in names:
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.int32 if name == "tmp" else dace.float64,
            transient=False,
        )
    sdfg.arrays["tmp"].transient = True

    tmp = state.add_access("tmp")
    state.add_mapped_tasklet(
        "indirect_accessing",
        map_ranges={"__i0": "0:10"},
        inputs={
            "__field": dace.Memlet("A[0:10]"),
            "__idx": dace.Memlet("idx[__i0]"),
        },
        code="__out = __field[__idx]",
        outputs={"__out": dace.Memlet("tmp[__i0]")},
        output_nodes={tmp},
        external_edges=True,
    )
    state.add_nedge(tmp, state.add_access("A"), dace.Memlet("tmp[0:10] -> [0:10]"))

    # TODO(phimuell): Update the transformation such that we can specify
    #       `assume_pointwise=True` and the test would still pass.
    count = sdfg.apply_transformations_repeated(
        gtx_transformations.GT4PyMapBufferElimination(
            assume_pointwise=False,
        ),
        validate=True,
        validate_all=True,
    )
    assert count == 0


def test_map_buffer_elimination_with_nested_sdfgs():
    """
    After removing a transient connected to a nested SDFG node, ensure that the strides
    are propagated to the arrays in nested SDFG.
    """

    stride1, stride2, stride3 = [dace.symbol(f"stride{i}", dace.int32) for i in range(3)]

    # top-level sdfg
    sdfg = dace.SDFG(util.unique_name("map_buffer"))
    inp, inp_desc = sdfg.add_array("__inp", (10,), dace.float64)
    out, out_desc = sdfg.add_array(
        "__out", (10, 10, 10), dace.float64, strides=(stride1, stride2, stride3)
    )
    tmp, _ = sdfg.add_temp_transient_like(out_desc)
    state = sdfg.add_state()
    tmp_node = state.add_access(tmp)

    nsdfg1 = dace.SDFG(util.unique_name("map_buffer"))
    inp1, inp1_desc = nsdfg1.add_array("__inp", (10,), dace.float64)
    out1, out1_desc = nsdfg1.add_array("__out", (10, 10), dace.float64)
    tmp1, _ = nsdfg1.add_temp_transient_like(out1_desc)
    state1 = nsdfg1.add_state()
    tmp1_node = state1.add_access(tmp1)

    nsdfg2 = dace.SDFG(util.unique_name("map_buffer"))
    inp2, _ = nsdfg2.add_array("__inp", (10,), dace.float64)
    out2, out2_desc = nsdfg2.add_array("__out", (10,), dace.float64)
    tmp2, _ = nsdfg2.add_temp_transient_like(out2_desc)
    state2 = nsdfg2.add_state()
    tmp2_node = state2.add_access(tmp2)

    state2.add_mapped_tasklet(
        "broadcast2",
        map_ranges={"__i": "0:10"},
        code="__oval = __ival + 1.0",
        inputs={
            "__ival": dace.Memlet(f"{inp2}[__i]"),
        },
        outputs={
            "__oval": dace.Memlet(f"{tmp2}[__i]"),
        },
        output_nodes={tmp2_node},
        external_edges=True,
    )
    state2.add_nedge(tmp2_node, state2.add_access(out2), dace.Memlet.from_array(out2, out2_desc))

    nsdfg2_node = state1.add_nested_sdfg(nsdfg2, nsdfg1, inputs={"__inp"}, outputs={"__out"})
    me1, mx1 = state1.add_map("broadcast1", ndrange={"__i": "0:10"})
    state1.add_memlet_path(
        state1.add_access(inp1),
        me1,
        nsdfg2_node,
        dst_conn="__inp",
        memlet=dace.Memlet.from_array(inp1, inp1_desc),
    )
    state1.add_memlet_path(
        nsdfg2_node, mx1, tmp1_node, src_conn="__out", memlet=dace.Memlet(f"{tmp1}[__i, 0:10]")
    )
    state1.add_nedge(tmp1_node, state1.add_access(out1), dace.Memlet.from_array(out1, out1_desc))

    nsdfg1_node = state.add_nested_sdfg(nsdfg1, sdfg, inputs={"__inp"}, outputs={"__out"})
    me, mx = state.add_map("broadcast", ndrange={"__i": "0:10"})
    state.add_memlet_path(
        state.add_access(inp),
        me,
        nsdfg1_node,
        dst_conn="__inp",
        memlet=dace.Memlet.from_array(inp, inp_desc),
    )
    state.add_memlet_path(
        nsdfg1_node, mx, tmp_node, src_conn="__out", memlet=dace.Memlet(f"{tmp}[__i, 0:10, 0:10]")
    )
    state.add_nedge(tmp_node, state.add_access(out), dace.Memlet.from_array(out, out_desc))

    sdfg.validate()

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.GT4PyMapBufferElimination(
            assume_pointwise=False,
        ),
        validate=True,
        validate_all=True,
    )
    assert count == 3
    assert out1_desc.strides == out_desc.strides[1:]
    assert out2_desc.strides == out_desc.strides[2:]
