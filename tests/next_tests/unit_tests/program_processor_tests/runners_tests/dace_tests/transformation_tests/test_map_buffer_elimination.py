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

from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)

from . import util

import dace


def _make_test_data(names: list[str]) -> dict[str, np.ndarray]:
    return {name: np.array(np.random.rand(10), dtype=np.float64, copy=True) for name in names}


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
