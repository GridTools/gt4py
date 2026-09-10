# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy

import dace
import numpy as np
import pytest

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util


def _mk_single_iteration_sdfg() -> dace.SDFG:
    """A Map whose second dimension has exactly one iteration."""
    sdfg = dace.SDFG(util.unique_name("single_iteration_map"))
    for name in ["a", "b"]:
        sdfg.add_array(name, shape=(20, 10), dtype=dace.float64, transient=False)

    state = sdfg.add_state(is_start_block=True)
    state.add_mapped_tasklet(
        "comp",
        map_ranges={"__i": "0:20", "__j": "7:8"},
        inputs={"__in": dace.Memlet("a[__i, __j]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("b[__i, __j]")},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg


def test_trivial_map_dimension_folding():
    sdfg = _mk_single_iteration_sdfg()

    ref = {
        name: np.array(np.random.rand(*desc.shape), copy=True, dtype=desc.dtype.as_numpy_dtype())
        for name, desc in sdfg.arrays.items()
        if not desc.transient
    }
    res = copy.deepcopy(ref)
    util.compile_and_run_sdfg(sdfg, **ref)

    # a single application although applied repeatedly, i.e. the rewrite is a fixpoint
    nb_apply = sdfg.apply_transformations_repeated(
        gtx_transformations.TrivialMapDimensionFolding(),
        validate=True,
        validate_all=True,
    )
    assert nb_apply == 1

    map_entry = next(
        node
        for state in sdfg.states()
        for node in state.nodes()
        if isinstance(node, dace.sdfg.nodes.MapEntry)
    )
    # The dimension must survive, only its uses are rewritten.
    assert map_entry.map.params == ["__i", "__j"]
    assert str(map_entry.map.range[1][0]) == "7"
    assert all(
        "__j" not in str(subset)
        for state in sdfg.states()
        for edge in state.edges()
        if edge.data is not None
        for subset in (edge.data.subset, edge.data.other_subset)
        if subset is not None
    )

    util.compile_and_run_sdfg(sdfg, **res)
    assert all(np.allclose(ref[name], res[name]) for name in ref.keys())


def test_trivial_map_dimension_folding_ignores_multi_iteration():
    """A Map without a single iteration dimension must not be touched."""
    sdfg = dace.SDFG(util.unique_name("multi_iteration_map"))
    for name in ["a", "b"]:
        sdfg.add_array(name, shape=(20, 10), dtype=dace.float64, transient=False)
    state = sdfg.add_state(is_start_block=True)
    state.add_mapped_tasklet(
        "comp",
        map_ranges={"__i": "0:20", "__j": "0:10"},
        inputs={"__in": dace.Memlet("a[__i, __j]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("b[__i, __j]")},
        external_edges=True,
    )
    sdfg.validate()

    assert (
        sdfg.apply_transformations_repeated(
            gtx_transformations.TrivialMapDimensionFolding(), validate=True
        )
        == 0
    )
