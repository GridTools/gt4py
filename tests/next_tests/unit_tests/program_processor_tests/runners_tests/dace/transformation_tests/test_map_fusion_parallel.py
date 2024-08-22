# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Any, Optional, Sequence, Union, Literal, overload

import pytest
import dace
import copy
import numpy as np
from dace.sdfg import nodes as dace_nodes
from dace.transformation import dataflow as dace_dataflow

from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)
from . import util


def _count_toplevel_maps(sdfg: dace.SDFG) -> int:
    topLevelMaps = 0
    for state in sdfg.nodes():
        scope = state.scope_dict()
        for maps in filter(lambda n: isinstance(n, dace_nodes.MapEntry), state.nodes()):
            if scope[maps] is None:
                topLevelMaps += 1
    return topLevelMaps


def _make_parallel_sdfg_1(
    N_a: str | int,
    N_b: str | int | None = None,
    N_c: str | int | None = None,
    it_var_b: str | None = None,
    it_var_c: str | None = None,
) -> dace.SDFG:
    """Create the "parallel_1_sdfg".

    This is a simple SDFG with two parallel Map, it has one input `a`. Further,
    it has the two outputs `b` and `c` defined as `b := a + 2.` and `c := a + 4.`.
    The array may have different length, but the size of `b` and `c` must be smaller
    or equal the size of `a`.
    By using `it_var_{b,c}` it is possible to control which iteration variables should
    be used by the Maps. If `it_var_b` is not given it defaults to `__i0` and if
    `it_var_c` is not given it defaults to the value given to `_it_var_b`.

    Args:
        N_a:        The length of array `a`, must be the largest.
        N_b:        The length of array `b`, if not given equals to `N_a`.
        N_c:        The length of array `c`, if not given equals to `N_a`.
        it_var_b:   The iteration variable used by the Map handling `b`.
        it_var_c:   The iteration variable used by the Map handling `c`.
    """

    if N_b is None:
        N_b = N_a
    if N_c is None:
        N_c = N_a
    if it_var_b is None:
        it_var_b = "__i0"
    if it_var_c is None:
        it_var_c = it_var_b

    shapes = {"a": N_a, "b": N_b, "c": N_c}

    sdfg = dace.SDFG("parallel_1_sdfg")
    state = sdfg.add_state(is_start_block=True)

    for name in ["a", "b", "c"]:
        sdfg.add_array(
            name=name,
            shape=(shapes[name],),
            dtype=dace.float64,
            transient=False,
        )
    a = state.add_access("a")

    state.add_mapped_tasklet(
        name="first_computation",
        map_ranges=[(it_var_b, f"0:{shapes['b']}")],
        inputs={"__in0": dace.Memlet(f"a[{it_var_b}]")},
        code="__out = __in0 + 2.0",
        outputs={"__out": dace.Memlet(f"b[{it_var_b}]")},
        input_nodes={"a": a},
        external_edges=True,
    )

    state.add_mapped_tasklet(
        name="second_computation",
        map_ranges=[(it_var_c, f"0:{shapes['c']}")],
        input_nodes={"a": a},
        inputs={"__in0": dace.Memlet(f"a[{it_var_c}]")},
        code="__out = __in0 + 4.0",
        outputs={"__out": dace.Memlet(f"c[{it_var_c}]")},
        external_edges=True,
    )

    return sdfg


def test_simple_fusing() -> None:
    """Tests a simple case of parallel map fusion.

    The parallel maps have the same sizes and same iteration bounds and variables.
    This means that the transformation applies.
    """

    # The size of the request.
    N = 10
    sdfg = _make_parallel_sdfg_1(N_a="N_a")
    assert _count_toplevel_maps(sdfg) == 2

    # Now run the optimization
    sdfg.apply_transformations_repeated([gtx_transformations.ParallelMapFusion], validate_all=True)
    assert _count_toplevel_maps(sdfg) == 1, f"Expected that the two maps were fused."

    # Now run the SDFG to check if the code is still valid.
    a = np.random.rand(N)
    b = np.zeros_like(a)
    c = np.zeros_like(a)

    # Compute the reference solution.
    ref_b = a + 2
    ref_c = a + 4

    # Now calling the SDFG.
    sdfg(a=a, b=b, c=c, N_a=N)

    assert np.allclose(ref_b, b), f"Computation of 'b' failed."
    assert np.allclose(ref_c, c), f"Computation of 'c' failed."


def test_non_fusable() -> None:
    """Tests a case where the bounds did not match.

    This can never be fused.
    """
    N = 10
    N_a, N_b, N_c = N, N, N - 1

    sdfg = _make_parallel_sdfg_1(N_a="N_a", N_b="N_a", N_c="N_c")
    assert _count_toplevel_maps(sdfg) == 2

    # Now run the optimization, which will not succeed.
    sdfg.apply_transformations_repeated([gtx_transformations.ParallelMapFusion], validate_all=True)
    assert _count_toplevel_maps(sdfg) == 2, f"Expected that the two maps could not be fused."

    # Testing if it still runs as expected.
    a = np.random.rand(N_a)
    b = np.zeros(N_b)
    c = np.zeros(N_c)

    # Compute the reference solution.
    ref_b = a[0:N_b] + 2
    ref_c = a[0:N_c] + 4

    # Now calling the SDFG.
    sdfg(a=a, b=b, c=c, N_a=N_a, N_c=N_c)

    assert np.allclose(ref_b, b), f"Computation of 'b' failed."
    assert np.allclose(ref_c, c), f"Computation of 'c' failed."


@pytest.mark.xfail(reason="Renaming of iteration variables is not implemented.")
def test_renaming_fusing() -> None:
    """Tests if the renaming works.

    The two parallel maps are technically fusable, but the iteration variables
    are different, thus a renaming must be done.

    Note:
        The renaming feature is currently not implemented, so this test will
        (currently) fail.
    """

    # The size of the request.
    N = 10
    sdfg = _make_parallel_sdfg_1(N_a="N_a", it_var_b="__itB", it_var_c="__itC")
    assert _count_toplevel_maps(sdfg) == 2

    # Now run the optimization
    sdfg.apply_transformations_repeated([gtx_transformations.ParallelMapFusion], validate_all=True)
    assert _count_toplevel_maps(sdfg) == 1, f"Expected that the two maps were fused."

    # Now run the SDFG to check if the code is still valid.
    a = np.random.rand(N)
    b = np.zeros_like(a)
    c = np.zeros_like(a)

    # Compute the reference solution.
    ref_b = a + 2
    ref_c = a + 4

    # Now calling the SDFG.
    sdfg(a=a, b=b, c=c, N_a=N)

    assert np.allclose(ref_b, b), f"Computation of 'b' failed."
    assert np.allclose(ref_c, c), f"Computation of 'c' failed."
