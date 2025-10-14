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

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util


import dace


def _make_sdfg_simple() -> tuple[dace.SDFG, dace.SDFGState]:
    sdfg = dace.SDFG(util.unique_name("simple_sdfg"))
    state = sdfg.add_state(is_start_block=True)

    for name in "abc":
        sdfg.add_array(
            name,
            shape=((10,) if name == "a" else (8,)),
            dtype=dace.float64,
        )
    sdfg.arrays["b"].transient = True

    a, b, c = (state.add_access(name) for name in "abc")

    state.add_mapped_tasklet(
        "computation",
        map_ranges={"__i": "2:6"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + 1.34",
        outputs={"__out": dace.Memlet("b[__i - 2]")},
        output_nodes={b},
        input_nodes={a},
        external_edges=True,
    )

    state.add_nedge(b, c, dace.Memlet("b[1:4] -> [0:3]"))

    sdfg.validate()
    return sdfg, state


def test_simple_split_case():
    sdfg, state = _make_sdfg_simple()

    ret = sdfg.apply_transformations_repeated(
        gtx_transformations.MapSplitter(),
        validate=True,
        validate_all=True,
    )
    assert ret == 1

    assert False, "Expand me"
