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
"""
Simple tests top verify the map fusion tests.
"""

import dace
import copy
from gt4py.next.common import NeighborTable
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.program_processors.runners import dace_fieldview as dace_backend
from gt4py.next.type_system import type_specifications as ts
from functools import reduce
import numpy as np

from typing import Sequence, Any

from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations,  # noqa: F401 [unused-import]  # For development.
)

from simple_icon_mesh import (
    IDim,  # Dimensions
    JDim,
    KDim,
    EdgeDim,
    VertexDim,
    CellDim,
    ECVDim,
    E2C2VDim,
    NbCells,  # Constants of the size
    NbEdges,
    NbVertices,
    E2C2VDim,  # Offsets
    E2C2V,
    SIZE_TYPE,  # Type definitions
    E2C2V_connectivity,
    E2ECV_connectivity,
    make_syms,  # Helpers
)

# For cartesian stuff.
N = 10
IFTYPE = ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
IJFTYPE = ts.FieldType(dims=[IDim, JDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))


def _perform_test(
    sdfg: dace.SDFG, ref: Any, return_names: Sequence[str] | str, args: dict[str, Any]
) -> dace.SDFG:
    unopt_sdfg = copy.deepcopy(sdfg)

    if not isinstance(ref, list):
        ref = [ref]
    if isinstance(return_names, str):
        return_names = [return_names]

    SYMBS = make_syms(**args)

    # Call the unoptimized version of the SDFG
    unopt_sdfg(**args, **SYMBS)
    unopt_res = [args[name] for name in return_names]

    assert np.allclose(ref, unopt_res), "The unoptimized verification failed."

    # Reset the results
    for name in return_names:
        args[name][:] = 0
    assert not np.allclose(ref, unopt_res)

    # Now perform the optimization
    opt_sdfg = copy.deepcopy(sdfg)
    transformations.gt_auto_optimize(opt_sdfg)
    opt_sdfg.validate()
    opt_sdfg(**args, **SYMBS)
    opt_res = [args[name] for name in return_names]

    assert np.allclose(ref, opt_res), "The optimized verification failed."

    return opt_sdfg


def _count_nodes(
    sdfg: dace.SDFG,
    state: dace.SDFGState | None = None,
    node_type: Sequence[type] | type = dace_nodes.MapEntry,
) -> int:
    states = sdfg.states() if state is None else [state]
    found_matches = 0
    for state_nodes in states:
        for node in state_nodes.nodes():
            if isinstance(node, node_type):
                found_matches += 1
    return found_matches


######################
#   TESTS


def exclusive_only():
    """Tests the sxclusive set merging mechanism only."""

    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=IDim.value), 0, "size")
    )
    stencil1 = im.call(
        im.call("as_fieldop")(
            im.lambda_("a")(im.plus(im.deref("a"), 1.0)),
            domain,
        )
    )(
        im.call(
            im.call("as_fieldop")(
                im.lambda_("a")(im.plus(im.deref("a"), 2.0)),
                domain,
            )
        )("x"),
    )

    a = np.random.rand(N)

    testee = itir.Program(
        id=f"sum_3fields_1",
        function_definitions=[],
        params=[
            itir.Sym(id="x", type=IFTYPE),
            itir.Sym(id="z", type=IFTYPE),
            itir.Sym(id="size", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=stencil1,
                domain=domain,
                target=itir.SymRef(id="z"),
            )
        ],
    )

    sdfg = dace_backend.build_sdfg_from_gtir(testee, {})
    sdfg.validate()

    assert _count_nodes(sdfg, node_type=dace_nodes.AccessNode) == 3
    assert _count_nodes(sdfg, node_type=dace_nodes.MapEntry) == 2

    a = np.random.rand(N)
    res1 = np.empty_like(a)

    args = {
        "x": a,
        "z": res1,
        "size": N,
    }
    return_names = ["z"]

    opt_sdfg = _perform_test(
        sdfg=sdfg,
        ref=a + 3.0,
        return_names="z",
        args=args,
    )

    assert _count_nodes(opt_sdfg, node_type=dace_nodes.AccessNode) == 3
    assert _count_nodes(opt_sdfg, node_type=dace_nodes.MapEntry) == 1


def exclusive_only_2():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=IDim.value), 0, "size")
    )
    stencil1 = im.call(
        im.call("as_fieldop")(
            im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
            domain,
        )
    )(
        "y",
        im.call(
            im.call("as_fieldop")(
                im.lambda_("a")(im.plus(im.deref("a"), 2.0)),
                domain,
            )
        )("x"),
    )

    a = np.random.rand(N)
    b = np.random.rand(N)

    testee = itir.Program(
        id=f"sum_3fields_1",
        function_definitions=[],
        params=[
            itir.Sym(id="x", type=IFTYPE),
            itir.Sym(id="y", type=IFTYPE),
            itir.Sym(id="z", type=IFTYPE),
            itir.Sym(id="size", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=stencil1,
                domain=domain,
                target=itir.SymRef(id="z"),
            )
        ],
    )

    sdfg = dace_backend.build_sdfg_from_gtir(testee, {})
    sdfg.validate()

    assert _count_nodes(sdfg, node_type=dace_nodes.AccessNode) == 4
    assert _count_nodes(sdfg, node_type=dace_nodes.MapEntry) == 2

    a = np.random.rand(N)
    res1 = np.empty_like(a)

    args = {
        "x": a,
        "y": b,
        "z": res1,
        "size": N,
    }
    return_names = ["z"]

    opt_sdfg = _perform_test(
        sdfg=sdfg,
        ref=(a + b + 2.0),
        return_names="z",
        args=args,
    )

    assert _count_nodes(opt_sdfg, node_type=dace_nodes.AccessNode) == 4
    assert _count_nodes(opt_sdfg, node_type=dace_nodes.MapEntry) == 1


def intermediate_branch():
    sdfg = dace.SDFG("intermediate")
    state = sdfg.add_state("state")

    ac: list[nodes.AccessNode] = []
    for i in range(3):
        name = "input" if i == 0 else f"output{i-1}"
        sdfg.add_array(
            name,
            shape=(N,),
            dtype=dace.float64,
            transient=False,  # All are global.
        )
        ac.append(state.add_access(name))
    sdfg.add_array(
        name="tmp",
        shape=(N,),
        dtype=dace.float64,
        transient=True,
    )
    ac.append(state.add_access("tmp"))

    state.add_mapped_tasklet(
        "first_add",
        map_ranges=[("i", f"0:{N}")],
        code="__out = __in0 + 1.0",
        inputs=dict(__in0=dace.Memlet("input[i]")),
        outputs=dict(__out=dace.Memlet("tmp[i]")),
        input_nodes=dict(input=ac[0]),
        output_nodes=dict(tmp=ac[-1]),
        external_edges=True,
    )

    for i in range(2):
        state.add_mapped_tasklet(
            f"level_{i}_add",
            map_ranges=[("i", f"0:{N}")],
            code=f"__out = __in0 + {i+3}",
            inputs=dict(__in0=dace.Memlet("tmp[i]")),
            outputs=dict(__out=dace.Memlet(f"output{i}[i]")),
            input_nodes=dict(tmp=ac[-1]),
            output_nodes={f"output{i}": ac[1 + i]},
            external_edges=True,
        )

    assert _count_nodes(sdfg, node_type=dace_nodes.AccessNode) == 4
    assert _count_nodes(sdfg, node_type=dace_nodes.MapEntry) == 3

    a = np.random.rand(N)
    ref0 = a + 1 + 3
    ref1 = a + 1 + 4

    res0 = np.empty_like(a)
    res1 = np.empty_like(a)

    args = {
        "input": a,
        "output0": res0,
        "output1": res1,
    }
    return_names = ["output0", "output1"]

    opt_sdfg = _perform_test(
        sdfg=sdfg,
        ref=[ref0, ref1],
        return_names=return_names,
        args=args,
    )
    assert _count_nodes(opt_sdfg, node_type=dace_nodes.AccessNode) == 4
    assert _count_nodes(opt_sdfg, node_type=dace_nodes.MapEntry) == 1


if "__main__" == __name__:
    # exclusive_only()
    # exclusive_only_2()
    intermediate_branch()
    print("SUCCESS")
