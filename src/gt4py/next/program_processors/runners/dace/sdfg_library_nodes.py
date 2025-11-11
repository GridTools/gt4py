# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Final

import dace
from dace import library as dace_library, nodes as dace_nodes
from dace.transformation import transformation as dace_transform


_INPUT_NAME: Final[str] = "_input"
_OUTPUT_NAME: Final[str] = "_output"


@dace_library.expansion
class ExpandPure(dace_transform.ExpandTransformation):
    """Implements pure expansion of the Fill library node."""

    environments: Final[list[Any]] = []

    @staticmethod
    def expansion(node: Fill, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        sdfg = dace.SDFG(f"{node.label}_sdfg")

        assert len(parent_state.out_edges(node)) == 1
        outedge = parent_state.out_edges(node)[0]
        out_desc = parent_sdfg.arrays[outedge.data.data]
        inner_out_desc = out_desc.clone()
        inner_out_desc.transient = False
        out = sdfg.add_datadesc(_OUTPUT_NAME, inner_out_desc)
        outedge._src_conn = _OUTPUT_NAME
        node.add_out_connector(_OUTPUT_NAME)

        state = sdfg.add_state(f"{node.label}_state")
        map_params = [f"__i{i}" for i in range(len(out_desc.shape))]
        map_rng = {i: f"0:{s}" for i, s in zip(map_params, out_desc.shape)}
        out_mem = dace.Memlet(expr=f"{out}[{','.join(map_params)}]")
        outputs = {"_out": out_mem}

        assert len(parent_state.in_edges(node)) == 1
        inedge = parent_state.in_edges(node)[0]
        inp_desc = parent_sdfg.arrays[inedge.data.data]
        inner_inp_desc = inp_desc.clone()
        inner_inp_desc.transient = False
        inp = sdfg.add_datadesc(_INPUT_NAME, inner_inp_desc)
        inedge._dst_conn = _INPUT_NAME
        node.add_in_connector(_INPUT_NAME)
        inputs = {"_in": dace.Memlet(data=inp, subset="0")}
        code = "_out = _in"

        state.add_mapped_tasklet(
            f"{node.label}_tasklet", map_rng, inputs, code, outputs, external_edges=True
        )

        return sdfg


@dace_library.node
class Fill(dace_nodes.LibraryNode):
    """Implements filling data containers with a single value"""

    implementations: Final[dict[str, dace_transform.ExpandTransformation]] = {"pure": ExpandPure}
    default_implementation: Final[str] = "pure"

    def __init__(self, name: str):
        super().__init__(name)
