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


import dace

from gt4py import eve
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace_fieldview import utility as dace_fieldview_util
from gt4py.next.program_processors.runners.dace_fieldview.gtir_builtins.gtir_builtin_translator import (
    GTIRBuiltinTranslator,
    SDFGField,
    SDFGFieldBuilder,
)


class GTIRBuiltinSelect(GTIRBuiltinTranslator):
    """Generates the dataflow subgraph for the `select` builtin function."""

    true_br_builder: SDFGFieldBuilder
    false_br_builder: SDFGFieldBuilder

    def __init__(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        dataflow_builder: eve.NodeVisitor,
        node: itir.FunCall,
    ):
        super().__init__(sdfg, state)

        assert cpm.is_call_to(node.fun, "select")
        assert len(node.fun.args) == 3
        cond_expr, true_expr, false_expr = node.fun.args

        # expect condition as first argument
        cond = dace_fieldview_util.get_symbolic_expr(cond_expr)

        # use current head state to terminate the dataflow, and add a entry state
        # to connect the true/false branch states as follows:
        #
        #               ------------
        #           === |  select  | ===
        #          ||   ------------   ||
        #          \/                  \/
        #     ------------       -------------
        #     |   true   |       |   false   |
        #     ------------       -------------
        #          ||                  ||
        #          ||   ------------   ||
        #           ==> |   head   | <==
        #               ------------
        #
        select_state = sdfg.add_state_before(state, state.label + "_select")
        sdfg.remove_edge(sdfg.out_edges(select_state)[0])

        # expect true branch as second argument
        true_state = sdfg.add_state(state.label + "_true_branch")
        sdfg.add_edge(select_state, true_state, dace.InterstateEdge(condition=cond))
        sdfg.add_edge(true_state, state, dace.InterstateEdge())
        self.true_br_builder = dataflow_builder.visit(true_expr, sdfg=sdfg, head_state=true_state)

        # and false branch as third argument
        false_state = sdfg.add_state(state.label + "_false_branch")
        sdfg.add_edge(select_state, false_state, dace.InterstateEdge(condition=(f"not {cond}")))
        sdfg.add_edge(false_state, state, dace.InterstateEdge())
        self.false_br_builder = dataflow_builder.visit(
            false_expr, sdfg=sdfg, head_state=false_state
        )

    def build(self) -> list[SDFGField]:
        # retrieve true/false states as predecessors of head state
        branch_states = tuple(edge.src for edge in self.sdfg.in_edges(self.head_state))
        assert len(branch_states) == 2
        if branch_states[0].label.endswith("_true_branch"):
            true_state, false_state = branch_states
        else:
            false_state, true_state = branch_states

        true_br_args = self.true_br_builder()
        false_br_args = self.false_br_builder()

        output_nodes = []
        for true_br, false_br in zip(true_br_args, false_br_args, strict=True):
            true_br_node, true_br_type = true_br
            assert isinstance(true_br_node, dace.nodes.AccessNode)
            false_br_node, false_br_type = false_br
            assert isinstance(false_br_node, dace.nodes.AccessNode)
            assert true_br_type == false_br_type
            array_type = self.sdfg.arrays[true_br_node.data]
            access_node = self.add_local_storage(true_br_type, array_type.shape)
            output_nodes.append((access_node, true_br_type))

            data_name = access_node.data
            true_br_output_node = true_state.add_access(data_name)
            true_state.add_nedge(
                true_br_node,
                true_br_output_node,
                dace.Memlet.from_array(
                    true_br_output_node.data, true_br_output_node.desc(self.sdfg)
                ),
            )

            false_br_output_node = false_state.add_access(data_name)
            false_state.add_nedge(
                false_br_node,
                false_br_output_node,
                dace.Memlet.from_array(
                    false_br_output_node.data, false_br_output_node.desc(self.sdfg)
                ),
            )
        return output_nodes
