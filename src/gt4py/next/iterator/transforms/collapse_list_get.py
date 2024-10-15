# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py import eve
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im


class CollapseListGet(eve.PreserveLocationVisitor, eve.NodeTranslator):
    """Simplifies expressions containing `list_get`.

    Examples
      - `list_get(i, neighbors(o, it))` -> `deref(shift(o, i)(it))`
      - `list_get(i, make_const_list(e))` -> `e`
    """

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Node:
        node = self.generic_visit(node)
        if node.fun == ir.SymRef(id="list_get"):
            if cpm.is_call_to(node.args[1], "if_"):
                list_idx = node.args[0]
                cond, true_val, false_val = node.args[1].args
                return im.if_(
                    cond,
                    self.visit(im.call("list_get")(list_idx, true_val)),
                    self.visit(im.call("list_get")(list_idx, false_val)),
                )
            if isinstance(node.args[1], ir.FunCall):
                if node.args[1].fun == ir.SymRef(id="neighbors"):
                    offset_tag = node.args[1].args[0]
                    offset_index = (
                        ir.OffsetLiteral(value=int(node.args[0].value))
                        if isinstance(node.args[0], ir.Literal)
                        else node.args[
                            0
                        ]  # else-branch: e.g. SymRef from unroll_reduce, TODO(havogt): remove when we replace unroll_reduce by list support in gtfn
                    )
                    it = node.args[1].args[1]
                    return ir.FunCall(
                        fun=ir.SymRef(id="deref"),
                        args=[
                            ir.FunCall(
                                fun=ir.FunCall(
                                    fun=ir.SymRef(id="shift"), args=[offset_tag, offset_index]
                                ),
                                args=[it],
                            )
                        ],
                    )
                if node.args[1].fun == ir.SymRef(id="make_const_list"):
                    return node.args[1].args[0]

        return node
