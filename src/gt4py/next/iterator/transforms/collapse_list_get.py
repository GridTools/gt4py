# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py import eve
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im


class CollapseListGet(eve.PreserveLocationVisitor, eve.NodeTranslator):
    """Simplifies expressions containing `list_get`.

    Examples
      - `list_get(i, neighbors(o, it))` -> `deref(shift(o, i)(it))`
      - `list_get(i, make_const_list(e))` -> `e`
    """

    def visit_FunCall(self, node: itir.FunCall, **kwargs) -> itir.Node:
        node = self.generic_visit(node)
        if cpm.is_call_to(node, "list_get"):
            if cpm.is_call_to(node.args[1], "if_"):
                list_idx = node.args[0]
                cond, true_val, false_val = node.args[1].args
                return im.if_(
                    cond,
                    self.visit(im.list_get(list_idx, true_val)),
                    self.visit(im.list_get(list_idx, false_val)),
                )
            if cpm.is_call_to(node.args[1], "neighbors"):
                offset_tag = node.args[1].args[0]
                offset_index = (
                    itir.OffsetLiteral(value=int(node.args[0].value))
                    if isinstance(node.args[0], itir.Literal)
                    else node.args[
                        0
                    ]  # else-branch: e.g. SymRef from unroll_reduce, TODO(havogt): remove when we replace unroll_reduce by list support in gtfn
                )
                it = node.args[1].args[1]
                return im.deref(im.shift(offset_tag, offset_index)(it))
            if cpm.is_call_to(node.args[1], "make_const_list"):
                return node.args[1].args[0]
            if cpm.is_applied_map(node.args[1]):
                # list_get(0, map_(λ(val_) → foo(val_, int64))(·__sym_1))
                # -> (λ(val_) → foo(val_, int64))(list_get(0, ·__sym_1))
                lsts = node.args[1].args
                assert len(node.args[1].fun.args) == 1  # a single lambda in the map
                mapped_lambda = node.args[1].fun.args[0]
                res = im.call(mapped_lambda)(*[im.list_get(node.args[0], lst) for lst in lsts])
                return res

        return node
