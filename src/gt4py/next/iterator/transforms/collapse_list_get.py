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

from gt4py import eve
from gt4py.next.iterator import ir


class CollapseListGet(eve.NodeTranslator):
    """Simplifies expressions containing `list_get`.

    Examples
      - `list_get(i, neighbors(o, it))` -> `deref(shift(o, i)(it))`
      - `list_get(i, make_const_list(e))` -> `e`
    """

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Node:
        node = self.generic_visit(node)
        if node.fun == ir.SymRef(id="list_get"):
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
                                    fun=ir.SymRef(id="shift"),
                                    args=[offset_tag, offset_index],
                                ),
                                args=[it],
                            )
                        ],
                    )
                if node.args[1].fun == ir.SymRef(id="make_const_list"):
                    return node.args[1].args[0]

        return node
