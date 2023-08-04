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

from gt4py.eve import NodeTranslator
from gt4py.next.iterator import embedded, ir, ir_makers as im


class ConstantFolding(NodeTranslator):
    @classmethod
    def apply(cls, node: ir.Node):
        return cls().visit(node)

    def visit_FunCall(self, node: ir.FunCall):
        new_node = self.generic_visit(node)

        if len(new_node.args) > 0:
            if (
                isinstance(new_node.args[0], ir.Literal) and new_node.fun.id == "if_"
            ):  # if_(True, true_branch, false_branch) -> true_branch
                if new_node.args[0].value == "True":
                    new_node = new_node.args[1]
                else:
                    new_node = new_node.args[2]

            if isinstance(new_node, ir.FunCall) and all(
                isinstance(arg, ir.Literal) for arg in new_node.args
            ):  # 1 + 1 -> 2
                assert isinstance(new_node.fun, ir.SymRef)  # for mypy
                if "make_" not in new_node.fun.id:  # for make_tuple, make_const_list
                    val_ls = []
                    for arg in new_node.args:
                        val_ls.append(getattr(embedded, str(arg.type))(arg.value))  # type: ignore[attr-defined] # arg type already established in if condition
                    new_node = im.literal_from_value(
                        (getattr(embedded, str(new_node.fun.id))(*val_ls))
                    )

        return new_node
