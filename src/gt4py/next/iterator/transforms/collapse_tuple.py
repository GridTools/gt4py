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

from dataclasses import dataclass

from gt4py import eve
from gt4py.next.iterator import ir, type_inference


def _get_tuple_size(node: ir.Node) -> int:
    # TODO(havogt): This fails if the tuple is a SymRef. Use type information from (entire) tree when available.
    infered_type = type_inference.infer(node)
    assert isinstance(infered_type, type_inference.Val)
    dtype = infered_type.dtype
    assert isinstance(dtype, (type_inference.Tuple, type_inference.EmptyTuple))
    return len(dtype)


@dataclass(frozen=True)
class CollapseTuple(eve.NodeTranslator):
    """Transform `make_tuple(tuple_get(0, t), tuple_get(1, t), ..., tuple_get(N-1,t))` -> t."""

    ignore_tuple_size: bool

    @classmethod
    def apply(cls, node: ir.Node, ignore_tuple_size: bool = False) -> ir.Node:
        """
        Transform `make_tuple(tuple_get(0, t), tuple_get(1, t), ..., tuple_get(N-1,t))` -> t.

        If `ignore_tuple_size`, apply the transformation even if length of the inner tuple
        is greater than the length of the outer tuple.
        """
        return cls(ignore_tuple_size).visit(node)

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Node:
        if node.fun == ir.SymRef(id="make_tuple") and all(
            isinstance(arg, ir.FunCall) and arg.fun == ir.SymRef(id="tuple_get")
            for arg in node.args
        ):
            assert isinstance(node.args[0], ir.FunCall)
            first_expr = node.args[0].args[1]

            for i, v in enumerate(node.args):
                assert isinstance(v, ir.FunCall)
                assert isinstance(v.args[0], ir.Literal)
                if not (int(v.args[0].value) == i and v.args[1] == first_expr):
                    return self.generic_visit(node)

            if self.ignore_tuple_size or _get_tuple_size(first_expr) == len(node.args):
                return first_expr
        if (
            node.fun == ir.SymRef(id="tuple_get")
            and isinstance(node.args[1], ir.FunCall)
            and node.args[1].fun == ir.SymRef(id="make_tuple")
            and isinstance(node.args[0], ir.Literal)
        ):
            assert node.args[0].type in ["int", "int32", "int64"]  # TODO
            return node.args[1].args[int(node.args[0].value)]  # TODO simplify

        return self.generic_visit(node)
