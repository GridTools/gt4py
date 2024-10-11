# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.type_system import type_specifications as ts


class PruneCasts(PreserveLocationVisitor, NodeTranslator):
    """Removes cast expression where the argument is already in the target type."""

    def visit_FunCall(self, node: ir.FunCall) -> ir.Node:
        if not cpm.is_call_to(node, "cast_"):
            node.args = self.visit(node.args)
            return node

        value, type_constructor = node.args
        self.visit(value)

        # cannot prune cast if type annotation is missing on input argument
        if value.type is None:
            return node

        assert isinstance(type_constructor, ir.SymRef) and (type_constructor.id in ir.TYPEBUILTINS)
        dtype = ts.ScalarType(kind=getattr(ts.ScalarKind, type_constructor.id.upper()))

        if value.type == dtype:
            return value

        return node
