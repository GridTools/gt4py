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
    """
    Removes cast expressions where the argument is already in the target type.

    This transformation requires the IR to be fully type-annotated,
    therefore it should be applied after type-inference.
    """

    def visit_FunCall(self, node: ir.FunCall) -> ir.Node:
        node = self.generic_visit(node)

        if cpm.is_call_to(node, "cast_"):
            value, type_constructor = node.args

            assert (
                value.type
                and isinstance(type_constructor, ir.SymRef)
                and (type_constructor.id in ir.TYPEBUILTINS)
            )
            dtype = ts.ScalarType(kind=getattr(ts.ScalarKind, type_constructor.id.upper()))

            if value.type == dtype:
                return value

        elif (
            cpm.is_applied_as_fieldop(node)
            and isinstance(node.fun.args[0], ir.Lambda)  # type: ignore[attr-defined]
            and cpm.is_call_to(node.fun.args[0].expr, "deref")  # type: ignore[attr-defined]
            and cpm.is_ref_to(node.fun.args[0].expr.args[0], node.fun.args[0].params[0].id)  # type: ignore[attr-defined]
        ):
            # pruning of cast expressions may leave some trivial `as_fieldop` expressions
            # with form '(⇑(λ(__arg) → ·__arg))(a)'
            return node.args[0]

        return node

    @classmethod
    def apply(cls, node: ir.Node) -> ir.Node:
        return cls().visit(node)
