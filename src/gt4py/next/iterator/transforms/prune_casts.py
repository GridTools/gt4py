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


def _is_cast_redundant(node: ir.FunCall) -> bool:
    assert cpm.is_call_to(node, "cast_")
    value, type_constructor = node.args

    assert (
        value.type
        and isinstance(type_constructor, ir.SymRef)
        and (type_constructor.id in ir.TYPEBUILTINS)
    )
    dtype = ts.ScalarType(kind=getattr(ts.ScalarKind, type_constructor.id.upper()))

    return value.type == dtype


class PruneCasts(PreserveLocationVisitor, NodeTranslator):
    """
    Removes cast expressions where the argument is already in the target type.

    This transformation requires the IR to be fully type-annotated,
    therefore it should be applied after type-inference.
    """

    def visit_FunCall(self, node: ir.FunCall) -> ir.Node:
        if cpm.is_op_as_fieldop(node, "cast_"):
            # Prune trivial `as_fieldop` cast expressions with the form:
            # as_fieldop(λ(__val) = cast_(deref(__val), float64)))(a)
            # where 'a' is already a field with data type float64, in this example.
            lambda_node = node.fun.args[0]  # type: ignore[attr-defined]
            assert isinstance(lambda_node, ir.Lambda)
            cast_expr = lambda_node.expr
            assert isinstance(cast_expr, ir.FunCall)
            cast_value = cast_expr.args[0]
            if (
                _is_cast_redundant(cast_expr)
                and cpm.is_call_to(cast_value, "deref")
                and cpm.is_ref_to(cast_value.args[0], lambda_node.params[0].id)
            ):
                return self.visit(node.args[0])

        elif cpm.is_call_to(node, "cast_"):
            if _is_cast_redundant(node):
                return self.visit(node.args[0])

        return self.generic_visit(node)

    @classmethod
    def apply(cls, node: ir.Node) -> ir.Node:
        return cls().visit(node)
