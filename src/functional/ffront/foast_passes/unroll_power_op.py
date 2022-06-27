from __future__ import annotations

from typing import cast

import eve
from functional.common import GTSyntaxError
from functional.ffront import common_types as ct, field_operator_ast as foast, type_info


class UnrollPowerOp(eve.NodeTranslator):
    @classmethod
    def apply(cls, node: foast.FieldOperator) -> foast.FieldOperator:
        return cls().visit(node)

    def visit_BinOp(self, node: foast.BinOp) -> foast.BinOp:
        if node.op == foast.BinaryOperator.POW:
            if not type(node.right) is foast.Constant:
                raise FieldOperatorPowerError.from_foast_node(
                    node,
                    msg=f"Exponent must be a constant value, but got `{type(node.right)}`.",
                )
            if (
                type_info.extract_dtype(cast(ct.ScalarType, node.right.type)).kind
                not in [
                    ct.ScalarKind.INT32,
                    ct.ScalarKind.INT64,
                ]
                or int(node.right.value) == 0
            ):
                raise FieldOperatorPowerError.from_foast_node(
                    node,
                    msg="Only integer values greater than zero allowed in the power operation.",
                )

            new_left = self.visit(node.left)
            unrolled_expr = self.visit(node.left)
            for _i in range(int(node.right.value) - 1):
                unrolled_expr = foast.BinOp(
                    left=unrolled_expr,
                    right=new_left,
                    op=foast.BinaryOperator.MULT,
                    location=node.location,
                    type=node.type,
                )
            return unrolled_expr

        return self.generic_visit(node)


class FieldOperatorPowerError(GTSyntaxError, SyntaxWarning):
    """Exceptions for unsupported power operators."""

    def __init__(
        self,
        msg="",
        *,
        lineno=0,
        offset=0,
        filename=None,
        end_lineno=None,
        end_offset=None,
        text=None,
    ):
        super().__init__(msg, (filename, lineno, offset, text, end_lineno, end_offset))

    @classmethod
    def from_foast_node(
        cls,
        node: foast.LocatedNode,
        *,
        msg: str = "",
    ):
        return cls(
            msg,
            lineno=node.location.line,
            offset=node.location.column,
            filename=node.location.source,
            end_lineno=node.location.end_line,
            end_offset=node.location.end_column,
        )
