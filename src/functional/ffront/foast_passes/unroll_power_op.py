from __future__ import annotations

from typing import cast

import eve
from functional.ffront import common_types as ct, field_operator_ast as foast, type_info
from functional.ffront.foast_passes.type_deduction import FieldOperatorTypeDeductionError


class UnrollPowerOp(eve.NodeTranslator):
    @classmethod
    def apply(cls, node: foast.LocatedNode):
        return cls().visit(node)

    def visit_BinOp(self, node: foast.BinOp) -> foast.BinOp:
        if node.op == foast.BinaryOperator.POW:
            if not type(node.right) is foast.Constant:
                raise FieldOperatorTypeDeductionError.from_foast_node(
                    node,
                    msg=f"'Power value '{type(node.right)}' not supported. Only integer values greater than zero allowed in the power operation",
                )
            if type_info.extract_dtype(cast(ct.ScalarType, node.right.type)).kind not in [
                ct.ScalarKind.INT32,
                ct.ScalarKind.INT64,
            ]:
                raise FieldOperatorTypeDeductionError.from_foast_node(
                    node, msg="Only integer values allowed in the power operation"
                )
            if int(node.right.value) == 0:
                raise FieldOperatorTypeDeductionError.from_foast_node(
                    node,
                    msg=f"'Power value '{type(node.right)}' not supported. Only integer values greater than zero allowed in the power operation",
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
