# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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
from typing import Optional

import functional.ffront.field_operator_ast as foast
from eve import NodeTranslator, SymbolTableTrait
from functional.common import GTSyntaxError
from functional.ffront import common_types as ct
from functional.ffront.type_info import TypeInfo, is_complete_symbol_type


def are_broadcast_compatible(left: TypeInfo, right: TypeInfo) -> bool:
    """
    Check if ``left`` and ``right`` types are compatible after optional broadcasting.

    A binary field operation between two arguments can proceed and the result is a field.
    on top of the dimensions, also the dtypes must match.

    Examples:
    ---------
    >>> int_scalar_t = TypeInfo(ct.ScalarType(kind=ct.ScalarKind.INT64))
    >>> are_broadcast_compatible(int_scalar_t, int_scalar_t)
    True
    >>> int_field_t = TypeInfo(ct.FieldType(dtype=ct.ScalarType(kind=ct.ScalarKind.INT64),
    ...                         dims=...))
    >>> are_broadcast_compatible(int_field_t, int_scalar_t)
    True

    """
    if left.is_field_type and right.is_field_type:
        return left.type.dims == right.type.dims
    elif left.is_field_type and right.is_scalar:
        return left.type.dtype == right.type
    elif left.is_scalar and left.is_field_type:
        return left.type == right.type.dtype
    elif left.is_scalar and right.is_scalar:
        return left.type == right.type
    return False


def broadcast_typeinfos(left: TypeInfo, right: TypeInfo) -> TypeInfo:
    """
    Decide the result type of a binary operation between arguments of ``left`` and ``right`` type.

    Return None if the two types are not compatible even after broadcasting.

    Examples:
    ---------
    >>> int_scalar_t = TypeInfo(ct.ScalarType(kind=ct.ScalarKind.INT64))
    >>> int_field_t = TypeInfo(ct.FieldType(dtype=ct.ScalarType(kind=ct.ScalarKind.INT64),
    ...                         dims=...))
    >>> assert broadcast_typeinfos(int_field_t, int_scalar_t).type == int_field_t.type

    """
    if not are_broadcast_compatible(left, right):
        return None
    if left.is_scalar and right.is_field_type:
        return right
    return left


class FieldOperatorTypeDeduction(NodeTranslator):
    """
    Deduce and check types of FOAST expressions and symbols.

    Examples:
    ---------
    >>> import ast
    >>> from functional.common import Field
    >>> from functional.ffront.source_utils import SourceDefinition, ClosureRefs
    >>> from functional.ffront.func_to_foast import FieldOperatorParser
    >>> def example(a: "Field[..., float]", b: "Field[..., float]"):
    ...     return a + b

    >>> sdef = SourceDefinition.from_function(example)
    >>> cref = ClosureRefs.from_function(example)
    >>> untyped_fieldop = FieldOperatorParser(
    ...     source=sdef.source, filename=sdef.filename, starting_line=sdef.starting_line, closure_refs=cref, externals_defs={}
    ... ).visit(ast.parse(sdef.source).body[0])
    >>> assert untyped_fieldop.body[0].value.type is None

    >>> typed_fieldop = FieldOperatorTypeDeduction.apply(untyped_fieldop)
    >>> assert typed_fieldop.body[0].value.type == ct.FieldType(dtype=ct.ScalarType(
    ...     kind=ct.ScalarKind.FLOAT64), dims=Ellipsis)
    """

    contexts = (SymbolTableTrait.symtable_merger,)

    @classmethod
    def apply(cls, node: foast.FieldOperator) -> foast.FieldOperator:
        return cls().visit(node)

    def visit_FieldOperator(self, node: foast.FieldOperator, **kwargs) -> foast.FieldOperator:
        return foast.FieldOperator(
            id=node.id,
            params=self.visit(node.params, **kwargs),
            body=self.visit(node.body, **kwargs),
            closure=self.visit(node.closure, **kwargs),
            location=node.location,
        )

    def visit_Name(self, node: foast.Name, **kwargs) -> foast.Name:
        symtable = kwargs["symtable"]
        if node.id not in symtable or symtable[node.id].type is None:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node, msg=f"Undeclared symbol {node.id}"
            )

        symbol = symtable[node.id]
        return foast.Name(id=node.id, type=symbol.type, location=node.location)

    def visit_Assign(self, node: foast.Assign, **kwargs) -> foast.Assign:
        new_value = node.value
        if not is_complete_symbol_type(node.value.type):
            new_value = self.visit(node.value, **kwargs)
        new_target = self.visit(node.target, refine_type=new_value.type, **kwargs)
        return foast.Assign(target=new_target, value=new_value, location=node.location)

    def visit_Symbol(
        self,
        node: foast.Symbol,
        refine_type: Optional[ct.FieldType] = None,
        **kwargs,
    ) -> foast.Symbol:
        symtable = kwargs["symtable"]
        if refine_type:
            if not TypeInfo(node.type).can_be_refined_to(TypeInfo(refine_type)):
                raise FieldOperatorTypeDeductionError.from_foast_node(
                    node,
                    msg=(
                        "type inconsistency: expression was deduced to be "
                        f"of type {refine_type}, instead of the expected type {node.type}"
                    ),
                )
            new_node = foast.Symbol[type(refine_type)](
                id=node.id, type=refine_type, location=node.location
            )
            symtable[new_node.id] = new_node
            return new_node
        return node

    def visit_Subscript(self, node: foast.Subscript, **kwargs) -> foast.Subscript:
        new_value = self.visit(node.value, **kwargs)
        new_type = None
        if kwargs.get("in_shift", False):
            return foast.Subscript(
                value=new_value,
                index=node.index,
                type=ct.OffsetType(),
                location=node.location,
            )
        match new_value.type:
            case ct.TupleType(types=types):
                new_type = types[node.index]
            case _:
                raise FieldOperatorTypeDeductionError.from_foast_node(
                    new_value, msg="Could not deduce type of subscript expression!"
                )

        return foast.Subscript(
            value=new_value, index=node.index, type=new_type, location=node.location
        )

    def visit_BinOp(self, node: foast.BinOp, **kwargs) -> foast.BinOp:
        new_left = self.visit(node.left, **kwargs)
        new_right = self.visit(node.right, **kwargs)
        new_type = self._deduce_binop_type(
            node.op, parent=node, left_type=new_left.type, right_type=new_right.type
        )
        return foast.BinOp(
            op=node.op, left=new_left, right=new_right, location=node.location, type=new_type
        )

    def _deduce_binop_type(
        self,
        op: foast.BinaryOperator,
        *,
        parent: foast.BinOp,
        left_type: ct.SymbolType,
        right_type: ct.SymbolType,
        **kwargs,
    ) -> ct.SymbolType:
        if op in [
            foast.BinaryOperator.ADD,
            foast.BinaryOperator.SUB,
            foast.BinaryOperator.MULT,
            foast.BinaryOperator.DIV,
        ]:
            return self._deduce_arithmetic_binop_type(
                op, parent=parent, left_type=left_type, right_type=right_type, **kwargs
            )
        else:
            return self._deduce_logical_binop_type(
                op, parent=parent, left_type=left_type, right_type=right_type, **kwargs
            )

    def _deduce_arithmetic_binop_type(
        self,
        op: foast.BinaryOperator,
        *,
        parent: foast.BinOp,
        left_type: ct.SymbolType,
        right_type: ct.SymbolType,
        **kwargs,
    ) -> ct.SymbolType:
        left, right = TypeInfo(left_type), TypeInfo(right_type)
        if (
            left.is_arithmetic_compatible
            and right.is_arithmetic_compatible
            and are_broadcast_compatible(left, right)
        ):
            return broadcast_typeinfos(left, right).type
        else:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                parent,
                msg=f"Incompatible type(s) for operator '{op}': {left.type}, {right.type}!",
            )

    def _deduce_logical_binop_type(
        self,
        op: foast.BinaryOperator,
        *,
        parent: foast.BinOp,
        left_type: ct.SymbolType,
        right_type: ct.SymbolType,
        **kwargs,
    ) -> ct.SymbolType:
        left, right = TypeInfo(left_type), TypeInfo(right_type)
        if (
            left.is_logics_compatible
            and right.is_logics_compatible
            and are_broadcast_compatible(left, right)
        ):
            return broadcast_typeinfos(left, right).type
        else:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                parent,
                msg=f"Incompatible type(s) for operator '{op}': {left.type}, {right.type}!",
            )

    def visit_UnaryOp(self, node: foast.UnaryOp, **kwargs) -> foast.UnaryOp:
        new_operand = self.visit(node.operand, **kwargs)
        if not self._is_unaryop_type_compatible(op=node.op, operand_type=new_operand.type):
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Incompatible type for unary operator '{node.op}': {new_operand.type}!",
            )
        return foast.UnaryOp(
            op=node.op, operand=new_operand, location=node.location, type=new_operand.type
        )

    def _is_unaryop_type_compatible(
        self, op: foast.UnaryOperator, operand_type: ct.FieldType
    ) -> bool:
        operand_ti = TypeInfo(operand_type)
        if op in [foast.UnaryOperator.UADD, foast.UnaryOperator.USUB]:
            return operand_ti.is_arithmetic_compatible
        elif op is foast.UnaryOperator.NOT:
            return operand_ti.is_logics_compatible

    def visit_TupleExpr(self, node: foast.TupleExpr, **kwargs) -> foast.TupleExpr:
        new_elts = self.visit(node.elts, **kwargs)
        new_type = ct.TupleType(types=[element.type for element in new_elts])
        return foast.TupleExpr(elts=new_elts, type=new_type, location=node.location)

    def visit_Call(self, node: foast.Call, **kwargs) -> foast.Call:
        new_func = self.visit(node.func, **kwargs)
        if isinstance(new_func.type, ct.FieldType):
            new_args = self.visit(node.args, in_shift=True, **kwargs)
            return foast.Call(func=new_func, args=new_args, location=node.location)
        return foast.Call(
            func=new_func, args=self.visit(node.args, **kwargs), location=node.location
        )


class FieldOperatorTypeDeductionError(GTSyntaxError, SyntaxWarning):
    """Exception for problematic type deductions that originate in user code."""

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
        msg = "Could not deduce type: " + msg
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
