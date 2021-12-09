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
import warnings
from typing import Optional, TypeGuard

import functional.ffront.field_operator_ast as foast
from eve import NodeTranslator, SymbolTableTrait
from functional.common import GTSyntaxError


def is_complete_symbol_type(fo_type: foast.SymbolType) -> TypeGuard[foast.SymbolType]:
    match fo_type:
        case None:
            return False
        case foast.DeferredSymbolType():
            return False
        case foast.SymbolType():
            return True
    return False


def check_type_refinement(node: foast.Expr, new: foast.DataType) -> None:
    old = node.type
    if old is None:
        return
    if is_complete_symbol_type(old):
        if old != new:
            warnings.warn(
                FieldOperatorTypeDeductionError.from_foast_node(
                    node,
                    msg=(
                        "type inconsistency: expression was deduced to be "
                        f"of type {new}, instead of the expected type {old}"
                    ),
                )
            )
    elif isinstance(old, foast.DeferredSymbolType) and old.constraint is not None:
        if not isinstance(new, old.constraint) or isinstance(new, foast.DeferredSymbolType):
            warnings.warn(
                FieldOperatorTypeDeductionError.from_foast_node(
                    node,
                    msg=(
                        "type inconsistency: expression was deduced to be "
                        f"of type {new}, instead a {type(old)} type was expected."
                    ),
                )
            )


class FieldOperatorTypeDeduction(NodeTranslator):
    """Deduce and check types of FOAST expressions and symbols."""

    contexts = (SymbolTableTrait.symtable_merger,)

    @classmethod
    def apply(cls, node: foast.FieldOperator) -> foast.FieldOperator:
        return cls().visit(node)

    def visit_FieldOperator(self, node: foast.FieldOperator, **kwargs) -> foast.FieldOperator:
        return foast.FieldOperator(
            id=node.id,
            params=self.visit(node.params, **kwargs),
            body=self.visit(node.body, **kwargs),
            location=node.location,
        )

    def visit_Name(self, node: foast.Name, **kwargs) -> foast.Name:
        symtable = kwargs["symtable"]
        if node.id not in symtable or symtable[node.id].type is None:
            warnings.warn(  # TODO(ricoh): raise this instead (requires externals)
                FieldOperatorTypeDeductionError.from_foast_node(
                    node, msg=f"Undeclared symbol {node.id}"
                )
            )
            return node

        symbol = symtable[node.id]
        return foast.Name(id=node.id, type=symbol.type, location=node.location)

    def visit_Assign(self, node: foast.Assign, **kwargs) -> foast.Assign:
        new_value = node.value
        if not is_complete_symbol_type(node.value.type):
            new_value = self.visit(node.value, **kwargs)
        new_target = self.visit(node.target, refine_type=new_value.type, **kwargs)
        return foast.Assign(target=new_target, value=new_value, location=node.location)

    def visit_FieldSymbol(
        self, node: foast.FieldSymbol, refine_type: Optional[foast.FieldType] = None, **kwargs
    ) -> foast.FieldSymbol:
        symtable = kwargs["symtable"]
        if refine_type:
            check_type_refinement(node, refine_type)
            new_node = foast.FieldSymbol(id=node.id, type=refine_type, location=node.location)
            symtable[new_node.id] = new_node
            return new_node
        return node

    def visit_TupleSymbol(
        self, node: foast.TupleSymbol, refine_type: Optional[foast.TupleType] = None, **kwargs
    ) -> foast.TupleSymbol:
        symtable = kwargs["symtable"]
        if refine_type:
            check_type_refinement(node, refine_type)
            new_node = foast.TupleSymbol(id=node.id, type=refine_type, location=node.location)
            symtable[new_node.id] = new_node
            return new_node
        return node

    def visit_Subscript(self, node: foast.Subscript, **kwargs) -> foast.Subscript:
        new_value = self.visit(node.value, **kwargs)
        new_type = None
        if kwargs.get("in_shift", False):
            return foast.Subscript(
                value=new_value, index=node.index, type=foast.OffsetType(), location=node.location
            )
        match new_value.type:
            case foast.TupleType(types=types) | foast.FunctionType(
                returns=foast.TupleType(types=types)
            ):
                new_type = types[node.index]
            case _:
                raise FieldOperatorTypeDeductionError.from_foast_node(
                    new_value, msg="Could not deduce type of subscript expression!"
                )

        return foast.Subscript(
            value=new_value, index=node.index, type=new_type, location=node.location
        )

    def visit_TupleExpr(self, node: foast.TupleExpr, **kwargs) -> foast.TupleExpr:
        new_elts = self.visit(node.elts, **kwargs)
        new_type = foast.TupleType(types=[element.type for element in new_elts])
        return foast.TupleExpr(elts=new_elts, type=new_type, location=node.location)

    def visit_Call(self, node: foast.Call, **kwargs) -> foast.Call:
        new_func = self.visit(node.func, **kwargs)
        if isinstance(new_func.type, foast.FieldType):
            new_args = self.visit(node.args, in_shift=True, **kwargs)
            return foast.Call(func=new_func, args=new_args, location=node.location)
        return foast.Call(
            func=new_func, args=self.visit(node.args, **kwargs), location=node.location
        )


class FieldOperatorTypeDeductionError(GTSyntaxError, SyntaxWarning):
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
