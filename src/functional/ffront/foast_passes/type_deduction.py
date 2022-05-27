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
import copy
from itertools import permutations
from typing import Optional, cast

import functional.ffront.field_operator_ast as foast
from eve import NodeTranslator, traits
from functional.common import GTSyntaxError, GTTypeError
from functional.ffront import common_types as ct, type_info
from functional.ffront.fbuiltins import FUN_BUILTIN_NAMES


def boolified_type(symbol_type: ct.SymbolType) -> ct.ScalarType | ct.FieldType:
    """
    Create a new symbol type from a symbol type, replacing the data type with ``bool``.

    Examples:
    ---------
    >>> from functional.common import Dimension
    >>> scalar_t = ct.ScalarType(kind=ct.ScalarKind.FLOAT64)
    >>> print(boolified_type(scalar_t))
    bool

    >>> field_t = ct.FieldType(dims=[Dimension(value="I")], dtype=ct.ScalarType(kind=ct.ScalarKind))
    >>> print(boolified_type(field_t))
    Field[[I], dtype=bool]
    """
    shape = None
    if type_info.is_concrete(symbol_type):
        shape = type_info.extract_dtype(symbol_type).shape
    scalar_bool = ct.ScalarType(kind=ct.ScalarKind.BOOL, shape=shape)
    type_class = type_info.type_class(symbol_type)
    if type_class is ct.ScalarType:
        return scalar_bool
    elif type_class is ct.FieldType:
        return ct.FieldType(dtype=scalar_bool, dims=type_info.extract_dims(symbol_type))
    raise GTTypeError(f"Can not boolify type {symbol_type}!")


class FieldOperatorTypeDeduction(traits.VisitorWithSymbolTableTrait, NodeTranslator):
    """
    Deduce and check types of FOAST expressions and symbols.

    Examples:
    ---------
    >>> import ast
    >>> from functional.common import Field
    >>> from functional.ffront.source_utils import SourceDefinition, CapturedVars
    >>> from functional.ffront.func_to_foast import FieldOperatorParser
    >>> def example(a: "Field[..., float]", b: "Field[..., float]"):
    ...     return a + b

    >>> source_definition = SourceDefinition.from_function(example)
    >>> captured_vars = CapturedVars.from_function(example)
    >>> untyped_fieldop = FieldOperatorParser(
    ...     source_definition=source_definition, captured_vars=captured_vars, externals_defs={}
    ... ).visit(ast.parse(source_definition.source).body[0])
    >>> untyped_fieldop.body[0].value.type
    DeferredSymbolType(constraint=None)

    >>> typed_fieldop = FieldOperatorTypeDeduction.apply(untyped_fieldop)
    >>> assert typed_fieldop.body[0].value.type == ct.FieldType(dtype=ct.ScalarType(
    ...     kind=ct.ScalarKind.FLOAT64), dims=Ellipsis)
    """

    @classmethod
    def apply(cls, node: foast.FieldOperator) -> foast.FieldOperator:
        return cls().visit(node)

    def visit_FieldOperator(self, node: foast.FieldOperator, **kwargs) -> foast.FieldOperator:
        return foast.FieldOperator(
            id=node.id,
            params=self.visit(node.params, **kwargs),
            body=self.visit(node.body, **kwargs),
            captured_vars=self.visit(node.captured_vars, **kwargs),
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
        if not type_info.is_concrete(node.value.type):
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
            if not type_info.is_concretizable(node.type, to_type=refine_type):
                raise FieldOperatorTypeDeductionError.from_foast_node(
                    node,
                    msg=(
                        "type inconsistency: expression was deduced to be "
                        f"of type {refine_type}, instead of the expected type {node.type}"
                    ),
                )
            new_node: foast.Symbol = foast.Symbol(
                id=node.id, type=refine_type, location=node.location
            )
            symtable[new_node.id] = new_node
            return new_node
        return node

    def visit_Subscript(self, node: foast.Subscript, **kwargs) -> foast.Subscript:
        new_value = self.visit(node.value, **kwargs)
        new_type: Optional[ct.SymbolType] = None
        match new_value.type:
            case ct.TupleType(types=types):
                new_type = types[node.index]
            case ct.OffsetType(source=source, target=(target1, target2)):
                if not target2.local:
                    raise FieldOperatorTypeDeductionError.from_foast_node(
                        new_value, msg="Second dimension in offset must be a local dimension."
                    )
                new_type = ct.OffsetType(source=source, target=(target1,))
            case ct.OffsetType(source=source, target=(target,)):
                # for cartesian axes (e.g. I, J) the index of the subscript only
                #  signifies the displacement in the respective dimension,
                #  but does not change the target type.
                if source != target:
                    raise FieldOperatorTypeDeductionError.from_foast_node(
                        new_value,
                        msg="Source and target must be equal for offsets with a single target.",
                    )
                new_type = new_value.type
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
        new_type = self._deduce_binop_type(node, left=new_left, right=new_right)
        return foast.BinOp(
            op=node.op, left=new_left, right=new_right, location=node.location, type=new_type
        )

    def visit_Compare(self, node: foast.Compare, **kwargs) -> foast.Compare:
        new_left = self.visit(node.left, **kwargs)
        new_right = self.visit(node.right, **kwargs)
        new_type = self._deduce_compare_type(node, left=new_left, right=new_right)
        return foast.Compare(
            op=node.op, left=new_left, right=new_right, location=node.location, type=new_type
        )

    def _deduce_compare_type(
        self, node: foast.Compare, *, left: foast.Expr, right: foast.Expr, **kwargs
    ) -> Optional[ct.SymbolType]:
        # check both types compatible
        for arg in (left, right):
            if not type_info.is_arithmetic(arg.type):
                raise FieldOperatorTypeDeductionError.from_foast_node(
                    arg, msg=f"Type {arg.type} can not be used in operator '{node.op}'!"
                )

        self._check_operand_dtypes_match(node, left=left, right=right)

        # check dimensions match and broadcast scalars to fields
        for one_type, other_type in permutations([left.type, right.type]):
            if type_info.is_dimensionally_promotable(other_type, one_type):
                return boolified_type(one_type)

        raise FieldOperatorTypeDeductionError.from_foast_node(
            node,
            msg=f"Incompatible types for operator '{node.op}': {left.type} and {right.type}!",
        )

    def _deduce_binop_type(
        self,
        node: foast.BinOp,
        *,
        left: foast.Expr,
        right: foast.Expr,
        **kwargs,
    ) -> Optional[ct.SymbolType]:
        logical_ops = {foast.BinaryOperator.BIT_AND, foast.BinaryOperator.BIT_OR}
        is_compatible = type_info.is_logical if node.op in logical_ops else type_info.is_arithmetic

        # check both types compatible
        for arg in (left, right):
            if not is_compatible(arg.type):
                raise FieldOperatorTypeDeductionError.from_foast_node(
                    arg, msg=f"Type {arg.type} can not be used in operator '{node.op}'!"
                )

        if left.type == right.type:
            return copy.copy(left.type)

        self._check_operand_dtypes_match(node, left=left, right=right)

        # check dimensions match and broadcast scalars to fields
        for one_type, other_type in permutations([left.type, right.type]):
            if type_info.is_dimensionally_promotable(other_type, one_type):
                return copy.copy(one_type)

        # the case of left_type == right_type is already handled above
        # so here they must be incompatible
        raise FieldOperatorTypeDeductionError.from_foast_node(
            node,
            msg=f"Incompatible dimensions in operator '{node.op}': {left.type} and {right.type}!",
        )

    def _check_operand_dtypes_match(
        self, node: foast.BinOp | foast.Compare, left: foast.Expr, right: foast.Expr
    ) -> None:
        # check dtypes match
        if not type_info.extract_dtype(left.type) == type_info.extract_dtype(right.type):
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Incompatible datatypes in operator '{node.op}': {left.type} and {right.type}!",
            )

    def visit_UnaryOp(self, node: foast.UnaryOp, **kwargs) -> foast.UnaryOp:
        new_operand = self.visit(node.operand, **kwargs)
        is_compatible = (
            type_info.is_logical if node.op is foast.UnaryOperator.NOT else type_info.is_arithmetic
        )
        if not is_compatible(new_operand.type):
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Incompatible type for unary operator '{node.op}': {new_operand.type}!",
            )
        return foast.UnaryOp(
            op=node.op, operand=new_operand, location=node.location, type=new_operand.type
        )

    def visit_TupleExpr(self, node: foast.TupleExpr, **kwargs) -> foast.TupleExpr:
        new_elts = self.visit(node.elts, **kwargs)
        new_type = ct.TupleType(types=[element.type for element in new_elts])
        return foast.TupleExpr(elts=new_elts, type=new_type, location=node.location)

    def visit_Call(self, node: foast.Call, **kwargs) -> foast.Call:
        new_func = self.visit(node.func, **kwargs)

        if isinstance(new_func.type, ct.FieldType):
            new_args = self.visit(node.args, **kwargs)
            source_dim = new_args[0].type.source
            target_dims = new_args[0].type.target
            if new_func.type.dims and source_dim not in new_func.type.dims:
                raise FieldOperatorTypeDeductionError.from_foast_node(
                    node,
                    msg=f"Incompatible offset at {new_func.id}: can not shift from {new_args[0].type.source} to {new_func.type.dims[0]}.",
                )
            new_dims = []
            for d in new_func.type.dims:
                if d != source_dim:
                    new_dims.append(d)
                else:
                    new_dims.extend(target_dims)
            new_type = ct.FieldType(dims=new_dims, dtype=new_func.type.dtype)
            return foast.Call(
                func=new_func, args=new_args, kwargs={}, location=node.location, type=new_type
            )
        elif isinstance(new_func.type, ct.FunctionType):
            return_type = new_func.type.returns
            new_node = foast.Call(
                func=new_func,
                args=self.visit(node.args, **kwargs),
                kwargs=self.visit(node.kwargs, **kwargs),
                location=node.location,
                type=return_type,
            )

            self._ensure_signature_valid(new_node, **kwargs)

            # todo(tehrengruber): solve in a more generic way, e.g. using
            #  parametric polymorphism.
            # deduce return type of polymorphic builtins
            if not type_info.is_concrete(return_type) and new_node.func.id in FUN_BUILTIN_NAMES:
                visitor = getattr(self, f"_visit_{new_node.func.id}")
                return visitor(new_node, **kwargs)

            return new_node

        raise FieldOperatorTypeDeductionError.from_foast_node(
            node,
            msg=f"Objects of type '{new_func.type}' are not callable.",
        )

    def _ensure_signature_valid(self, node: foast.Call, **kwargs) -> None:
        try:
            type_info.is_callable(
                cast(ct.FunctionType, node.func.type),
                with_args=[arg.type for arg in node.args],
                with_kwargs={keyword: arg.type for keyword, arg in node.kwargs.items()},
                raise_exception=True,
            )
        except GTTypeError as err:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node, msg=f"Invalid argument types in call to '{node.func.id}'!"
            ) from err

    def _visit_neighbor_sum(self, node: foast.Call, **kwargs) -> foast.Call:
        field_type = cast(ct.FieldType, node.args[0].type)
        reduction_dim = cast(ct.DimensionType, node.kwargs["axis"].type).dim
        if reduction_dim not in field_type.dims:
            field_dims_str = ", ".join(str(dim) for dim in field_type.dims)
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Incompatible field argument in {node.func.id}. Expected "
                f"a field with dimension {reduction_dim}, but got "
                f"{field_dims_str}.",
            )
        return_type = ct.FieldType(
            dims=[dim for dim in field_type.dims if dim != reduction_dim],
            dtype=field_type.dtype,
        )

        return foast.Call(
            func=node.func,
            args=node.args,
            kwargs=node.kwargs,
            location=node.location,
            type=return_type,
        )

    def _visit_broadcast(self, node: foast.Call, **kwargs) -> foast.Call:
        field_type = cast(ct.FieldType, node.args[0].type)
        broadcast_dims_expr = node.args[1].elts

        if any([not (isinstance(elt.type, ct.DimensionType)) for elt in broadcast_dims_expr]):
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Incompatible broadcast dimension type in {node.func.id}. Expected "
                f"all broadcast dimensions to be of type Dimension.",
            )

        broadcast_dims = [cast(ct.DimensionType, elt.type).dim for elt in broadcast_dims_expr]

        if not set(field_type.dims).issubset(set(broadcast_dims)):
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Incompatible broadcast dimensions in {node.func.id}. Expected "
                f"broadcast dimension is missing {set(field_type.dims).difference(set(broadcast_dims))}",
            )

        return_type = ct.FieldType(
            dims=broadcast_dims,
            dtype=field_type.dtype,
        )

        return foast.Call(
            func=node.func,
            args=node.args,
            kwargs=node.kwargs,
            location=node.location,
            type=return_type,
        )

    def visit_Constant(self, node: foast.Constant, **kwargs) -> foast.Constant:
        if not node.type:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node, msg=f"Found a literal with unrecognized type {node.type}."
            )
        return node


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
