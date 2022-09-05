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
from typing import Optional, cast

import functional.ffront.field_operator_ast as foast
from eve import NodeTranslator, NodeVisitor, traits
from functional.common import DimensionKind, GTSyntaxError, GTTypeError
from functional.ffront import common_types as ct, fbuiltins, type_info


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


class FieldOperatorTypeDeductionCompletnessValidator(NodeVisitor):
    """Validate an FOAST expression is fully typed."""

    @classmethod
    def apply(cls, node: foast.LocatedNode) -> None:
        incomplete_nodes: list[foast.LocatedNode] = []
        cls().visit(node, incomplete_nodes=incomplete_nodes)

        if incomplete_nodes:
            raise AssertionError("FOAST expression is not fully typed.")

    def visit_LocatedNode(
        self, node: foast.LocatedNode, *, incomplete_nodes: list[foast.LocatedNode]
    ):
        self.generic_visit(node, incomplete_nodes=incomplete_nodes)

        if hasattr(node, "type") and not type_info.is_concrete(node.type):
            incomplete_nodes.append(node)


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
        typed_foast_node = cls().visit(node)

        FieldOperatorTypeDeductionCompletnessValidator.apply(typed_foast_node)

        return typed_foast_node

    def visit_FunctionDefinition(self, node: foast.FunctionDefinition, **kwargs):
        new_params = self.visit(node.params, **kwargs)
        new_body = self.visit(node.body, **kwargs)
        assert isinstance(new_body[-1], foast.Return)
        return_type = new_body[-1].value.type
        new_type = ct.FunctionType(
            args=[new_param.type for new_param in new_params], kwargs={}, returns=return_type
        )

        return foast.FunctionDefinition(
            id=node.id,
            params=new_params,
            body=new_body,
            captured_vars=self.visit(node.captured_vars, **kwargs),
            type=new_type,
            location=node.location,
        )

    # TODO(tehrengruber): make sure all scalar arguments are lifted to 0-dim field args
    def visit_FieldOperator(self, node: foast.FieldOperator, **kwargs) -> foast.FieldOperator:
        new_definition = self.visit(node.definition, **kwargs)
        return foast.FieldOperator(
            id=node.id,
            definition=new_definition,
            location=node.location,
            type=ct.FieldOperatorType(definition=new_definition.type),
        )

    def visit_ScanOperator(self, node: foast.ScanOperator, **kwargs) -> foast.ScanOperator:
        new_axis = self.visit(node.axis, **kwargs)
        if not isinstance(new_axis.type, ct.DimensionType):
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Argument `axis` to scan operator `{node.id}` must be a dimension.",
            )
        if not new_axis.type.dim.kind == DimensionKind.VERTICAL:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Argument `axis` to scan operator `{node.id}` must be a vertical dimension.",
            )
        new_forward = self.visit(node.forward, **kwargs)
        if not new_forward.type.kind == ct.ScalarKind.BOOL:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node, msg=f"Argument `forward` to scan operator `{node.id}` must" f"be a boolean."
            )
        new_init = self.visit(node.init, **kwargs)
        if not all(
            type_info.is_arithmetic(type_)
            for type_ in type_info.primitive_constituents(new_init.type)
        ):
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Argument `init` to scan operator `{node.id}` must "
                f"be an arithmetic type or a composite of arithmetic types.",
            )
        new_definition = self.visit(node.definition, **kwargs)
        new_type = ct.ScanOperatorType(
            axis=new_axis.type.dim,
            definition=new_definition.type,
        )
        return foast.ScanOperator(
            id=node.id,
            axis=new_axis,
            forward=new_forward,
            init=new_init,
            definition=new_definition,
            type=new_type,
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
                new_type = types[node.index]  # type: ignore[has-type]  # used to work, now mypy is going berserk for unknown reasons
            case ct.OffsetType(source=source, target=(target1, target2)):
                if not target2.kind == DimensionKind.LOCAL:  # type: ignore[has-type]  # used to work, now mypy is going berserk for unknown reasons
                    raise FieldOperatorTypeDeductionError.from_foast_node(
                        new_value, msg="Second dimension in offset must be a local dimension."
                    )
                new_type = ct.OffsetType(source=source, target=(target1,))  # type: ignore[has-type]  # used to work, now mypy is going berserk for unknown reasons
            case ct.OffsetType(source=source, target=(target,)):
                # for cartesian axes (e.g. I, J) the index of the subscript only
                #  signifies the displacement in the respective dimension,
                #  but does not change the target type.
                if source != target:  # type: ignore[has-type]  # used to work, now mypy is going berserk for unknown reasons
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

        try:
            # transform operands to have bool dtype and use regular promotion
            #  mechanism to handle dimension promotion
            return type_info.promote(boolified_type(left.type), boolified_type(right.type))
        except GTTypeError as ex:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Could not promote `{left.type}` and `{right.type}` to common type"
                f" in call to `{node.op}`.",
            ) from ex

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
                    arg, msg=f"Type {arg.type} can not be used in operator `{node.op}`!"
                )

        left_type = cast(ct.FieldType | ct.ScalarType, left.type)
        right_type = cast(ct.FieldType | ct.ScalarType, right.type)

        if node.op == foast.BinaryOperator.POW:
            return left_type

        try:
            return type_info.promote(left_type, right_type)
        except GTTypeError as ex:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Could not promote `{left_type}` and `{right_type}` to common type"
                f" in call to `{node.op}`.",
            ) from ex

    def _check_operand_dtypes_match(
        self, node: foast.BinOp | foast.Compare, left: foast.Expr, right: foast.Expr
    ) -> None:
        # check dtypes match
        if not type_info.extract_dtype(left.type) == type_info.extract_dtype(right.type):
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Incompatible datatypes in operator `{node.op}`: {left.type} and {right.type}!",
            )

    def visit_UnaryOp(self, node: foast.UnaryOp, **kwargs) -> foast.UnaryOp:
        new_operand = self.visit(node.operand, **kwargs)
        is_compatible = (
            type_info.is_logical if node.op is foast.UnaryOperator.NOT else type_info.is_arithmetic
        )
        if not is_compatible(new_operand.type):
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Incompatible type for unary operator `{node.op}`: `{new_operand.type}`!",
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
        new_args = self.visit(node.args, **kwargs)
        new_kwargs = self.visit(node.kwargs, **kwargs)

        func_type = new_func.type
        arg_types = [arg.type for arg in new_args]
        kwarg_types = {name: arg.type for name, arg in new_kwargs.items()}

        # ensure signature is valid
        try:
            type_info.accepts_args(
                func_type,
                with_args=arg_types,
                with_kwargs=kwarg_types,
                raise_exception=True,
            )
        except GTTypeError as err:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node, msg=f"Invalid argument types in call to `{node.func.id}`!"
            ) from err

        return_type = type_info.return_type(func_type, with_args=arg_types, with_kwargs=kwarg_types)

        new_node = foast.Call(
            func=new_func,
            args=new_args,
            kwargs=new_kwargs,
            location=node.location,
            type=return_type,
        )

        if (
            isinstance(new_func.type, ct.FunctionType)
            and new_func.id in fbuiltins.MATH_BUILTIN_NAMES
        ):
            return self._visit_math_built_in(new_node, **kwargs)
        elif (
            isinstance(new_func.type, ct.FunctionType)
            and not type_info.is_concrete(return_type)
            and new_func.id in fbuiltins.FUN_BUILTIN_NAMES
        ):
            visitor = getattr(self, f"_visit_{new_func.id}")
            return visitor(new_node, **kwargs)

        return new_node

    def _ensure_signature_valid(self, node: foast.Call, **kwargs) -> None:
        try:
            type_info.accepts_args(
                cast(ct.FunctionType, node.func.type),
                with_args=[arg.type for arg in node.args],
                with_kwargs={keyword: arg.type for keyword, arg in node.kwargs.items()},
                raise_exception=True,
            )
        except GTTypeError as err:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node, msg=f"Invalid argument types in call to `{node.func.id}`!"
            ) from err

    def _visit_math_built_in(self, node: foast.Call, **kwargs) -> foast.Call:
        func_name = node.func.id

        # validate arguments
        error_msg_preamble = f"Incompatible argument in call to `{func_name}`."
        error_msg_for_validator = {
            type_info.is_arithmetic: "an arithmetic",
            type_info.is_floating_point: "a floating point",
        }
        if func_name in fbuiltins.UNARY_MATH_NUMBER_BUILTIN_NAMES:
            arg_validator = type_info.is_arithmetic
        elif func_name in fbuiltins.UNARY_MATH_FP_BUILTIN_NAMES:
            arg_validator = type_info.is_floating_point
        elif func_name in fbuiltins.UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES:
            arg_validator = type_info.is_floating_point
        elif func_name in fbuiltins.BINARY_MATH_NUMBER_BUILTIN_NAMES:
            arg_validator = type_info.is_arithmetic
        else:
            raise AssertionError(f"Unknown math builtin `{func_name}`.")

        error_msgs = []
        for i, arg in enumerate(node.args):
            if not arg_validator(arg.type):
                error_msgs.append(
                    f"Expected {i}-th argument to be {error_msg_for_validator[arg_validator]} type, but got `{arg.type}`."
                )
        if error_msgs:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg="\n".join([error_msg_preamble] + [f"  - {error}" for error in error_msgs]),
            )

        if func_name == "power" and all(type_info.is_integral(arg.type) for arg in node.args):
            print(f"Warning: return type of {func_name} might be inconsistent (not implemented).")

        # deduce return type
        return_type: Optional[ct.FieldType | ct.ScalarType] = None
        if (
            func_name
            in fbuiltins.UNARY_MATH_NUMBER_BUILTIN_NAMES + fbuiltins.UNARY_MATH_FP_BUILTIN_NAMES
        ):
            return_type = cast(ct.FieldType | ct.ScalarType, node.args[0].type)
        elif func_name in fbuiltins.UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES:
            return_type = boolified_type(cast(ct.FieldType | ct.ScalarType, node.args[0].type))
        elif func_name in fbuiltins.BINARY_MATH_NUMBER_BUILTIN_NAMES:
            try:
                return_type = type_info.promote(
                    *((cast(ct.FieldType | ct.ScalarType, arg.type)) for arg in node.args)
                )
            except GTTypeError as ex:
                raise FieldOperatorTypeDeductionError.from_foast_node(
                    node, msg=error_msg_preamble
                ) from ex
        else:
            raise AssertionError(f"Unknown math builtin `{func_name}`.")

        return foast.Call(
            func=node.func,
            args=node.args,
            kwargs=node.kwargs,
            location=node.location,
            type=return_type,
        )

    def _visit_reduction(self, node: foast.Call, **kwargs) -> foast.Call:
        field_type = cast(ct.FieldType, node.args[0].type)
        reduction_dim = cast(ct.DimensionType, node.kwargs["axis"].type).dim
        if reduction_dim not in field_type.dims:
            field_dims_str = ", ".join(str(dim) for dim in field_type.dims)
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Incompatible field argument in call to `{node.func.id}`. "
                f"Expected a field with dimension {reduction_dim}, but got "
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

    def _visit_neighbor_sum(self, node: foast.Call, **kwargs) -> foast.Call:
        return self._visit_reduction(node, **kwargs)

    def _visit_max_over(self, node: foast.Call, **kwargs) -> foast.Call:
        return self._visit_reduction(node, **kwargs)

    def _visit_where(self, node: foast.Call, **kwargs) -> foast.Call:
        mask_type = cast(ct.FieldType, node.args[0].type)
        left_type = cast(ct.FieldType, node.args[1].type)
        right_type = cast(ct.FieldType, node.args[2].type)
        if not type_info.is_logical(mask_type):
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Incompatible argument in call to `{node.func.id}`. Expected "
                f"a field with dtype bool, but got `{mask_type}`.",
            )

        try:
            return_type = type_info.promote(left_type, right_type)

            if isinstance(mask_type, ct.FieldType):
                if isinstance(return_type, ct.ScalarType):
                    return_dtype = return_type
                elif isinstance(return_type, ct.FieldType):
                    return_dtype = return_type.dtype
                return_type = type_info.promote(
                    return_type, ct.FieldType(dims=mask_type.dims, dtype=return_dtype)
                )

        except GTTypeError as ex:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Incompatible argument in call to `{node.func.id}`.",
            ) from ex

        return foast.Call(
            func=node.func,
            args=node.args,
            kwargs=node.kwargs,
            type=return_type,
            location=node.location,
        )

    def _visit_broadcast(self, node: foast.Call, **kwargs) -> foast.Call:
        arg_type = cast(ct.FieldType | ct.ScalarType, node.args[0].type)
        broadcast_dims_expr = cast(foast.TupleExpr, node.args[1]).elts

        if any([not (isinstance(elt.type, ct.DimensionType)) for elt in broadcast_dims_expr]):
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Incompatible broadcast dimension type in {node.func.id}. Expected "
                f"all broadcast dimensions to be of type Dimension.",
            )

        broadcast_dims = [cast(ct.DimensionType, elt.type).dim for elt in broadcast_dims_expr]

        if not set((arg_dims := type_info.extract_dims(arg_type))).issubset(set(broadcast_dims)):
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Incompatible broadcast dimensions in {node.func.id}. Expected "
                f"broadcast dimension is missing {set(arg_dims).difference(set(broadcast_dims))}",
            )

        return_type = ct.FieldType(
            dims=broadcast_dims,
            dtype=type_info.extract_dtype(arg_type),
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
