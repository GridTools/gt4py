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

from typing import Optional, cast

from gt4py.eve import NodeTranslator, traits
from gt4py.next import errors
from gt4py.next.ffront import (
    dialect_ast_enums,
    program_ast as past,
    type_specifications as ts_ffront,
)
from gt4py.next.type_system import type_info, type_specifications as ts


def _ensure_no_sliced_field(entry: past.Expr):
    """
    Check that all arguments are of type past.Name or past.TupleExpr.

    In the latter case, unfold tuple.

    For example, if argument is of type past.Subscript, this function will throw an error as both slicing and domain are being applied
    """
    if not isinstance(entry, past.Name) and not isinstance(entry, past.TupleExpr):
        raise ValueError("Either only domain or slicing allowed.")
    elif isinstance(entry, past.TupleExpr):
        for param in entry.elts:
            _ensure_no_sliced_field(param)


def _is_integral_scalar(expr: past.Expr) -> bool:
    """Check that expression is an integral scalar."""
    return isinstance(expr.type, ts.ScalarType) and type_info.is_integral(expr.type)


def _validate_operator_call(new_func: past.Name, new_kwargs: dict):
    """
    Perform checks for domain and output field types.

    Keyword `out` has to be present in function call.

    Domain has to be of type dictionary, including dimensions with values expressed as tuples of 2 numbers.
    """
    if not isinstance(
        new_func.type,
        (ts_ffront.FieldOperatorType, ts_ffront.ScanOperatorType),
    ):
        raise ValueError(
            f"Only calls to 'FieldOperators' and 'ScanOperators' "
            f"allowed in 'Program', got '{new_func.type}'."
        )

    if "out" not in new_kwargs:
        raise ValueError("Missing required keyword argument 'out'.")
    if "domain" in new_kwargs:
        _ensure_no_sliced_field(new_kwargs["out"])

        domain_kwarg = new_kwargs["domain"]
        if not isinstance(domain_kwarg, past.Dict):
            raise ValueError(f"Only Dictionaries allowed in 'domain', got '{type(domain_kwarg)}'.")

        if len(domain_kwarg.values_) == 0 and len(domain_kwarg.keys_) == 0:
            raise ValueError("Empty domain not allowed.")

        for dim in domain_kwarg.keys_:
            if not isinstance(dim.type, ts.DimensionType):
                raise ValueError(
                    f"Only 'Dimension' allowed in domain dictionary keys, got '{dim}' which is of type '{dim.type}'."
                )
        for domain_values in domain_kwarg.values_:
            if len(domain_values.elts) != 2:
                raise ValueError(
                    f"Only 2 values allowed in domain range, got {len(domain_values.elts)}."
                )
            if not _is_integral_scalar(domain_values.elts[0]) or not _is_integral_scalar(
                domain_values.elts[1]
            ):
                raise ValueError(
                    f"Only integer values allowed in domain range, got '{domain_values.elts[0].type}' and '{domain_values.elts[1].type}'."
                )


class ProgramTypeDeduction(traits.VisitorWithSymbolTableTrait, NodeTranslator):
    @classmethod
    def apply(cls, node: past.Program) -> past.Program:
        return cls().visit(node)

    def visit_Program(self, node: past.Program, **kwargs):
        params = self.visit(node.params, **kwargs)

        definition_type = ts.FunctionType(
            pos_only_args=[],
            pos_or_kw_args={str(param.id): param.type for param in params},
            kw_only_args={},
            returns=ts.VoidType(),
        )
        return past.Program(
            id=self.visit(node.id, **kwargs),
            type=ts_ffront.ProgramType(definition=definition_type),
            params=params,
            body=self.visit(node.body, **kwargs),
            closure_vars=self.visit(node.closure_vars, **kwargs),
            location=node.location,
        )

    def visit_Subscript(self, node: past.Subscript, **kwargs):
        value = self.visit(node.value, **kwargs)
        return past.Subscript(
            value=value,
            slice_=self.visit(node.slice_, **kwargs),
            type=value.type,
            location=node.location,
        )

    def visit_TupleExpr(self, node: past.TupleExpr, **kwargs):
        elts = self.visit(node.elts, **kwargs)
        return past.TupleExpr(
            elts=elts, type=ts.TupleType(types=[el.type for el in elts]), location=node.location
        )

    def _deduce_binop_type(
        self,
        node: past.BinOp,
        *,
        left: past.Expr,
        right: past.Expr,
        **kwargs,
    ) -> Optional[ts.TypeSpec]:
        logical_ops = {
            dialect_ast_enums.BinaryOperator.BIT_AND,
            dialect_ast_enums.BinaryOperator.BIT_OR,
        }
        is_compatible = type_info.is_logical if node.op in logical_ops else type_info.is_arithmetic

        # check both types compatible
        for arg in (left, right):
            if not isinstance(arg.type, ts.ScalarType) or not is_compatible(arg.type):
                raise errors.DSLError(
                    arg.location, f"Type '{arg.type}' can not be used in operator '{node.op}'."
                )

        left_type = cast(ts.ScalarType, left.type)
        right_type = cast(ts.ScalarType, right.type)

        if node.op == dialect_ast_enums.BinaryOperator.POW:
            return left_type

        if node.op == dialect_ast_enums.BinaryOperator.MOD and not type_info.is_integral(
            right_type
        ):
            raise errors.DSLError(
                arg.location,
                f"Type '{right_type}' can not be used in operator '{node.op}', it only accepts 'int'.",
            )

        try:
            return type_info.promote(left_type, right_type)
        except ValueError as ex:
            raise errors.DSLError(
                node.location,
                f"Could not promote '{left_type}' and '{right_type}' to common type"
                f" in call to '{node.op}'.",
            ) from ex

    def visit_BinOp(self, node: past.BinOp, **kwargs) -> past.BinOp:
        new_left = self.visit(node.left, **kwargs)
        new_right = self.visit(node.right, **kwargs)
        new_type = self._deduce_binop_type(node, left=new_left, right=new_right)
        return past.BinOp(
            op=node.op, left=new_left, right=new_right, location=node.location, type=new_type
        )

    def visit_Call(self, node: past.Call, **kwargs):
        new_func = self.visit(node.func, **kwargs)
        new_args = self.visit(node.args, **kwargs)
        new_kwargs = self.visit(node.kwargs, **kwargs)

        try:
            is_operator = isinstance(
                new_func.type, (ts_ffront.FieldOperatorType, ts_ffront.ScanOperatorType)
            )
            if is_operator:
                _validate_operator_call(new_func, new_kwargs)
            arg_types = [arg.type for arg in new_args]
            kwarg_types = {
                name: expr.type
                for name, expr in new_kwargs.items()
                if not (is_operator and name in ["out", "domain"])
            }

            type_info.accepts_args(
                new_func.type,
                with_args=arg_types,
                with_kwargs=kwarg_types,
                raise_exception=True,
            )
            return_type = ts.VoidType()
            if is_operator:
                operator_return_type = type_info.return_type(
                    new_func.type, with_args=arg_types, with_kwargs=kwarg_types
                )
                if operator_return_type != new_kwargs["out"].type:
                    raise ValueError(
                        "Expected keyword argument 'out' to be of "
                        f"type '{operator_return_type}', got "
                        f"'{new_kwargs['out'].type}'."
                    )
            elif new_func.id in ["minimum", "maximum"]:
                if new_args[0].type != new_args[1].type:
                    raise ValueError(
                        f"First and second argument in '{new_func.id}' must be of the same type."
                        f"Got '{new_args[0].type}' and '{new_args[1].type}'."
                    )
                return_type = new_args[0].type
            else:
                raise AssertionError(
                    "Only calls to 'FieldOperator', 'ScanOperator' or 'minimum' and 'maximum' builtins allowed."
                )

        except ValueError as ex:
            raise errors.DSLError(node.location, f"Invalid call to '{node.func.id}'.\n{ex}") from ex

        return past.Call(
            func=new_func,
            args=new_args,
            kwargs=new_kwargs,
            type=return_type,
            location=node.location,
        )

    def visit_Name(self, node: past.Name, **kwargs) -> past.Name:
        symtable = kwargs["symtable"]
        if node.id not in symtable or symtable[node.id].type is None:
            raise errors.DSLError(node.location, f"Undeclared or untyped symbol '{node.id}'.")

        return past.Name(id=node.id, type=symtable[node.id].type, location=node.location)
