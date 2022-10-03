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
from eve import NodeTranslator, traits
from functional.common import GTTypeError
from functional.ffront import common_types as ct, program_ast as past, type_info


def _check_tuple_args(tuple_param: past.TupleExpr):
    """
    Unfold tuple and check that all arguments are of type past.Name.

    For example, if argument is of type past.Subscript, this function will throw an error as both slicing and domain are being applied
    """
    for entry in tuple_param.elts:
        if isinstance(entry, past.TupleExpr):
            check_tuple_args(entry)
        elif not isinstance(entry, past.Name):
            raise GTTypeError("Either only domain or slicing allowed")


def _validate_call_params(new_func: past.Name, new_kwargs: dict):
    """
    Perform checks for domain and output field types.

    Keyword `out` has to be present in function call.

    Domain has to be of type dictionary, including dimensions with values expressed as tuples of 2 numbers.
    """
    if not isinstance(new_func.type, (ct.FieldOperatorType, ct.ScanOperatorType)):
        raise GTTypeError(
            f"Only calls `FieldOperator`s and `ScanOperators` "
            f"allowed in `Program`, but got `{new_func.type}`."
        )

    if "out" not in new_kwargs:
        raise GTTypeError("Missing required keyword argument(s) `out`.")
    if "domain" in new_kwargs:

        if not isinstance(new_kwargs["out"], past.Name) and not isinstance(
            new_kwargs["out"], past.TupleExpr
        ):
            raise GTTypeError("Either only domain or slicing allowed")
        elif isinstance(new_kwargs["out"], past.TupleExpr):
            check_tuple_args(new_kwargs["out"])

        domain_kwarg = new_kwargs["domain"]
        if not isinstance(domain_kwarg, past.Dict):
            raise GTTypeError(
                f"Only Dictionaries allowed in domain, but got `{type(domain_kwarg)}`."
            )

        for domain_keys in domain_kwarg.keys_:
            if not isinstance(domain_keys.type, ct.DimensionType):
                raise GTTypeError(
                    f"Only Dimension allowed in domain dictionary keys, but got `{domain_keys.type}`."
                )
        for domain_values in domain_kwarg.values_:
            if len(domain_values.elts) != 2:
                raise GTTypeError(
                    f"Only 2 values allowed in domain range, but got `{len(domain_values.elts)}`."
                )
            if domain_values.elts[0].type != ct.ScalarType(
                kind=ct.ScalarKind.INT64
            ) or domain_values.elts[1].type != ct.ScalarType(kind=ct.ScalarKind.INT64):
                raise GTTypeError(
                    f"Only integer values allowed in domain range, but got {domain_values.elts[0].type} and {domain_values.elts[1].type}."
                )


class ProgramTypeDeduction(traits.VisitorWithSymbolTableTrait, NodeTranslator):
    @classmethod
    def apply(cls, node: past.Program) -> past.Program:
        return cls().visit(node)

    def visit_Program(self, node: past.Program, **kwargs):
        params = self.visit(node.params, **kwargs)

        definition_type = ct.FunctionType(
            args=[param.type for param in params], kwargs={}, returns=ct.VoidType()
        )

        return past.Program(
            id=self.visit(node.id, **kwargs),
            type=ct.ProgramType(definition=definition_type),
            params=params,
            body=self.visit(node.body, **kwargs),
            captured_vars=self.visit(node.captured_vars, **kwargs),
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
            elts=elts, type=ct.TupleType(types=[el.type for el in elts]), location=node.location
        )

    def visit_Call(self, node: past.Call, **kwargs):
        new_func = self.visit(node.func, **kwargs)
        new_args = self.visit(node.args, **kwargs)
        new_kwargs = self.visit(node.kwargs, **kwargs)

        try:
            validate_call_params(new_func, new_kwargs)
            arg_types = [arg.type for arg in new_args]
            kwarg_types = {
                name: expr.type
                for name, expr in new_kwargs.items()
                if name != "out" and name != "domain"
            }

            type_info.accepts_args(
                new_func.type,
                with_args=arg_types,
                with_kwargs=kwarg_types,
                raise_exception=True,
            )

            return_type = type_info.return_type(
                new_func.type, with_args=arg_types, with_kwargs=kwarg_types
            )
            if return_type != new_kwargs["out"].type:
                raise GTTypeError(
                    f"Expected keyword argument `out` to be of "
                    f"type {return_type}, but got "
                    f"{new_kwargs['out'].type}."
                )
        except GTTypeError as ex:
            raise ProgramTypeError.from_past_node(
                node, msg=f"Invalid call to `{node.func.id}`."
            ) from ex

        return past.Call(
            func=new_func,
            args=new_args,
            kwargs=new_kwargs,
            type=ct.VoidType(),
            location=node.location,
        )

    def visit_Name(self, node: past.Name, **kwargs) -> past.Name:
        symtable = kwargs["symtable"]
        if node.id not in symtable or symtable[node.id].type is None:
            raise ProgramTypeError.from_past_node(
                node, msg=f"Undeclared or untyped symbol `{node.id}`."
            )

        return past.Name(id=node.id, type=symtable[node.id].type, location=node.location)


class ProgramTypeError(GTTypeError):
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
        super().__init__(msg, (filename, lineno, offset, text, end_lineno, end_offset))

    @classmethod
    def from_past_node(
        cls,
        node: past.LocatedNode,
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
