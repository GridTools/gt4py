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
from typing import Iterator

from eve import NodeTranslator, SymbolTableTrait
from functional.common import GTTypeError
from functional.ffront import common_types as ct
from functional.ffront import program_ast as past


def check_signature(
    func_type: ct.FunctionType, args: list[ct.SymbolType], kwargs: dict[str, ct.SymbolType]
) -> Iterator[str]:
    """Check if a function can be called given arguments and keyword arguments types.

    All types must be concrete/complete.
    """
    # check positional arguments
    if len(func_type.args) != len(args):
        yield f"Function takes {len(func_type.args)} arguments, but {len(args)} were given."
    for i, (a_arg, b_arg) in enumerate(zip(func_type.args, args)):
        if a_arg != b_arg:
            yield f"Argument {i} expected to be of type {a_arg}, but got {b_arg}."

    # check for missing or extra keyword arguments
    kw_a_m_b = set(func_type.kwargs.keys()).difference(set(kwargs.keys()))
    if len(kw_a_m_b) > 0:
        yield f"Missing required keyword argument(s) `{'`, `'.join(kw_a_m_b)}`."
    kw_b_m_a = set(kwargs.keys()).difference(set(func_type.kwargs.keys()))
    if len(kw_b_m_a) > 0:
        yield f"Got unexpected keyword argument(s) `{'`, `'.join(kw_b_m_a)}`."

    for kwarg in set(func_type.kwargs.keys()) & set(kwargs.keys()):
        if func_type.kwargs[kwarg] != kwargs[kwarg]:
            yield f"Expected keyword argument {kwarg} to be of type {func_type.kwargs[kwarg]}, but got {kwargs[kwarg]}."


class ProgramTypeDeduction(NodeTranslator):
    contexts = (SymbolTableTrait.symtable_merger,)

    @classmethod
    def apply(cls, node: past.Program) -> past.Program:
        return cls().visit(node)

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
        return past.TupleExpr(elts=elts, type=ct.TupleType(types=[el.type for el in elts]))

    def visit_Call(self, node: past.Call, **kwargs):
        func = self.visit(node.func, **kwargs)
        args = self.visit(node.args, **kwargs)
        kwargs = self.visit(node.kwargs, **kwargs)

        func_type = func.type
        # functions returning fields in a program are implicitly converted into stencil closures. Change function
        #  signature accordingly
        if isinstance(func.type.returns, ct.FieldType):
            assert "out" not in func.type.kwargs
            func_type = ct.FunctionType(
                args=func.type.args,
                kwargs={**func.type.kwargs, "out": func.type.returns},
                returns=ct.VoidType(),
            )

        sig_diffs = list(
            check_signature(
                func_type,
                [arg.type for arg in args],
                {name: expr.type for name, expr in kwargs.items()},
            )
        )
        if len(sig_diffs) > 0:
            raise ProgramTypeError(
                f"Invalid call to `{node.func.id}`:\n"
                + ("\n".join([f"  - {diff}" for diff in sig_diffs]))
            )

        return past.Call(
            func=func, args=args, kwargs=kwargs, type=func_type.returns, location=node.location
        )

    def visit_Name(self, node: past.Name, **kwargs) -> past.Name:
        symtable = kwargs["symtable"]
        if node.id not in symtable or symtable[node.id].type is None:
            raise ProgramTypeError.from_foast_node(
                node, msg=f"Undeclared or untyped symbol {node.id}."
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
        msg = "Could not deduce type: " + msg
        super().__init__(msg, (filename, lineno, offset, text, end_lineno, end_offset))

    @classmethod
    def from_foast_node(
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
