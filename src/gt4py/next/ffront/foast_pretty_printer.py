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

from __future__ import annotations

import enum
import textwrap
from typing import Final, TypeAlias, Union

import gt4py.next.ffront.field_operator_ast as foast
from gt4py.eve.codegen import FormatTemplate as as_fmt, MakoTemplate as as_mako, TemplatedGenerator
from gt4py.next.ffront import dialect_ast_enums
from gt4py.next.type_system import type_specifications as ts


PropertyIdentifier: TypeAlias = Union[type[foast.LocatedNode], tuple[type[foast.LocatedNode], str]]

INDENTATION_PREFIX: Final[str] = "  "


# See https://docs.python.org/3/reference/expressions.html#operator-precedence
# The following list contains all entries from the above link. Operators that
# are not modeled in FOAST are kept for ease of comparability and future
# extension.
PRECEDENCE: Final[dict[PropertyIdentifier, int]] = {
    # Binding or parenthesized expression, list display, dictionary display, set display
    foast.TupleExpr: 18,
    # Subscription, slicing, call, attribute reference
    foast.Subscript: 17,
    foast.Call: 17,
    foast.Attribute: 17,
    # Await expression
    #  not modeled in FOAST
    # Exponentiation
    (foast.BinOp, "power"): 16,
    # Positive, negative, bitwise NOT
    (foast.UnaryOp, "plus"): 14,
    (foast.UnaryOp, "minus"): 14,
    (foast.UnaryOp, "not_"): 14,
    # Multiplication, matrix multiplication, division, floor division, remainder
    (foast.BinOp, "multiplies"): 13,
    (foast.BinOp, "divides"): 13,
    (foast.BinOp, "floordiv"): 13,
    # (foast.BinOp, "remainder"): 13,
    # Addition and subtraction
    (foast.BinOp, "plus"): 12,
    (foast.BinOp, "minus"): 12,
    # Shifts
    #  not modeled in FOAST
    # Bitwise AND
    #  not modeled in FOAST
    # Bitwise XOR
    #  not modeled in FOAST
    # Bitwise OR
    #  not modeled in FOAST
    # Comparisons, including membership tests and identity tests
    (foast.Compare, "less"): 7,
    (foast.Compare, "less_equal"): 7,
    (foast.Compare, "greater"): 7,
    (foast.Compare, "greater_equal"): 7,
    (foast.Compare, "not_eq"): 7,
    (foast.Compare, "eq"): 7,
    # Boolean AND
    (foast.BinOp, "and_"): 5,
    # Boolean OR
    (foast.BinOp, "or_"): 4,
    # Conditional expression
    foast.TernaryExpr: 3,
    # Lambda expression
    #  not modeled in FOAST
    # Assignment expression
    #  not modeled in FOAST
}

RIGHT_GROUPING = {(foast.BinOp, "power"), foast.TernaryExpr}


class Group(enum.IntEnum):
    LEFT = 0
    RIGHT = 1


def _property_identifier(node: foast.LocatedNode) -> PropertyIdentifier:
    if isinstance(node, foast.BinOp):
        return (foast.BinOp, node.op)
    elif isinstance(node, foast.UnaryOp):
        return (foast.UnaryOp, node.op)
    elif isinstance(node, foast.Compare):
        return (foast.Compare, node.op)

    return type(node)


class _PrettyPrinter(TemplatedGenerator):
    @classmethod
    def apply(cls, node: foast.LocatedNode, **kwargs) -> str:  # type: ignore[override]
        node_type_name = type(node).__name__
        if not hasattr(cls, node_type_name) and not hasattr(cls, f"visit_{node_type_name}"):
            raise NotImplementedError(
                f"Pretty printer does not support nodes of type '{node_type_name}'."
            )
        return cls().visit(node, **kwargs)

    Symbol = as_fmt("{id}")

    Name = as_fmt("{id}")

    Constant = as_fmt("{value}")

    Subscript = as_fmt("{value}[{index}]")

    Attribute = as_fmt("{value}.{attr}")

    TupleExpr = as_fmt("({', '.join(elts)}{',' if len(elts)==1 else ''})")

    UnaryOp = as_fmt("{op}{operand}")

    def visit_UnaryOp(self, node: foast.UnaryOp, **kwargs) -> str:
        if node.op is dialect_ast_enums.UnaryOperator.NOT:
            op = "not "
        else:
            op = str(node.op)
        return f"{op}{node.operand}"

    def visit_BinOp(self, node: foast.BinOp, **kwargs) -> str:
        left = self._parenthesize(node.left, node, Group.LEFT)
        right = self._parenthesize(node.right, node, Group.RIGHT)
        return f"{left} {node.op} {right}"

    def visit_Compare(self, node: foast.Compare, **kwargs) -> str:
        left = self._parenthesize(node.left, node, Group.LEFT)
        right = self._parenthesize(node.right, node, Group.RIGHT)
        return f"{left} {node.op} {right}"

    def visit_TernaryExpr(self, node: foast.TernaryExpr, **kwargs) -> str:
        cond = self.visit(node.condition)
        true_expr = self._parenthesize(node.true_expr, node, Group.LEFT)
        false_expr = self._parenthesize(node.false_expr, node, Group.RIGHT)
        return f"{true_expr} if {cond} else {false_expr}"

    def visit_Call(self, node: foast.Call, **kwargs):
        args = self.visit(node.args, **kwargs)
        for k, v in node.kwargs.items():
            args.append(f"{self.visit(k, **kwargs)}={self.visit(v, **kwargs)}")
        return f"{node.func}({', '.join(args)})"

    Assign = as_fmt("{target} = {value}")

    Return = as_fmt("return {value}")

    BlockStmt = as_mako("${'\\n'.join(stmts)}")

    FunctionDefinition = as_mako(
        textwrap.dedent(
            """
            def ${id}(${', '.join(params_annotated)})${return_type}:
            ${indented_body}
            """
        ).strip()
    )

    def visit_FunctionDefinition(self, node: foast.FunctionDefinition, **kwargs):
        params = self.visit(node.params)
        types = [
            str(param.type) if not isinstance(param.type, ts.DeferredType) else None
            for param in node.params
        ]
        params_annotated = [
            f"{param}: {type_}" if type_ else param for param, type_ in zip(params, types)
        ]
        return_type = (
            f" -> {node.type.returns}" if not isinstance(node.type, ts.DeferredType) else ""
        )
        indented_body = textwrap.indent(self.visit(node.body), INDENTATION_PREFIX)
        return self.generic_visit(
            node,
            indented_body=indented_body,
            params_annotated=params_annotated,
            return_type=return_type,
        )

    FieldOperator = as_fmt("@field_operator\n{definition}")

    ScanOperator = as_fmt(
        "@scan_operator(axis={axis}, forward={forward}, init={init})\n{definition}"
    )

    def _precedence(self, node: foast.LocatedNode) -> int:
        prop_id = _property_identifier(node)
        if prop_id in PRECEDENCE:
            return PRECEDENCE[prop_id]
        return max(PRECEDENCE.values()) + 1

    def _grouping(self, node: foast.LocatedNode) -> Group:
        prop_id = _property_identifier(node)
        return Group.RIGHT if prop_id in RIGHT_GROUPING else Group.LEFT

    def _parenthesize(
        self,
        inner_node: foast.LocatedNode,
        outer_node: foast.LocatedNode,
        group: Group | None = None,
    ) -> str:
        inner_precedence = self._precedence(inner_node)
        outer_precedence = self._precedence(outer_node)
        parenthesize = False
        if inner_precedence < outer_precedence:
            parenthesize = True
        elif group is not None and inner_precedence == outer_precedence:
            parenthesize = group != self._grouping(inner_node)
        inner_node_str = self.visit(inner_node)
        return f"({inner_node_str})" if parenthesize else inner_node_str


def pretty_format(node: foast.LocatedNode) -> str:
    """
    Pretty print (to string) an `foast.LocatedNode`.

    >>> from gt4py.next import Field, Dimension, field_operator, float64
    >>> IDim = Dimension("IDim")
    >>> @field_operator
    ... def field_op(a: Field[[IDim], float64]) -> Field[[IDim], float64]:
    ...     return a + 1.0
    >>> print(pretty_format(field_op.foast_node))
    @field_operator
    def field_op(a: Field[[IDim], float64]) -> Field[[IDim], float64]:
      return a + 1.0
    """
    return _PrettyPrinter().apply(node)
