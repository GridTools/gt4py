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

"""A pretty printer for the functional IR.

Inspired by P. Yelland, “A New Approach to Optimal Code Formatting”, 2015
"""

# TODO(tehrengruber): add support for printing the types of itir.Sym, itir.Literal nodes
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Final

from gt4py.eve import NodeTranslator
from gt4py.next.iterator import ir


# replacements for builtin binary operations
BINARY_OPS: Final = {
    "plus": "+",
    "minus": "-",
    "multiplies": "×",  # noqa: RUF001 [ambiguous-unicode-character-string]
    "divides": "/",
    "eq": "==",
    "less": "<",
    "greater": ">",
    "and_": "∧",
    "or_": "∨",  # noqa: RUF001 [ambiguous-unicode-character-string]
}

# replacements for builtin unary operations
UNARY_OPS: Final = {
    "deref": "·",
    "lift": "↑",
    "not_": "¬",
}

# operator precedence
PRECEDENCE: Final = {
    "__lambda__": 0,
    "if_": 1,
    "or_": 2,
    "and_": 3,
    "eq": 4,
    "less": 4,
    "greater": 4,
    "plus": 5,
    "minus": 5,
    "multiplies": 6,
    "divides": 6,
    "deref": 7,
    "not_": 7,
    "lift": 7,
    "tuple_get": 8,
    "__call__": 8,
}


DEFAULT_INDENT: Final = 2
DEFAULT_WIDTH: Final = 100


class PrettyPrinter(NodeTranslator):
    def __init__(self, indent: int = DEFAULT_INDENT, width: int = DEFAULT_WIDTH) -> None:
        super().__init__()
        self.indent: int = indent
        self.width: int = width

    @staticmethod
    def _hmerge(*blocks: list[str]) -> list[str]:
        if not blocks:
            return []

        def impl(a: list[str], b: list[str]) -> list[str]:
            if not a:
                return b
            if not b:
                return a
            spaces = len(a[-1]) * " "
            return a[:-1] + [a[-1] + b[0]] + [spaces + line for line in b[1:]]

        res = blocks[0]
        for b in blocks[1:]:
            res = impl(res, b)
        return res

    @staticmethod
    def _vmerge(*blocks: list[str]) -> list[str]:
        return [s for b in blocks for s in b]

    def _prec_parens(self, block: list[str], prec: int, op_prec: int) -> list[str]:
        if prec > op_prec:
            return self._hmerge(["("], block, [")"])
        return block

    def _indent(self, block: list[str]) -> list[str]:
        return [" " * self.indent + line for line in block]

    def _cost(self, block: list[str]) -> int:
        if not block:
            return 0
        max_line_length = max(len(line) for line in block)
        return (
            # preferring blocks of fewer lines:
            len(block)
            # strongly preferring blocks with lines not longer than self.width:
            + max(max_line_length - self.width, 0) * 100
            # preferring blocks with lines of uniform length:
            + sum(max_line_length - len(line) for line in block)
        )

    def _optimum(self, a: list[str], b: list[str]) -> list[str]:
        return a if self._cost(a) < self._cost(b) else b

    def _hinterleave(
        self, blocks: Sequence[list[str]], sep: str, *, indent: bool = False
    ) -> Iterable[list[str]]:
        if not blocks:
            return blocks
        do_indent = self._indent if indent else lambda x: x
        for block in blocks[:-1]:
            yield do_indent(self._hmerge(block, [sep]))
        yield do_indent(blocks[-1])

    def visit_Sym(self, node: ir.Sym, *, prec: int) -> list[str]:
        return [node.id]

    def visit_Literal(self, node: ir.Literal, *, prec: int) -> list[str]:
        return [str(node.value)]

    def visit_OffsetLiteral(self, node: ir.OffsetLiteral, *, prec: int) -> list[str]:
        return [str(node.value) + "ₒ"]

    def visit_AxisLiteral(self, node: ir.AxisLiteral, *, prec: int) -> list[str]:
        return [str(node.value)]

    def visit_SymRef(self, node: ir.SymRef, *, prec: int) -> list[str]:
        return [node.id]

    def visit_Lambda(self, node: ir.Lambda, *, prec: int) -> list[str]:
        params = self.visit(node.params, prec=0)
        expr = self.visit(node.expr, prec=0)

        start, bridge = ["λ("], [") → "]
        if not params:
            params = self._hmerge(start, bridge)
        else:
            hparams = self._hmerge(start, *self._hinterleave(params, ", "), bridge)
            vparams = self._vmerge(start, *self._hinterleave(params, ",", indent=True), bridge)
            params = self._optimum(hparams, vparams)

        hbody = self._hmerge(params, expr)
        vbody = self._vmerge(params, self._indent(expr))
        return self._prec_parens(self._optimum(hbody, vbody), prec, PRECEDENCE["__lambda__"])

    def visit_FunCall(self, node: ir.FunCall, *, prec: int) -> list[str]:
        if isinstance(node.fun, ir.SymRef):
            fun_name = node.fun.id
            if fun_name in BINARY_OPS and len(node.args) == 2:
                # replacing binary ops: plus(x, y) → x + y etc.
                op = BINARY_OPS[fun_name]
                lhs = self.visit(node.args[0], prec=PRECEDENCE[fun_name])
                rhs = self.visit(node.args[1], prec=PRECEDENCE[fun_name] + 1)
                h = self._hmerge(lhs, [" " + op + " "], rhs)
                v = self._vmerge(lhs, self._hmerge([op + " "], rhs))
                return self._prec_parens(self._optimum(h, v), prec, PRECEDENCE[fun_name])
            if fun_name in UNARY_OPS and len(node.args) == 1:
                # replacing unary ops: deref(x) → ·x etc.
                op = UNARY_OPS[fun_name]
                res = self._hmerge([op], self.visit(node.args[0], prec=PRECEDENCE[fun_name]))
                return self._prec_parens(res, prec, PRECEDENCE[fun_name])
            if fun_name == "tuple_get" and len(node.args) == 2:
                # tuple_get(i, x) → x[i]
                idx, tup = self.visit(node.args, prec=PRECEDENCE[fun_name])
                res = self._hmerge(tup, ["["], idx, ["]"])
                return self._prec_parens(res, prec, PRECEDENCE[fun_name])
            if fun_name == "shift":
                shifts = self.visit(node.args, prec=0)
                return self._hmerge(["⟪"], *self._hinterleave(shifts, ", "), ["⟫"])
            if fun_name == "named_range" and len(node.args) == 3:
                # named_range(dim, start, stop) → dim: [star, stop)
                dim, start, end = self.visit(node.args, prec=0)
                res = self._hmerge(dim, [": ["], start, [", "], end, [")"])
                return self._prec_parens(res, prec, PRECEDENCE["__call__"])
            if fun_name == "cartesian_domain" and len(node.args) >= 1:
                # cartesian_domain(x, y, ...) → c{ x × y × ... } # noqa: RUF003 [ambiguous-unicode-character-comment]
                args = self.visit(node.args, prec=PRECEDENCE["__call__"])
                return self._hmerge(["c⟨ "], *self._hinterleave(args, ", "), [" ⟩"])
            if fun_name == "unstructured_domain" and len(node.args) >= 1:
                # unstructured_domain(x, y, ...) → u{ x × y × ... } # noqa: RUF003 [ambiguous-unicode-character-comment]
                args = self.visit(node.args, prec=PRECEDENCE["__call__"])
                return self._hmerge(["u⟨ "], *self._hinterleave(args, ", "), [" ⟩"])
            if fun_name == "if_" and len(node.args) == 3:
                # if_(x, y, z) → if x then y else z
                ifb, thenb, elseb = self.visit(node.args, prec=PRECEDENCE["if_"])
                hblocks = self._hmerge(["if "], ifb, [" then "], thenb, [" else "], elseb)
                vblocks = self._vmerge(
                    self._hmerge(["if   "], ifb),
                    self._hmerge(["then "], thenb),
                    self._hmerge(["else "], elseb),
                )
                return self._prec_parens(self._optimum(hblocks, vblocks), prec, PRECEDENCE["if_"])

        fun = self.visit(node.fun, prec=PRECEDENCE["__call__"])
        args = self.visit(node.args, prec=0)

        if not args:
            args = [""]
        else:
            hargs = self._hmerge(*self._hinterleave(args, ", "))
            vargs = self._vmerge(*self._hinterleave(args, ","))
            args = self._optimum(hargs, vargs)

        if node.fun == ir.SymRef(id="make_tuple"):
            # make_tuple(args...) → {args...}
            htup = self._hmerge(["{"], args, ["}"])
            vtup = self._vmerge(["{"], self._indent(args), ["}"])
            return self._optimum(htup, vtup)

        hfun = self._hmerge(fun, ["("], args, [")"])
        vfun = self._vmerge(self._hmerge(fun, ["("]), self._indent(args), [")"])
        return self._prec_parens(self._optimum(hfun, vfun), prec, PRECEDENCE["__call__"])

    def visit_FunctionDefinition(self, node: ir.FunctionDefinition, prec: int) -> list[str]:
        assert prec == 0
        params = self.visit(node.params, prec=0)
        expr = self._hmerge(self.visit(node.expr, prec=0), [";"])

        start, bridge = [node.id + " = λ("], [") → "]
        if not params:
            params = self._hmerge(start, bridge)
        else:
            hparams = self._hmerge(start, *self._hinterleave(params, ", "), bridge)
            vparams = self._vmerge(start, *self._hinterleave(params, ",", indent=True), bridge)
            params = self._optimum(hparams, vparams)

        hbody = self._hmerge(params, expr)
        vbody = self._vmerge(params, self._indent(expr))
        return self._optimum(hbody, vbody)

    def visit_StencilClosure(self, node: ir.StencilClosure, *, prec: int) -> list[str]:
        assert prec == 0
        domain = self.visit(node.domain, prec=0)
        stencil = self.visit(node.stencil, prec=0)
        output = self.visit(node.output, prec=0)
        inputs = self.visit(node.inputs, prec=0)

        hinputs = self._hmerge(["("], *self._hinterleave(inputs, ", "), [")"])
        vinputs = self._vmerge(["("], *self._hinterleave(inputs, ",", indent=True), [")"])
        inputs = self._optimum(hinputs, vinputs)

        head = self._hmerge(output, [" ← "])
        foot = self._hmerge(inputs, [" @ "], domain, [";"])

        h = self._hmerge(head, ["("], stencil, [")"], foot)
        v = self._vmerge(
            self._hmerge(head, ["("]),
            self._indent(self._indent(stencil)),
            self._indent(self._hmerge([")"], foot)),
        )
        return self._optimum(h, v)

    def visit_FencilDefinition(self, node: ir.FencilDefinition, *, prec: int) -> list[str]:
        assert prec == 0
        function_definitions = self.visit(node.function_definitions, prec=0)
        closures = self.visit(node.closures, prec=0)
        params = self.visit(node.params, prec=0)

        hparams = self._hmerge([node.id + "("], *self._hinterleave(params, ", "), [") {"])
        vparams = self._vmerge(
            [node.id + "("], *self._hinterleave(params, ",", indent=True), [") {"]
        )
        params = self._optimum(hparams, vparams)

        function_definitions = self._vmerge(*function_definitions)
        closures = self._vmerge(*closures)

        return self._vmerge(
            params, self._indent(function_definitions), self._indent(closures), ["}"]
        )

    @classmethod
    def apply(cls, node: ir.Node, indent: int, width: int) -> str:
        return "\n".join(cls(indent=indent, width=width).visit(node, prec=0))


def pformat(x: ir.Node, indent: int = DEFAULT_INDENT, width: int = DEFAULT_WIDTH) -> str:
    return PrettyPrinter.apply(x, indent, width)


def pprint(x: ir.Node, indent: int = DEFAULT_INDENT, width: int = DEFAULT_WIDTH) -> None:
    print(pformat(x, indent, width))
