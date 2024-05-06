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

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im


# TODO(tehrengruber): This pass can be generalized to all builtins, e.g.
#  `plus((λ(...) → multiplies(...))(...), ...)` can be transformed into
#  `(λ(...) → plus(multiplies(...), ...))(...)`.


class PropagateDeref(PreserveLocationVisitor, NodeTranslator):
    @classmethod
    def apply(cls, node: ir.Node):
        """
        Propagate calls to deref into lambda functions.

        Transform::

            ·(λ(inner_it) → lift(stencil)(inner_it))(outer_it)

        into::

            (λ(inner_it) → ·(lift(stencil)(inner_it)))(outer_it)

        After this pass calls to `deref` are grouped together (better) with lambda functions
        being propagated outwards. This increases the ability of the lift inliner to remove the
        deref + lift combination.
        """
        return cls().visit(node)

    def visit_FunCall(self, node: ir.FunCall):
        if cpm.is_call_to(node, "deref") and cpm.is_let(node.args[0]):
            fun: ir.Lambda = node.args[0].fun  # type: ignore[assignment]  # ensured by is_let
            args: list[ir.Expr] = node.args[0].args
            node = im.let(*zip(fun.params, args))(im.deref(fun.expr))  # type: ignore[arg-type] # mypy not smart enough
        elif cpm.is_call_to(node, "deref") and cpm.is_call_to(node.args[0], "if_"):
            cond, true_branch, false_branch = node.args[0].args
            return im.if_(cond, im.deref(true_branch), im.deref(false_branch))

        return self.generic_visit(node)
