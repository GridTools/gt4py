# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
            fun: ir.Lambda = node.args[0].fun
            args: list[ir.Expr] = node.args[0].args
            node = im.let(*zip(fun.params, args, strict=True))(im.deref(fun.expr))
        elif cpm.is_call_to(node, "deref") and cpm.is_call_to(node.args[0], "if_"):
            cond, true_branch, false_branch = node.args[0].args
            return im.if_(cond, im.deref(true_branch), im.deref(false_branch))

        return self.generic_visit(node)
