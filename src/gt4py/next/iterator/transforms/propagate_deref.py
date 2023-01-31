# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

from gt4py.eve import NodeTranslator
from gt4py.eve.pattern_matching import ObjectPattern as P
from gt4py.next.iterator import ir


# TODO(tehrengruber): This pass can be generalized to all builtins, e.g.
#  `plus((λ(...) → multiplies(...))(...), ...)` can be transformed into
#  `(λ(...) → plus(multiplies(...), ...))(...)`.


class PropagateDeref(NodeTranslator):
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
        if P(ir.FunCall, fun=ir.SymRef(id="deref"), args=[P(ir.FunCall, fun=P(ir.Lambda))]).match(
            node
        ):
            builtin = node.fun
            lambda_fun: ir.Lambda = node.args[0].fun  # type: ignore[attr-defined] # invariant ensured by pattern match above
            lambda_args: list[ir.Expr] = node.args[0].args  # type: ignore[attr-defined] # invariant ensured by pattern match above
            node = ir.FunCall(
                fun=ir.Lambda(
                    params=lambda_fun.params,
                    expr=ir.FunCall(fun=builtin, args=[lambda_fun.expr]),
                ),
                args=lambda_args,
            )
        return self.generic_visit(node)
