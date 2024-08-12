# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import ir


def _is_scan(node: ir.Node):
    return (
        isinstance(node, ir.FunCall)
        and isinstance(node.fun, ir.FunCall)
        and node.fun.fun == ir.SymRef(id="scan")
    )


class ScanEtaReduction(PreserveLocationVisitor, NodeTranslator):
    """Applies eta-reduction-like transformation involving scans.

    Simplifies `λ(x, y) → scan(λ(state, param_y, param_x) → ..., ...)(y, x)` to `scan(λ(state, param_x, param_y) → ..., ...)`.
    Note, unlike the eta reduction, this pass works even if parameters of the lambda and arguments in the call
    of the scanpass are not in the same order: parameters of the scanpass are re-ordered. This limits the pass
    to lambdas, i.e. doesn't apply to builtins and `FunctionDefinition`.
    """

    def visit_Lambda(self, node: ir.Lambda) -> ir.Node:
        if _is_scan(node.expr):
            assert isinstance(node.expr, ir.FunCall)
            if len(node.params) == len(node.expr.args) and all(
                isinstance(a, ir.SymRef) and p.id == a.id
                for p, a in zip(
                    sorted(node.params, key=lambda x: str(x)),
                    sorted(node.expr.args, key=lambda x: str(x)),
                )
            ):
                # node.expr.fun is the unapplied scan
                assert isinstance(node.expr.fun, ir.FunCall)
                original_scanpass = node.expr.fun.args[0]
                assert isinstance(original_scanpass, ir.Lambda)
                new_scanpass_params_idx = []
                args_str = [str(a) for a in node.expr.args]
                for p in (str(p) for p in node.params):
                    new_scanpass_params_idx.append(args_str.index(p))
                new_scanpass_params = [original_scanpass.params[0]] + [
                    original_scanpass.params[i + 1] for i in new_scanpass_params_idx
                ]
                new_scanpass = ir.Lambda(params=new_scanpass_params, expr=original_scanpass.expr)
                return ir.FunCall(
                    fun=ir.SymRef(id="scan"), args=[new_scanpass, *node.expr.fun.args[1:]]
                )

        return self.generic_visit(node)
