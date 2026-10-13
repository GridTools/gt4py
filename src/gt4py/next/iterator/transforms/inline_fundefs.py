# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import graphlib
from typing import Callable

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import symbol_ref_utils


def _sorted_by_dependency(
    function_definitions: list[itir.FunctionDefinition],
) -> list[itir.FunctionDefinition]:
    """Order function definitions such that each one only references its predecessors."""
    fundefs = {str(fundef.id): fundef for fundef in function_definitions}
    dependencies = {
        name: symbol_ref_utils.collect_symbol_refs(fundef.expr, fundefs.keys())
        for name, fundef in fundefs.items()
    }
    return [fundefs[name] for name in graphlib.TopologicalSorter(dependencies).static_order()]


@dataclasses.dataclass(frozen=True)
class _WrapStatementExpressions(PreserveLocationVisitor, NodeTranslator):
    """Replace every expression a statement is made of by `wrap` applied to it."""

    wrap: Callable[[itir.Expr], itir.Expr]

    def visit_SetAt(self, node: itir.SetAt) -> itir.SetAt:
        return itir.SetAt(expr=self.wrap(node.expr), domain=node.domain, target=node.target)

    def visit_IfStmt(self, node: itir.IfStmt) -> itir.IfStmt:
        return itir.IfStmt(
            cond=self.wrap(node.cond),
            true_branch=self.visit(node.true_branch),
            false_branch=self.visit(node.false_branch),
        )


def inline_fundefs(program: itir.Program) -> itir.Program:
    """
    Turn the function definitions of a program into `let` bindings of its statements.

    Every statement expression is wrapped in a `let` binding all function definitions to the
    corresponding lambdas. Since the function definitions thereby become regular bindings, the
    usual scoping rules apply, in particular a lambda parameter of the same name shadows a
    function definition instead of being erroneously replaced by it. Bindings that are unused,
    e.g. because the function definition is never called, are removed by dead code elimination.

    >>> from gt4py.next import common
    >>> from gt4py.next.iterator.ir_utils import ir_makers as im
    >>> fun1 = itir.FunctionDefinition(id="fun1", params=[im.sym("a")], expr=im.deref("a"))
    >>> fun2 = itir.FunctionDefinition(id="fun2", params=[im.sym("a")], expr=im.call("fun1")("a"))
    >>> IDim = common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL)
    >>> program = itir.Program(
    ...     id="testee",
    ...     function_definitions=[fun2, fun1],
    ...     params=[im.sym("inp"), im.sym("out")],
    ...     declarations=[],
    ...     body=[
    ...         itir.SetAt(
    ...             expr=im.call("fun2")("inp"),
    ...             domain=im.domain("cartesian_domain", {IDim: (0, 10)}),
    ...             target=im.ref("out"),
    ...         )
    ...     ],
    ... )
    >>> print(inline_fundefs(program))
    testee(inp, out) {
      out @ c⟨ IDimₕ: [0, 10[ ⟩ ← (λ(fun1) → (λ(fun2) → fun2(inp))(λ(a) → fun1(a)))(λ(a) → ·a);
    }
    """
    if not program.function_definitions:
        return program

    # dependent function definitions are bound further inside, such that they see the function
    # definitions they reference
    bindings = [
        (im.sym(fundef.id), im.lambda_(*fundef.params)(fundef.expr))
        for fundef in _sorted_by_dependency(program.function_definitions)
    ]

    def wrap(expr: itir.Expr) -> itir.Expr:
        for param, value in reversed(bindings):
            expr = im.let(param, value)(expr)
        return expr

    return itir.Program(
        id=program.id,
        function_definitions=[],
        params=program.params,
        declarations=program.declarations,
        body=_WrapStatementExpressions(wrap).visit(program.body),
    )
