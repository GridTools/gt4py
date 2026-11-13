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


def _dependencies(
    function_definitions: list[itir.FunctionDefinition],
) -> dict[str, list[str]]:
    """Map each function definition to the function definitions it references."""
    fundefs = {str(fundef.id): fundef for fundef in function_definitions}
    return {
        # wrapped in a lambda such that references shadowed by the definition's own parameters
        # are not collected
        name: symbol_ref_utils.collect_symbol_refs(
            im.lambda_(*fundef.params)(fundef.expr), fundefs.keys()
        )
        for name, fundef in fundefs.items()
    }


@dataclasses.dataclass(frozen=True)
class _WrapExpressions(PreserveLocationVisitor, NodeTranslator):
    """Replace every expression a statement or declaration is made of by `wrap` applied to it."""

    wrap: Callable[[itir.Expr], itir.Expr]

    def visit_SetAt(self, node: itir.SetAt) -> itir.SetAt:
        # note: the target only refers to parameters and temporaries, hence it is not wrapped
        return itir.SetAt(
            expr=self.wrap(node.expr), domain=self.wrap(node.domain), target=node.target
        )

    def visit_IfStmt(self, node: itir.IfStmt) -> itir.IfStmt:
        return itir.IfStmt(
            cond=self.wrap(node.cond),
            true_branch=self.visit(node.true_branch),
            false_branch=self.visit(node.false_branch),
        )

    def visit_Temporary(self, node: itir.Temporary) -> itir.Temporary:
        return itir.Temporary(
            id=node.id,
            domain=self.wrap(node.domain) if node.domain is not None else None,
            dtype=node.dtype,
        )


def inline_fundefs(program: itir.Program) -> itir.Program:
    """
    Turn the function definitions of a program into `let` bindings of its statements.

    Every expression of a statement or declaration is wrapped in a `let` binding the function
    definitions it references to the corresponding lambdas. Since the function definitions thereby
    become regular bindings, the usual scoping rules apply, in particular a lambda parameter of the
    same name shadows a function definition instead of being erroneously replaced by it.

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

    fundefs = {str(fundef.id): fundef for fundef in program.function_definitions}
    dependencies = _dependencies(program.function_definitions)
    # a function definition precedes the ones referencing it, such that it is bound further
    # outside and hence visible in their bodies
    order = list(graphlib.TopologicalSorter(dependencies).static_order())

    def wrap(expr: itir.Expr) -> itir.Expr:
        # Only the function definitions that are referenced, directly or through another
        # definition, are bound. Walking `order` backwards visits a definition before all
        # definitions it references, so the transitive closure needs no extra traversal.
        required = set(symbol_ref_utils.collect_symbol_refs(expr, fundefs.keys()))
        for name in reversed(order):
            if name not in required:
                continue
            required.update(dependencies[name])
            fundef = fundefs[name]
            expr = im.let(im.sym(name), im.lambda_(*fundef.params)(fundef.expr))(expr)
        return expr

    wrapper = _WrapExpressions(wrap)
    return itir.Program(
        id=program.id,
        function_definitions=[],
        params=program.params,
        declarations=wrapper.visit(program.declarations),
        body=wrapper.visit(program.body),
    )
