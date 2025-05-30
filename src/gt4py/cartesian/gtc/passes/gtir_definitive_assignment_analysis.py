# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from typing import List, Set

from gt4py import eve
from gt4py.cartesian.gtc import gtir


class DefinitiveAssignmentAnalysis(eve.NodeVisitor):
    """
    Analyse gtir stencil expression for access to undefined symbols.

    A symbol is said to be defined if it was assigned to in the code-path of its usage. In
    other words: If a symbol is defined in both branches of an if-statement it may be used
    outside. If a symbol is only defined in a single branch it may only be used inside that
    branch unless it was already defined previously (as this is equal to assigning to the
    previous value again). Note that whether a symbols is defined is independent of the actual
    result of the condition.
    """

    def visit_IfStmt(self, node: gtir.FieldIfStmt, *, alive_vars: Set[str], **kwargs) -> None:
        true_branch_vars = {*alive_vars}
        false_branch_vars = {*alive_vars}
        self.visit(node.true_branch, alive_vars=true_branch_vars, **kwargs)
        self.visit(node.false_branch, alive_vars=false_branch_vars, **kwargs)
        alive_vars.update(true_branch_vars & false_branch_vars)

    def visit_ParAssignStmt(
        self, node: gtir.ParAssignStmt, *, alive_vars: Set[str], **kwargs
    ) -> None:
        self.visit(node.right, alive_vars=alive_vars, **kwargs)
        alive_vars.add(node.left.name)

    def visit_FieldAccess(
        self,
        node: gtir.FieldAccess,
        *,
        alive_vars: Set[str],
        invalid_accesses: List[gtir.FieldAccess],
        **kwargs,
    ) -> None:
        if node.name not in alive_vars:
            invalid_accesses.append(node)

    @classmethod
    def apply(cls, gtir_stencil_expr: gtir.Stencil) -> List[gtir.FieldAccess]:
        """Execute analysis and return all accesses to undefined symbols."""
        invalid_accesses: List[gtir.FieldAccess] = []
        DefinitiveAssignmentAnalysis().visit(
            gtir_stencil_expr,
            alive_vars=set(gtir_stencil_expr.param_names),
            invalid_accesses=invalid_accesses,
        )
        return invalid_accesses


analyze = DefinitiveAssignmentAnalysis.apply


def check(gtir_stencil_expr: gtir.Stencil) -> gtir.Stencil:
    """Execute definitive assignment analysis and warn on errors."""
    invalid_accesses = analyze(gtir_stencil_expr)
    for invalid_access in invalid_accesses:
        warnings.warn(f"`{invalid_access.name}` may be uninitialized.", stacklevel=2)

    return gtir_stencil_expr
