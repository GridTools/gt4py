import copy
import warnings
from typing import List, Set

from eve import NodeVisitor
from gtc import gtir


class DefinitiveAssignmentAnalysis(NodeVisitor):
    def visit_IfStmt(self, node: gtir.FieldIfStmt, *, alive_vars: Set[str], **kwargs):
        true_branch_vars = copy.copy(alive_vars)
        false_branch_vars = copy.copy(alive_vars)
        self.visit(node.true_branch, alive_vars=true_branch_vars, **kwargs)
        self.visit(node.false_branch, alive_vars=false_branch_vars, **kwargs)
        alive_vars.update(true_branch_vars & false_branch_vars)

    def visit_ParAssignStmt(self, node: gtir.ParAssignStmt, *, alive_vars: Set[str], **kwargs):
        self.visit(node.right, alive_vars=alive_vars, **kwargs)
        alive_vars.add(node.left.name)

    def visit_FieldAccess(
        self,
        node: gtir.FieldAccess,
        *,
        alive_vars: Set,
        invalid_accesses: List[gtir.FieldAccess],
        **kwargs,
    ):
        if node.name not in alive_vars:
            invalid_accesses.append(node)


def analyse(gtir_stencil_expr: gtir.Stencil):
    """
    Analyse gtir stencil expression for access to undefined symbols.

    A symbol is said to be defined if it was assigned to in the code-path of its usage. In other words: If a symbol is
    defined in both branches of an if-statement it may be used outside. If a symbol is only defined in a single branch
    it may only be used inside that branch unless it was already defined previously (as this is equal to assigning to
    the previous value again). Note that whether a symbols is defined is independent of the actual result of the
    condition.
    """
    invalid_accesses: List[gtir.FieldAccess] = []
    DefinitiveAssignmentAnalysis().visit(
        gtir_stencil_expr,
        alive_vars=set(gtir_stencil_expr.param_names),
        invalid_accesses=invalid_accesses,
    )
    return invalid_accesses


def check(gtir_stencil_expr: gtir.Stencil):
    """
    Execute definitive assignment analysis and warn on errors.

    :param gtir_stencil_expr:
    :return:
    """
    invalid_accesses = analyse(gtir_stencil_expr)
    for invalid_access in invalid_accesses:
        warnings.warn(f"`{invalid_access.name}` may be uninitialized.")

    return gtir_stencil_expr
