# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import enum

from gt4py.next.ffront import field_operator_ast as foast


class StmtReturnKind(enum.IntEnum):
    UNCONDITIONAL_RETURN = 0
    CONDITIONAL_RETURN = 1
    NO_RETURN = 2


def deduce_stmt_return_kind(node: foast.Stmt) -> StmtReturnKind:
    """
    Deduce if a statement returns and if so, whether it does unconditionally.

    Example with ``StmtReturnKind.UNCONDITIONAL_RETURN``::

        if cond:
            return 1
        else:
            return 2

    Example with ``StmtReturnKind.CONDITIONAL_RETURN``::

        if cond:
            return 1
        else:
            result = 2

    Example with ``StmtReturnKind.NO_RETURN``::

        if cond:
            result = 1
        else:
            result = 2
    """
    if isinstance(node, foast.IfStmt):
        return_kinds = (
            deduce_stmt_return_kind(node.true_branch),
            deduce_stmt_return_kind(node.false_branch),
        )
        if all(return_kind is StmtReturnKind.UNCONDITIONAL_RETURN for return_kind in return_kinds):
            return StmtReturnKind.UNCONDITIONAL_RETURN
        elif any(
            return_kind in (StmtReturnKind.UNCONDITIONAL_RETURN, StmtReturnKind.CONDITIONAL_RETURN)
            for return_kind in return_kinds
        ):
            return StmtReturnKind.CONDITIONAL_RETURN
        assert all(return_kind is StmtReturnKind.NO_RETURN for return_kind in return_kinds)
        return StmtReturnKind.NO_RETURN
    elif isinstance(node, foast.Return):
        return StmtReturnKind.UNCONDITIONAL_RETURN
    elif isinstance(node, foast.BlockStmt):
        for stmt in node.stmts:
            return_kind = deduce_stmt_return_kind(stmt)
            if return_kind != StmtReturnKind.NO_RETURN:
                return return_kind
        return StmtReturnKind.NO_RETURN
    elif isinstance(node, (foast.Assign, foast.TupleTargetAssign)):
        return StmtReturnKind.NO_RETURN
    else:
        raise AssertionError(f"Statements of type '{type(node).__name__}' not understood.")
