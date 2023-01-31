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

import enum

from gt4py.next.ffront import field_operator_ast as foast


class StmtReturnKind(enum.IntEnum):
    UNCONDITIONAL_RETURN = 0
    NO_RETURN = 2


def deduce_stmt_return_kind(node: foast.Stmt) -> StmtReturnKind:
    """
    Deduce if a statement returns and if so, whether it does unconditionally.

    Example with ``StmtReturnKind.UNCONDITIONAL_RETURN``::

        if cond:
          return 1
        else:
          return 2

    Example with ``StmtReturnKind.NO_RETURN``::

        if cond:
          result = 1
        else:
          result = 2
    """
    if isinstance(node, foast.Return):
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
        raise AssertionError(f"Statements of type `{type(node).__name__}` not understood.")
