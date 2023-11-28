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
from typing import TypeGuard

from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im


def is_applied_lift(arg: itir.Node) -> TypeGuard[itir.FunCall]:
    """Match expressions of the form `lift(λ(...) → ...)(...)`."""
    return (
        isinstance(arg, itir.FunCall)
        and isinstance(arg.fun, itir.FunCall)
        and isinstance(arg.fun.fun, itir.SymRef)
        and arg.fun.fun.id == "lift"
    )


def is_let(node: itir.Node) -> bool:
    """Match expression of the form `(λ(...) → ...)(...)`"""
    return isinstance(node, itir.FunCall) and isinstance(node.fun, itir.Lambda)


def is_if_call(node: itir.Expr):
    """Match expression of the form `if_(cond, true_branch, false_branch)`"""
    return isinstance(node, itir.FunCall) and node.fun == im.ref("if_")
