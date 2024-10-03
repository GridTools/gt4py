# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from typing import TypeGuard

from gt4py.next.iterator import ir as itir


def is_applied_lift(arg: itir.Node) -> TypeGuard[itir.FunCall]:
    """Match expressions of the form `lift(λ(...) → ...)(...)`."""
    return (
        isinstance(arg, itir.FunCall)
        and isinstance(arg.fun, itir.FunCall)
        and isinstance(arg.fun.fun, itir.SymRef)
        and arg.fun.fun.id == "lift"
    )


def is_applied_map(arg: itir.Node) -> TypeGuard[itir.FunCall]:
    """Match expressions of the form `map(λ(...) → ...)(...)`."""
    return (
        isinstance(arg, itir.FunCall)
        and isinstance(arg.fun, itir.FunCall)
        and isinstance(arg.fun.fun, itir.SymRef)
        and arg.fun.fun.id == "map_"
    )


def is_applied_reduce(arg: itir.Node) -> TypeGuard[itir.FunCall]:
    """Match expressions of the form `reduce(λ(...) → ...)(...)`."""
    return (
        isinstance(arg, itir.FunCall)
        and isinstance(arg.fun, itir.FunCall)
        and isinstance(arg.fun.fun, itir.SymRef)
        and arg.fun.fun.id == "reduce"
    )


def is_applied_shift(arg: itir.Node) -> TypeGuard[itir.FunCall]:
    """Match expressions of the form `shift(λ(...) → ...)(...)`."""
    return (
        isinstance(arg, itir.FunCall)
        and isinstance(arg.fun, itir.FunCall)
        and isinstance(arg.fun.fun, itir.SymRef)
        and arg.fun.fun.id == "shift"
    )


def is_applied_as_fieldop(arg: itir.Node) -> TypeGuard[itir.FunCall]:
    """Match expressions of the form `as_fieldop(stencil)(*args)`."""
    return isinstance(arg, itir.FunCall) and is_call_to(arg.fun, "as_fieldop")


def is_let(node: itir.Node) -> TypeGuard[itir.FunCall]:
    """Match expression of the form `(λ(...) → ...)(...)`."""
    return isinstance(node, itir.FunCall) and isinstance(node.fun, itir.Lambda)


def is_call_to(node: itir.Node, fun: str | Iterable[str]) -> TypeGuard[itir.FunCall]:
    """
    Match call expression to a given function.

    >>> from gt4py.next.iterator.ir_utils import ir_makers as im
    >>> node = im.call("plus")(1, 2)
    >>> is_call_to(node, "plus")
    True
    >>> is_call_to(node, "minus")
    False
    >>> is_call_to(node, ("plus", "minus"))
    True
    """
    if isinstance(fun, (list, tuple, set, Iterable)) and not isinstance(fun, str):
        return any((is_call_to(node, f) for f in fun))
    return (
        isinstance(node, itir.FunCall) and isinstance(node.fun, itir.SymRef) and node.fun.id == fun
    )
