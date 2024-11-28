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
from gt4py.next.iterator.ir_utils import ir_makers as im


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


def is_ref_to(node, ref: str):
    return isinstance(node, itir.SymRef) and node.id == ref


def is_identity_as_fieldop(node: itir.Expr):
    """
    Match field operators implementing element-wise copy of a field argument,
    that is expressions of the form `as_fieldop(stencil)(*args)`

    >>> from gt4py.next.iterator.ir_utils import ir_makers as im
    >>> node = im.as_fieldop(im.lambda_("__arg0")(im.deref("__arg0")))("a")
    >>> is_identity_as_fieldop(node)
    True
    >>> node = im.as_fieldop("deref")("a")
    >>> is_identity_as_fieldop(node)
    False
    """
    if not is_applied_as_fieldop(node):
        return False
    stencil = node.fun.args[0]  # type: ignore[attr-defined]
    if (
        isinstance(stencil, itir.Lambda)
        and len(stencil.params) == 1
        and stencil == im.lambda_(stencil.params[0])(im.deref(stencil.params[0].id))
    ):
        return True
    return False
