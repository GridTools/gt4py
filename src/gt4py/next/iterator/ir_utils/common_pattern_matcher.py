# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from typing import Any, Generic, List, TypeAlias, TypeGuard, TypeVar

from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im


_Fun = TypeVar("_Fun", bound=itir.Expr)


class _FunCallTo(itir.FunCall, Generic[_Fun]):
    fun: _Fun
    args: List[itir.Expr]


_FunCallToSymRef: TypeAlias = _FunCallTo[itir.SymRef]


def is_call_to(node: Any, fun: str | Iterable[str]) -> TypeGuard[_FunCallToSymRef]:
    """
    Match call expression to a given function.

    If the `node` argument is not an `itir.Node` the function does not error, but just returns
    `False`. This is useful in visitors, where sometimes we pass a list of nodes or a leaf
    attribute which can be anything.

    >>> from gt4py.next.iterator.ir_utils import ir_makers as im
    >>> node = im.plus(1, 2)
    >>> is_call_to(node, "plus")
    True
    >>> is_call_to(node, "minus")
    False
    >>> is_call_to(node, ("plus", "minus"))
    True
    """
    assert not isinstance(fun, itir.Node)  # to avoid accidentally passing the fun as first argument
    if isinstance(fun, str):
        return (
            isinstance(node, itir.FunCall)
            and isinstance(node.fun, itir.SymRef)
            and node.fun.id == fun
        )
    else:
        return any((is_call_to(node, f) for f in fun))


_FunCallToFunCallToRef: TypeAlias = _FunCallTo[_FunCallToSymRef]


def is_applied_lift(arg: itir.Node) -> TypeGuard[_FunCallToFunCallToRef]:
    """Match expressions of the form `lift(λ(...) → ...)(...)`."""
    return (
        isinstance(arg, itir.FunCall)
        and isinstance(arg.fun, itir.FunCall)
        and isinstance(arg.fun.fun, itir.SymRef)
        and arg.fun.fun.id == "lift"
    )


def is_applied_map(arg: itir.Node) -> TypeGuard[_FunCallToFunCallToRef]:
    """Match expressions of the form `map(λ(...) → ...)(...)`."""
    return (
        isinstance(arg, itir.FunCall)
        and isinstance(arg.fun, itir.FunCall)
        and isinstance(arg.fun.fun, itir.SymRef)
        and arg.fun.fun.id == "map_"
    )


def is_applied_reduce(arg: itir.Node) -> TypeGuard[_FunCallToFunCallToRef]:
    """Match expressions of the form `reduce(λ(...) → ...)(...)`."""
    return (
        isinstance(arg, itir.FunCall)
        and isinstance(arg.fun, itir.FunCall)
        and isinstance(arg.fun.fun, itir.SymRef)
        and arg.fun.fun.id == "reduce"
    )


def is_applied_shift(arg: itir.Node) -> TypeGuard[_FunCallToFunCallToRef]:
    """Match expressions of the form `shift(λ(...) → ...)(...)`."""
    return (
        isinstance(arg, itir.FunCall)
        and isinstance(arg.fun, itir.FunCall)
        and isinstance(arg.fun.fun, itir.SymRef)
        and arg.fun.fun.id == "shift"
    )


def is_applied_as_fieldop(arg: itir.Node) -> TypeGuard[_FunCallToFunCallToRef]:
    """Match expressions of the form `as_fieldop(stencil)(*args)`."""
    return isinstance(arg, itir.FunCall) and is_call_to(arg.fun, "as_fieldop")


_FunCallToLambda: TypeAlias = _FunCallTo[itir.Lambda]


def is_let(node: itir.Node) -> TypeGuard[_FunCallToLambda]:
    """Match expression of the form `(λ(...) → ...)(...)`."""
    return isinstance(node, itir.FunCall) and isinstance(node.fun, itir.Lambda)


def is_ref_to(node, ref: str) -> TypeGuard[itir.SymRef]:
    return isinstance(node, itir.SymRef) and node.id == ref


def is_identity_as_fieldop(node: itir.Expr) -> TypeGuard[_FunCallToFunCallToRef]:
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
    stencil = node.fun.args[0]
    if (
        isinstance(stencil, itir.Lambda)
        and len(stencil.params) == 1
        and stencil == im.lambda_(stencil.params[0])(im.deref(stencil.params[0].id))
    ):
        return True
    return False
