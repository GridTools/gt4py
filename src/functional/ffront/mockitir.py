# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

from typing import Union

from functional.iterator import ir as itir


def sym(sym_or_name: Union[str, itir.Sym]) -> itir.Sym:
    """
    Convert to Sym if necessary.

    Examples
    --------
    >>> sym("a")
    Sym(id='a')

    >>> sym(itir.Sym(id="b"))
    Sym(id='b')
    """
    if isinstance(sym_or_name, itir.Sym):
        return sym_or_name
    return itir.Sym(id=sym_or_name)


def ref(ref_or_name: Union[str, itir.SymRef]) -> itir.SymRef:
    """
    Convert to SymRef if necessary.

    Examples
    --------
    >>> ref("a")
    SymRef(id='a')

    >>> ref(itir.SymRef(id="b"))
    SymRef(id='b')
    """
    if isinstance(ref_or_name, itir.SymRef):
        return ref_or_name
    return itir.SymRef(id=ref_or_name)


def exprify(literal_or_expr: Union[str, int, itir.Expr]) -> itir.Expr:
    """
    Convert into a SymRef or IntLiteral if necessary.

    Examples
    --------
    >>> exprify("a")
    SymRef(id='a')

    >>> exprify(3)
    IntLiteral(value=3)

    >>> exprify(itir.OffsetLiteral(value="i"))
    OffsetLiteral(value='i')
    """
    if isinstance(literal_or_expr, str):
        return ref(literal_or_expr)
    elif isinstance(literal_or_expr, int):
        return itir.IntLiteral(value=literal_or_expr)
    return literal_or_expr


class lambda_:
    """
    Create a lambda from params and an expression.

    Examples
    --------
    >>> lambda_("a")(deref("a"))  # doctest: +ELLIPSIS
    Lambda(params=[Sym(id='a')], expr=FunCall(fun=SymRef(id='deref'), args=[SymRef(id='a')]), ...)
    """

    def __init__(self, *args):
        self.args = args

    def __call__(self, expr):
        return itir.Lambda(params=[sym(arg) for arg in self.args], expr=exprify(expr))


class call:
    """
    Create a FunCall from an expression and arguments.

    Examples
    --------
    >>> call("plus")(1, 1)
    FunCall(fun=SymRef(id='plus'), args=[IntLiteral(value=1), IntLiteral(value=1)])
    """

    def __init__(self, expr):
        self.fun = exprify(expr)

    def __call__(self, *exprs):
        return itir.FunCall(fun=self.fun, args=[exprify(expr) for expr in exprs])


def deref(expr):
    """Create a deref FunCall, shorthand for ``call("deref")(expr)``."""
    return call("deref")(expr)


def plus(left, right):
    """Create a plus FunCall, shorthand for ``call("plus")(left, right)``."""
    return call("plus")(left, right)


def minus(left, right):
    """Create a minus FunCall, shorthand for ``call("minus")(left, right)``."""
    return call("minus")(left, right)


def multiplies(left, right):
    """Create a multiplies FunCall, shorthand for ``call("multiplies")(left, right)``."""
    return call("multiplies")(left, right)


def divides(left, right):
    """Create a divides FunCall, shorthand for ``call("divides")(left, right)``."""
    return call("divides")(left, right)


def and_(left, right):
    """Create an and_ FunCall, shorthand for ``call("and_")(left, right)``."""
    return call("and_")(left, right)


def or_(left, right):
    """Create an or_ FunCall, shorthand for ``call("or_")(left, right)``."""
    return call("or_")(left, right)


def greater(left, right):
    """Create a greater FunCall, shorthand for ``call("greater")(left, right)``."""
    return call("greater")(left, right)


def less(left, right):
    """Create a less FunCall, shorthand for ``call("less")(left, right)``."""
    return call("less")(left, right)


def eq(left, right):
    """Create a eq FunCall, shorthand for ``call("eq")(left, right)``."""
    return call("eq")(left, right)


def not_(expr):
    """Create a not_ FunCall, shorthand for ``call("not_")(expr)``."""
    return call("not_")(expr)


def make_tuple(*args):
    """Create a make_tuple FunCall, shorthand for ``call("make_tuple")(*args)``."""
    return call("make_tuple")(*args)


def tuple_get(tuple_expr, index):
    """Create a tuple_get FunCall, shorthand for ``call("tuple_get")(tuple_expr, index)``."""
    return call("tuple_get")(tuple_expr, index)


def lift(expr):
    """Create a lift FunCall, shorthand for ``call("lift")(expr)``."""
    return call(call("lift")(expr))


class let:
    """
    Create a lambda expression that works as a let.

    Examples
    --------
    >>> let("a", "b")("a")  # doctest: +ELLIPSIS
    FunCall(fun=Lambda(params=[Sym(id='a')], expr=SymRef(id='a'), ...), args=[SymRef(id='b')])
    """

    def __init__(self, var, init_form):
        self.var = var
        self.init_form = init_form

    def __call__(self, form):
        return call(lambda_(self.var)(form))(self.init_form)


def shift(offset, value=None):
    """
    Create a shift call.

    Examples
    --------
    >>> shift("i", 0)("a")
    FunCall(fun=FunCall(fun=SymRef(id='shift'), args=[OffsetLiteral(value='i'), IntLiteral(value=0)]), args=[SymRef(id='a')])

    >>> shift("V2E")("b")
    FunCall(fun=FunCall(fun=SymRef(id='shift'), args=[OffsetLiteral(value='V2E')]), args=[SymRef(id='b')])
    """
    offset = itir.OffsetLiteral(value=offset)
    args = [offset]
    if value is not None:
        value = exprify(value)
        args.append(value)
    return call(call("shift")(*args))
