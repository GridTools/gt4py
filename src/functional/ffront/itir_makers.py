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


def ensure_expr(literal_or_expr: Union[str, int, itir.Expr]) -> itir.Expr:
    """
    Convert literals into a SymRef or Literal and let expressions pass unchanged.

    Examples
    --------
    >>> ensure_expr("a")
    SymRef(id='a')

    >>> ensure_expr(3)
    Literal(value='3', type='int')

    >>> ensure_expr(itir.OffsetLiteral(value="i"))
    OffsetLiteral(value='i')
    """
    if isinstance(literal_or_expr, str):
        return ref(literal_or_expr)
    elif isinstance(literal_or_expr, int):
        return itir.Literal(value=str(literal_or_expr), type="int")
    return literal_or_expr


def ensure_offset(str_or_offset: Union[str, itir.OffsetLiteral]) -> itir.OffsetLiteral:
    """
    Convert Python literals into an OffsetLiteral and let OffsetLiterals pass unchanged.

    Examples
    --------
    >>> ensure_offset("V2E")
    OffsetLiteral(value='V2E')

    >>> ensure_offset(itir.OffsetLiteral(value="J"))
    OffsetLiteral(value='J')
    """
    if isinstance(str_or_offset, str):
        return itir.OffsetLiteral(value=str_or_offset)
    return str_or_offset


class lambda__:
    """
    Create a lambda from params and an expression.

    Examples
    --------
    >>> lambda__("a")(deref_("a"))  # doctest: +ELLIPSIS
    Lambda(params=[Sym(id='a')], expr=FunCall(fun=SymRef(id='deref'), args=[SymRef(id='a')]), ...)
    """

    def __init__(self, *args):
        self.args = args

    def __call__(self, expr):
        return itir.Lambda(params=[sym(arg) for arg in self.args], expr=ensure_expr(expr))


class call_:
    """
    Create a FunCall from an expression and arguments.

    Examples
    --------
    >>> call_("plus")(1, 1)
    FunCall(fun=SymRef(id='plus'), args=[Literal(value='1', type='int'), Literal(value='1', type='int')])
    """

    def __init__(self, expr):
        self.fun = ensure_expr(expr)

    def __call__(self, *exprs):
        return itir.FunCall(fun=self.fun, args=[ensure_expr(expr) for expr in exprs])


def deref_(expr):
    """Create a deref FunCall, shorthand for ``call("deref")(expr)``."""
    return call_("deref")(expr)


def plus_(left, right):
    """Create a plus FunCall, shorthand for ``call("plus")(left, right)``."""
    return call_("plus")(left, right)


def minus_(left, right):
    """Create a minus FunCall, shorthand for ``call("minus")(left, right)``."""
    return call_("minus")(left, right)


def multiplies_(left, right):
    """Create a multiplies FunCall, shorthand for ``call("multiplies")(left, right)``."""
    return call_("multiplies")(left, right)


def divides_(left, right):
    """Create a divides FunCall, shorthand for ``call("divides")(left, right)``."""
    return call_("divides")(left, right)


def and__(left, right):
    """Create an and_ FunCall, shorthand for ``call("and_")(left, right)``."""
    return call_("and_")(left, right)


def or__(left, right):
    """Create an or_ FunCall, shorthand for ``call("or_")(left, right)``."""
    return call_("or_")(left, right)


def greater_(left, right):
    """Create a greater FunCall, shorthand for ``call("greater")(left, right)``."""
    return call_("greater")(left, right)


def less_(left, right):
    """Create a less FunCall, shorthand for ``call("less")(left, right)``."""
    return call_("less")(left, right)


def eq_(left, right):
    """Create a eq FunCall, shorthand for ``call("eq")(left, right)``."""
    return call_("eq")(left, right)


def not__(expr):
    """Create a not_ FunCall, shorthand for ``call("not_")(expr)``."""
    return call_("not_")(expr)


def make_tuple_(*args):
    """Create a make_tuple FunCall, shorthand for ``call("make_tuple")(*args)``."""
    return call_("make_tuple")(*args)


def tuple_get_(tuple_expr, index):
    """Create a tuple_get FunCall, shorthand for ``call("tuple_get")(tuple_expr, index)``."""
    return call_("tuple_get")(tuple_expr, index)


def lift_(expr):
    """Create a lift FunCall, shorthand for ``call(call("lift")(expr))``."""
    return call_(call_("lift")(expr))


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
        return call_(lambda__(self.var)(form))(self.init_form)


def shift_(offset, value=None):
    """
    Create a shift call.

    Examples
    --------
    >>> shift_("i", 0)("a")
    FunCall(fun=FunCall(fun=SymRef(id='shift'), args=[OffsetLiteral(value='i'), Literal(value='0', type='int')]), args=[SymRef(id='a')])

    >>> shift_("V2E")("b")
    FunCall(fun=FunCall(fun=SymRef(id='shift'), args=[OffsetLiteral(value='V2E')]), args=[SymRef(id='b')])
    """
    offset = ensure_offset(offset)
    args = [offset]
    if value is not None:
        value = ensure_expr(value)
        args.append(value)
    return call_(call_("shift")(*args))


def literal_(value: str, typename: str):
    return itir.Literal(value=value, type=typename)
