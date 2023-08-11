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

from typing import Callable, Union

from gt4py._core import definitions as core_defs
from gt4py.next.iterator import ir as itir
from gt4py.next.type_system import type_specifications as ts, type_translation


def sym(sym_or_name: Union[str, itir.Sym]) -> itir.Sym:
    """
    Convert to Sym if necessary.

    Examples
    --------
    >>> sym("a")
    Sym(id=SymbolName('a'), kind=None, dtype=None)

    >>> sym(itir.Sym(id="b"))
    Sym(id=SymbolName('b'), kind=None, dtype=None)
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
    SymRef(id=SymbolRef('a'))

    >>> ref(itir.SymRef(id="b"))
    SymRef(id=SymbolRef('b'))
    """
    if isinstance(ref_or_name, itir.SymRef):
        return ref_or_name
    return itir.SymRef(id=ref_or_name)


def ensure_expr(literal_or_expr: Union[str, core_defs.Scalar, itir.Expr]) -> itir.Expr:
    """
    Convert literals into a SymRef or Literal and let expressions pass unchanged.

    Examples
    --------
    >>> ensure_expr("a")
    SymRef(id=SymbolRef('a'))

    >>> ensure_expr(3)
    Literal(value='3', type='int32')

    >>> ensure_expr(itir.OffsetLiteral(value="i"))
    OffsetLiteral(value='i')
    """
    if isinstance(literal_or_expr, str):
        return ref(literal_or_expr)
    elif core_defs.is_scalar_type(literal_or_expr):
        return literal_from_value(literal_or_expr)
    assert isinstance(literal_or_expr, itir.Expr)
    return literal_or_expr


def ensure_offset(str_or_offset: Union[str, int, itir.OffsetLiteral]) -> itir.OffsetLiteral:
    """
    Convert Python literals into an OffsetLiteral and let OffsetLiterals pass unchanged.

    Examples
    --------
    >>> ensure_offset("V2E")
    OffsetLiteral(value='V2E')

    >>> ensure_offset(itir.OffsetLiteral(value="J"))
    OffsetLiteral(value='J')
    """
    if isinstance(str_or_offset, (str, int)):
        return itir.OffsetLiteral(value=str_or_offset)
    return str_or_offset


class lambda_:
    """
    Create a lambda from params and an expression.

    Examples
    --------
    >>> lambda_("a")(deref("a"))  # doctest: +ELLIPSIS
    Lambda(params=[Sym(id=SymbolName('a'), kind=None, dtype=None)], expr=FunCall(fun=SymRef(id=SymbolRef('deref')), args=[SymRef(id=SymbolRef('a'))]))
    """

    def __init__(self, *args):
        self.args = args

    def __call__(self, expr):
        return itir.Lambda(params=[sym(arg) for arg in self.args], expr=ensure_expr(expr))


class call:
    """
    Create a FunCall from an expression and arguments.

    Examples
    --------
    >>> call("plus")(1, 1)
    FunCall(fun=SymRef(id=SymbolRef('plus')), args=[Literal(value='1', type='int32'), Literal(value='1', type='int32')])
    """

    def __init__(self, expr):
        self.fun = ensure_expr(expr)

    def __call__(self, *exprs):
        return itir.FunCall(fun=self.fun, args=[ensure_expr(expr) for expr in exprs])


def deref(expr):
    """Create a deref FunCall, shorthand for ``call("deref")(expr)``."""
    return call("deref")(expr)


def plus(left, right):
    """Create a plus FunCall, shorthand for ``call("plus")(left, right)``."""
    return call("plus")(left, right)


def minus(left, right):
    """Create a minus FunCall, shorthand for ``call("minus")(left, right)``."""
    return call("minus")(left, right)


def multiplies_(left, right):
    """Create a multiplies FunCall, shorthand for ``call("multiplies")(left, right)``."""
    return call("multiplies")(left, right)


def divides_(left, right):
    """Create a divides FunCall, shorthand for ``call("divides")(left, right)``."""
    return call("divides")(left, right)


def floordiv_(left, right):
    """Create a floor division FunCall, shorthand for ``call("floordiv")(left, right)``."""
    # TODO(tehrengruber): Use int(floor(left/right)) as soon as we support integer casting
    #  and remove the `floordiv` builtin again.
    return call("floordiv")(left, right)


def mod(left, right):
    """Create a modulo FunCall, shorthand for ``call("mod")(left, right)``."""
    return call("mod")(left, right)


def and_(left, right):
    """Create an and_ FunCall, shorthand for ``call("and_")(left, right)``."""
    return call("and_")(left, right)


def or_(left, right):
    """Create an or_ FunCall, shorthand for ``call("or_")(left, right)``."""
    return call("or_")(left, right)


def xor_(left, right):
    """Create an xor_ FunCall, shorthand for ``call("xor_")(left, right)``."""
    return call("xor_")(left, right)


def greater(left, right):
    """Create a greater FunCall, shorthand for ``call("greater")(left, right)``."""
    return call("greater")(left, right)


def less(left, right):
    """Create a less FunCall, shorthand for ``call("less")(left, right)``."""
    return call("less")(left, right)


def less_equal(left, right):
    """Create a less_equal FunCall, shorthand for ``call("less_equal")(left, right)``."""
    return call("less_equal")(left, right)


def greater_equal(left, right):
    """Create a greater_equal FunCall, shorthand for ``call("greater_equal")(left, right)``."""
    return call("greater_equal")(left, right)


def not_eq(left, right):
    """Create a not_eq FunCall, shorthand for ``call("not_eq")(left, right)``."""
    return call("not_eq")(left, right)


def eq(left, right):
    """Create a eq FunCall, shorthand for ``call("eq")(left, right)``."""
    return call("eq")(left, right)


def not_(expr):
    """Create a not_ FunCall, shorthand for ``call("not_")(expr)``."""
    return call("not_")(expr)


def make_tuple(*args):
    """Create a make_tuple FunCall, shorthand for ``call("make_tuple")(*args)``."""
    return call("make_tuple")(*args)


def tuple_get(index: str | int, tuple_expr):
    """Create a tuple_get FunCall, shorthand for ``call("tuple_get")(index, tuple_expr)``."""
    return call("tuple_get")(literal(str(index), itir.INTEGER_INDEX_BUILTIN), tuple_expr)


def if_(cond, true_val, false_val):
    """Create a not_ FunCall, shorthand for ``call("if_")(expr)``."""
    return call("if_")(cond, true_val, false_val)


def lift(expr):
    """Create a lift FunCall, shorthand for ``call(call("lift")(expr))``."""
    return call(call("lift")(expr))


class let:
    """
    Create a lambda expression that works as a let.

    Examples
    --------
    >>> str(let("a", "b")("a"))  # doctest: +ELLIPSIS
    '(λ(a) → a)(b)'
    >>> str(let("a", 1,
    ...         "b", 2
    ... )(plus("a", "b")))
    '(λ(a, b) → a + b)(1, 2)'
    """

    def __init__(self, *vars_and_values):
        assert len(vars_and_values) % 2 == 0
        self.vars = vars_and_values[0::2]
        self.init_forms = vars_and_values[1::2]

    def __call__(self, form):
        return call(lambda_(*self.vars)(form))(*self.init_forms)


def shift(offset, value=None):
    """
    Create a shift call.

    Examples
    --------
    >>> shift("i", 0)("a")
    FunCall(fun=FunCall(fun=SymRef(id=SymbolRef('shift')), args=[OffsetLiteral(value='i'), OffsetLiteral(value=0)]), args=[SymRef(id=SymbolRef('a'))])

    >>> shift("V2E")("b")
    FunCall(fun=FunCall(fun=SymRef(id=SymbolRef('shift')), args=[OffsetLiteral(value='V2E')]), args=[SymRef(id=SymbolRef('b'))])
    """
    offset = ensure_offset(offset)
    args = [offset]
    if value is not None:
        value = ensure_offset(value)
        args.append(value)
    return call(call("shift")(*args))


def literal(value: str, typename: str):
    return itir.Literal(value=value, type=typename)


def literal_from_value(val: core_defs.Scalar) -> itir.Literal:
    """
    Make a literal node from a value.

    >>> literal_from_value(1.)
    Literal(value='1.0', type='float64')
    >>> literal_from_value(1)
    Literal(value='1', type='int32')
    >>> literal_from_value(2147483648)
    Literal(value='2147483648', type='int64')
    >>> literal_from_value(True)
    Literal(value='True', type='bool')
    """
    if not isinstance(val, core_defs.Scalar):  # type: ignore[arg-type] # mypy bug #11673
        raise ValueError(f"Value must be a scalar, but got {type(val).__name__}")

    # At the time this has been written the iterator module has its own type system that is
    # uncoupled from the one used in the frontend. However since we decided to eventually replace
    # it with the frontend type system we already use it here (avoiding unnecessary code
    # duplication).
    type_spec = type_translation.from_value(val)
    assert isinstance(type_spec, ts.ScalarType)

    typename = type_spec.kind.name.lower()
    assert typename in itir.TYPEBUILTINS

    return itir.Literal(value=str(val), type=typename)


def neighbors(offset, it):
    offset = ensure_offset(offset)
    return call("neighbors")(offset, it)


def lifted_neighbors(offset, it) -> itir.Expr:
    """
    Create a lifted neighbors call.

    Examples
    --------
    >>> str(lifted_neighbors("off", "a"))
    '(↑(λ(it) → neighbors(offₒ, it)))(a)'
    """
    return lift(lambda_("it")(neighbors(offset, "it")))(it)


def promote_to_const_iterator(expr: str | itir.Expr) -> itir.Expr:
    """
    Create a lifted nullary lambda that captures `expr`.

    Examples
    --------
    >>> str(promote_to_const_iterator("foo"))
    '(↑(λ() → foo))()'
    """
    return lift(lambda_()(expr))()


def promote_to_lifted_stencil(op: str | itir.SymRef | Callable) -> Callable[..., itir.Expr]:
    """
    Promotes a function `op` from values to iterators.

    `op` is a function from values to value.

    Returns:
        A lifted stencil, a function from iterators to iterator.

    Examples
    --------
    >>> str(promote_to_lifted_stencil("op")("a", "b"))
    '(↑(λ(__arg0, __arg1) → op(·__arg0, ·__arg1)))(a, b)'
    """
    if isinstance(op, (str, itir.SymRef)):
        op = call(op)

    def _impl(*its: itir.Expr) -> itir.Expr:
        args = [
            f"__arg{i}" for i in range(len(its))
        ]  # TODO: `op` must not contain `SymRef(id="__argX")`
        return lift(lambda_(*args)(op(*[deref(arg) for arg in args])))(*its)

    return _impl


def map_(op):
    """Create a `map_` call."""
    return call(call("map_")(op))
