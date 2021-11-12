"""Test field view interface.

Basic Interface Tests
=====================

    - declare a connectivity
    - create and run a stencil
        - field args declaration
        - scalar args
    - create and run a fencil
        - pass fields
        - pass connectivities (at run time, later at compile time too)
        - out field
        - think about ways to pass backend/connectivities etc
            (in function signature / in functor config method / with block)
    - built-in field operators
        - arithmetics
        - shift
        - neighbor reductions
        - math functions: abs(), max(), min, mod(), sin(), cos(), tan(), arcsin(), arccos(),
            arctan(), sqrt(), exp(), log(), isfinite(), isinf(), isnan(), floor(), ceil(), trunc()
    - evaluation test cases
"""
from __future__ import annotations

import inspect
import typing

import pytest

from functional.ffront.foast_to_itir import FieldOperatorLowering
from functional.ffront.func_to_foast import FieldOperatorParser, FieldOperatorSyntaxError
from functional.iterator import ir as itir
from functional.iterator.builtins import (
    and_,
    deref,
    divides,
    eq,
    greater,
    if_,
    less,
    make_tuple,
    minus,
    multiplies,
    not_,
    or_,
    plus,
    tuple_get,
)


DEREF = itir.SymRef(id=deref.fun.__name__)
PLUS = itir.SymRef(id=plus.fun.__name__)
MINUS = itir.SymRef(id=minus.fun.__name__)
MULTIPLIES = itir.SymRef(id=multiplies.fun.__name__)
DIVIDES = itir.SymRef(id=divides.fun.__name__)
MAKE_TUPLE = itir.SymRef(id=make_tuple.fun.__name__)
TUPLE_GET = itir.SymRef(id=tuple_get.fun.__name__)
IF = itir.SymRef(id=if_.fun.__name__)
NOT = itir.SymRef(id=not_.fun.__name__)
GREATER = itir.SymRef(id=greater.fun.__name__)
LESS = itir.SymRef(id=less.fun.__name__)
EQ = itir.SymRef(id=eq.fun.__name__)
AND = itir.SymRef(id=and_.fun.__name__)
OR = itir.SymRef(id=or_.fun.__name__)


COPY_FUN_DEF = itir.FunctionDefinition(
    id="copy_field",
    params=[itir.Sym(id="inp")],
    expr=itir.FunCall(fun=DEREF, args=[itir.SymRef(id="inp")]),
)


def test_invalid_syntax_error_emtpy_return():
    """Field operator syntax errors point to the file, line and column."""

    def wrong_syntax(inp):
        return

    lineno = inspect.getsourcelines(wrong_syntax)[1] + 1

    with pytest.raises(
        FieldOperatorSyntaxError,
        match=(
            r"Invalid Field Operator Syntax: "
            rf"Empty return not allowed \(test_interface.py, line {lineno}\)"
        ),
    ):
        _ = FieldOperatorParser.apply(wrong_syntax)


def test_invalid_syntax_no_return():
    """Field operators must end with a return statement."""

    def no_return(inp):
        tmp = inp  # noqa

    with pytest.raises(
        FieldOperatorSyntaxError,
        match=r"Field operator must return a field expression on the last line!",
    ):
        _ = FieldOperatorParser.apply(no_return)


def test_invalid_assign_to_expr():
    """Assigning to subscripts disallowed until a usecase can be found."""

    def invalid_assign_to_expr(inp1, inp2):
        tmp = inp1
        tmp[-1] = inp2
        return tmp

    with pytest.raises(FieldOperatorSyntaxError, match=r"Can only assign to names!"):
        _ = FieldOperatorParser.apply(invalid_assign_to_expr)


def test_copy_lower():
    def copy_field(inp):
        return inp

    # ast_passes
    parsed = FieldOperatorParser.apply(copy_field)
    lowered = FieldOperatorLowering.apply(parsed)
    assert lowered == COPY_FUN_DEF
    assert lowered.expr == COPY_FUN_DEF.expr


def test_syntax_unpacking():
    """For now, only single target assigns are allowed."""

    def unpacking(inp1, inp2):
        tmp1, tmp2 = inp1, inp2  # noqa
        return tmp1

    parsed = FieldOperatorParser.apply(unpacking)
    lowered = FieldOperatorLowering.apply(parsed)
    assert lowered.expr == itir.FunCall(
        fun=itir.SymRef(id="tuple_get"),
        args=[
            itir.FunCall(
                fun=itir.SymRef(id="make_tuple"),
                args=[
                    itir.FunCall(fun=DEREF, args=[itir.SymRef(id="inp1")]),
                    itir.FunCall(fun=DEREF, args=[itir.SymRef(id="inp2")]),
                ],
            ),
            itir.IntLiteral(value=0),
        ],
    )


def test_temp_assignment():
    def copy_field(inp):
        tmp = inp
        inp = tmp
        tmp2 = inp
        return tmp2

    parsed = FieldOperatorParser.apply(copy_field)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered == COPY_FUN_DEF
    assert lowered.expr == COPY_FUN_DEF.expr


def test_annotated_assignment():
    Field = typing.TypeVar("Field")

    def copy_field(inp: Field):
        tmp: Field = inp
        return tmp

    parsed = FieldOperatorParser.apply(copy_field)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered == COPY_FUN_DEF
    assert lowered.expr == COPY_FUN_DEF.expr


def test_call():
    def identity(x):
        return x

    def call(inp):
        return identity(inp)

    parsed = FieldOperatorParser.apply(call)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=itir.SymRef(id="identity"), args=[itir.FunCall(fun=DEREF, args=[itir.SymRef(id="inp")])]
    )


def test_call_expression():
    def get_identity():
        return lambda x: x

    def call_expr(inp):
        return get_identity()(inp)

    with pytest.raises(
        FieldOperatorSyntaxError,
        match=r"functions can only be called directly!",
    ):
        _ = FieldOperatorParser.apply(call_expr)


def test_unary_ops():
    def unary(inp):
        tmp = +inp
        tmp = -tmp
        return tmp

    parsed = FieldOperatorParser.apply(unary)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=MINUS,
        args=[
            itir.IntLiteral(value=0),
            itir.FunCall(
                fun=PLUS,
                args=[
                    itir.IntLiteral(value=0),
                    itir.FunCall(fun=DEREF, args=[itir.SymRef(id="inp")]),
                ],
            ),
        ],
    )


def test_unary_not():
    def unary_not(cond):
        return not cond

    parsed = FieldOperatorParser.apply(unary_not)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=NOT, args=[itir.FunCall(fun=DEREF, args=[itir.SymRef(id="cond")])]
    )


def test_binary_plus():
    def plus(a, b):
        return a + b

    parsed = FieldOperatorParser.apply(plus)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=PLUS,
        args=[
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="a")]),
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="b")]),
        ],
    )


def test_binary_mult():
    def mult(a, b):
        return a * b

    parsed = FieldOperatorParser.apply(mult)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=MULTIPLIES,
        args=[
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="a")]),
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="b")]),
        ],
    )


def test_binary_minus():
    def minus(a, b):
        return a - b

    parsed = FieldOperatorParser.apply(minus)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=MINUS,
        args=[
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="a")]),
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="b")]),
        ],
    )


def test_binary_div():
    def division(a, b):
        return a / b

    parsed = FieldOperatorParser.apply(division)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=DIVIDES,
        args=[
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="a")]),
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="b")]),
        ],
    )


def test_binary_and():
    def bit_and(a, b):
        return a & b

    parsed = FieldOperatorParser.apply(bit_and)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=AND,
        args=[
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="a")]),
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="b")]),
        ],
    )


def test_binary_or():
    def bit_or(a, b):
        return a | b

    parsed = FieldOperatorParser.apply(bit_or)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=OR,
        args=[
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="a")]),
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="b")]),
        ],
    )


def test_binary_pow():
    def power(inp):
        return inp ** 3

    with pytest.raises(
        FieldOperatorSyntaxError,
        match=(r"`\*\*` operator not supported!"),
    ):
        _ = FieldOperatorParser.apply(power)


def test_binary_mod():
    def power(inp):
        return inp % 3

    with pytest.raises(
        FieldOperatorSyntaxError,
        match=(r"`%` operator not supported!"),
    ):
        _ = FieldOperatorParser.apply(power)


def test_compare_gt():
    def comp_gt(a, b):
        return a > b

    parsed = FieldOperatorParser.apply(comp_gt)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=GREATER,
        args=[
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="a")]),
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="b")]),
        ],
    )


def test_compare_lt():
    def comp_lt(a, b):
        return a < b

    parsed = FieldOperatorParser.apply(comp_lt)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=LESS,
        args=[
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="a")]),
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="b")]),
        ],
    )


def test_compare_eq():
    def comp_eq(a, b):
        return a == b

    parsed = FieldOperatorParser.apply(comp_eq)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=EQ,
        args=[
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="a")]),
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="b")]),
        ],
    )


def test_compare_chain():
    def compare_chain(a, b, c):
        return a > b > c

    parsed = FieldOperatorParser.apply(compare_chain)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=GREATER,
        args=[
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="a")]),
            itir.FunCall(
                fun=GREATER,
                args=[
                    itir.FunCall(fun=DEREF, args=[itir.SymRef(id="b")]),
                    itir.FunCall(fun=DEREF, args=[itir.SymRef(id="c")]),
                ],
            ),
        ],
    )


def test_bool_and():
    def bool_and(a, b):
        return a and b

    with pytest.raises(
        FieldOperatorSyntaxError,
        match=(r"`and` operator not allowed!"),
    ):
        _ = FieldOperatorParser.apply(bool_and)


def test_bool_or():
    def bool_or(a, b):
        return a or b

    with pytest.raises(
        FieldOperatorSyntaxError,
        match=(r"`or` operator not allowed!"),
    ):
        _ = FieldOperatorParser.apply(bool_or)
