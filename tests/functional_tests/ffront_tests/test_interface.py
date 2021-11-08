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

import pytest

from functional.ffront.foir_to_itir import FieldOperatorLowering
from functional.ffront.func_to_foir import FieldOperatorParser, FieldOperatorSyntaxError
from functional.iterator import ir as itir


COPY_FUN_DEF = itir.FunctionDefinition(
    id="copy_field",
    params=[itir.Sym(id="inp")],
    expr=itir.FunCall(fun=itir.SymRef(id="deref"), args=[itir.SymRef(id="inp")]),
)


def test_invalid_syntax_error_emtpy_return():
    """Field operator syntax errors point to the file, line and column."""
    import inspect

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
                    itir.FunCall(fun=itir.SymRef(id="deref"), args=[itir.SymRef(id="inp1")]),
                    itir.FunCall(fun=itir.SymRef(id="deref"), args=[itir.SymRef(id="inp2")]),
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
