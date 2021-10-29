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

from functional.ffront.parsers import (
    FieldOperatorLowering,
    FieldOperatorParser,
    FieldOperatorSyntaxError,
)
from functional.iterator.ir import FunCall, FunctionDefinition, Sym, SymRef


COPY_FUN_DEF = FunctionDefinition(
    id="copy_field",
    params=[Sym(id="inp")],
    expr=FunCall(fun=SymRef(id="deref"), args=[SymRef(id="inp")]),
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
        _ = FieldOperatorParser.parse(wrong_syntax)


def test_invalid_syntax_no_return():
    """Field operators must end with a return statement."""

    def no_return(inp):
        tmp = inp  # noqa

    with pytest.raises(
        FieldOperatorSyntaxError,
        match=r"Field operator must return a field expression on the last line!",
    ):
        _ = FieldOperatorParser.parse(no_return)


def test_invalid_syntax_unpacking():
    """For now, only single target assigns are allowed."""

    def invalid_unpacking(inp1, inp2):
        tmp1, tmp2 = inp1, inp2  # noqa
        return tmp1

    with pytest.raises(FieldOperatorSyntaxError, match=r"Unpacking not allowed!"):
        _ = FieldOperatorParser.parse(invalid_unpacking)


def test_invalid_assign_to_expr():
    """Assigning to subscripts disallowed until a usecase can be found."""

    def invalid_assign_to_expr(inp1, inp2):
        tmp = inp1
        tmp[-1] = inp2
        return tmp

    with pytest.raises(FieldOperatorSyntaxError, match=r"Can only assign to names!"):
        _ = FieldOperatorParser.parse(invalid_assign_to_expr)


def test_copy_lower():
    def copy_field(inp):
        return inp

    # parsing
    parsed = FieldOperatorParser.parse(copy_field)
    lowered = FieldOperatorLowering.parse(parsed)
    assert lowered == COPY_FUN_DEF
    assert lowered.expr == COPY_FUN_DEF.expr


def test_temp_assignment():
    def copy_field(inp):
        tmp = inp
        inp = tmp
        tmp2 = inp
        return tmp2

    parsed = FieldOperatorParser.parse(copy_field)
    lowered = FieldOperatorLowering.parse(parsed)

    assert lowered == COPY_FUN_DEF
    assert lowered.expr == COPY_FUN_DEF.expr
