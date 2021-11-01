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

from typing import Any

import pytest

from functional.ffront import field_operator_ir as foir
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


def test_invalid_assign_to_expr():
    """Assigning to subscripts disallowed until a usecase can be found."""

    def invalid_assign_to_expr(inp1, inp2):
        tmp = inp1
        tmp[-1] = inp2
        return tmp

    with pytest.raises(FieldOperatorSyntaxError, match=r"Invalid assignment target!"):
        _ = FieldOperatorParser.parse(invalid_assign_to_expr)


def test_copy_lower():
    """A simple copy stencil should always be lowered to the same Iterator IR representation."""

    def copy_field(inp):
        return inp

    # parsing
    parsed = FieldOperatorParser.parse(copy_field)
    lowered = FieldOperatorLowering.parse(parsed)
    assert isinstance(lowered, FunctionDefinition)
    assert lowered.expr == COPY_FUN_DEF.expr
    assert lowered.params == COPY_FUN_DEF.params


def test_temp_assignment():
    """
    Assignments to temporaries should work like C preprocessor macros.

    Every subsequent occurence of the name assigned to should be expanded
    to the expression it was assigned. Circular assignment, overwriting
    temps and shadowing input parameters is allowed and taken into account.

    Finally everything is inlined into one expression. All names not used
    directly or indirectly in the return expression(s) are ignored and do not
    appear in the Iterator IR.
    """

    def copy_field_with_tmp(inp):
        tmp = inp
        inp = tmp
        tmp2 = inp
        return tmp2

    parsed = FieldOperatorParser.parse(copy_field_with_tmp)
    lowered = FieldOperatorLowering.parse(parsed)

    assert isinstance(lowered, FunctionDefinition)
    assert lowered.expr == COPY_FUN_DEF.expr
    assert lowered.params == COPY_FUN_DEF.params


def test_multi_target_assign():
    """
    Assigning to multiple targets should work as expected.

    Multiple targets means:

        target1 = target2 = source

    This should be equivalent to:

        target1 = source
        target2 = source

    """

    def multi_target_assign(inp):
        tmp1 = tmp2 = inp  # noqa
        return tmp2

    parsed = FieldOperatorParser.parse(multi_target_assign)
    lowered = FieldOperatorLowering.parse(parsed)

    assert parsed.body[0] == foir.SymExpr(id="tmp1", expr=foir.SymRef(id="inp"))
    assert parsed.body[1] == foir.SymExpr(id="tmp2", expr=foir.SymRef(id="inp"))

    assert isinstance(lowered, FunctionDefinition)
    assert lowered.expr == COPY_FUN_DEF.expr
    assert lowered.params == COPY_FUN_DEF.params


def test_tuple_lit_assign():
    """
    Unpacking assignments from a tuple literal are simply unrolled.

        a, b = c, d

    is turned into

        a = c
        b = d
    """

    def unpacking_assign_tuple(inp1, inp2):
        tmp1, [tmp2, [tmp3]] = inp1, [inp2, inp1]  # noqa
        return tmp3

    parsed = FieldOperatorParser.parse(unpacking_assign_tuple)
    lowered = FieldOperatorLowering.parse(parsed)

    assert parsed.body[0] == foir.SymExpr(id="tmp1", expr=foir.SymRef(id="inp1"))
    assert parsed.body[1] == foir.SymExpr(id="tmp2", expr=foir.SymRef(id="inp2"))
    assert parsed.body[2] == foir.SymExpr(id="tmp3", expr=foir.SymRef(id="inp1"))

    assert isinstance(lowered, FunctionDefinition)
    assert lowered.expr == FunCall(fun=SymRef(id="deref"), args=[SymRef(id="inp1")])
    assert lowered.params == [Sym(id="inp1"), Sym(id="inp2")]


def test_invalid_nonname_unpack():
    """Non-name assign targets buried in nested unpacks must be caught."""
    tmp2: list[Any]

    def nonname_unpack(inp1, inp2):
        tmp1, [tmp2[0]] = inp2, inp1  # noqa
        return tmp2[0]

    with pytest.raises(FieldOperatorSyntaxError, match=r"Invalid assignment target!"):
        _ = FieldOperatorParser.parse(nonname_unpack)
