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


def test_syntax_error():
    def wrong_syntax(inp):
        return

    with pytest.raises(
        FieldOperatorSyntaxError,
        match=r"Invalid Field Operator Syntax: Empty return not allowed \(test_interface.py, line \d+\)",
    ):
        _ = FieldOperatorParser.parse(wrong_syntax)


def test_copy_lower():
    def copy_field(inp):
        return inp

    # parsing
    parsed = FieldOperatorParser.parse(copy_field)
    lowered = FieldOperatorLowering.parse(parsed)
    assert isinstance(lowered, FunctionDefinition)
    assert lowered == FunctionDefinition(
        id="copy_field",
        params=[Sym(id="inp")],
        expr=FunCall(fun=SymRef(id="deref"), args=[SymRef(id="inp")]),
    )
