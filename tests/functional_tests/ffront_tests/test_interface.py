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
        - math functions: abs(), max(), min, mod(), sin(), cos(), tan(), arcsin(), arccos(), arctan(),
            sqrt(), exp(), log(), isfinite(), isinf(), isnan(), floor(), ceil(), trunc()
    - evaluation test cases
"""
from __future__ import annotations

from functional.ffront.parsers import FieldOperatorParser
from functional.iterator.ir import FunCall, FunctionDefinition, Sym, SymRef


def test_copy_lower():
    def copy_field(inp):
        return inp

    # parsing
    parsed = FieldOperatorParser.parse(copy_field)
    assert isinstance(parsed, FunctionDefinition)
    assert parsed == FunctionDefinition(
        id="copy_field",
        params=[Sym(id="inp")],
        expr=FunCall(fun=SymRef(id="deref"), args=[SymRef(id="inp")]),
    )
