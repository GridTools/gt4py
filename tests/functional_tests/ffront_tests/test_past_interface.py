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
import re
from typing import Tuple

import pytest

from functional.common import Field
from functional.ffront.decorator import field_operator
from functional.ffront.fbuiltins import float64
from functional.ffront.func_to_past import ProgramParser


@field_operator
def make_tuple_op(inp: Field[..., float64]) -> Tuple[Field[..., float64], Field[..., float64]]:
    return inp, inp


# --- Parsing ---
def test_tuple_constructed_in_out():
    def tuple_program(
        inp: Field[..., float64], out1: Field[..., float64], out2: Field[..., float64]
    ):
        make_tuple_op(inp, out=(out1, out2))

    _ = ProgramParser.apply_to_function(tuple_program)
