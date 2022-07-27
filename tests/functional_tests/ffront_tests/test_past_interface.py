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

from functional.common import Field, GridType
from functional.ffront import common_types
from functional.ffront.decorator import field_operator
from functional.ffront.fbuiltins import float32, float64, int32, int64, where
from functional.ffront.foast_passes.type_deduction import FieldOperatorTypeDeductionError
from functional.ffront.func_to_foast import FieldOperatorParser, FieldOperatorSyntaxError
from functional.ffront.func_to_past import ProgramParser
from functional.ffront.past_to_itir import ProgramLowering
from functional.ffront.symbol_makers import TypingError
from functional.iterator import ir as itir
from functional.iterator.builtins import (
    and_,
    deref,
    divides,
    eq,
    greater,
    if_,
    less,
    lift,
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
LIFT = itir.SymRef(id=lift.fun.__name__)


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


def test_tuple_constructed_in_out_invalid_slicing():
    def tuple_program(
        inp: Field[..., float64], out1: Field[..., float64], out2: Field[..., float64]
    ):
        make_tuple_op(inp, out=(out1[1:], out2))

    # with pytest.raises(
    #     FieldOperatorSyntaxError,
    #     match="Untyped parameters not allowed!",
    # ):
    _ = ProgramParser.apply_to_function(tuple_program)
    ProgramLowering.apply(_, function_definitions=[], grid_type=GridType.CARTESIAN)
