# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

import pytest

from gt4py.eve.pattern_matching import ObjectPattern as P
from gt4py.next.common import Field, GTTypeError
from gt4py.next.ffront import field_operator_ast as foast
from gt4py.next.ffront.fbuiltins import (
    Dimension,
    astype,
    broadcast,
    float32,
    float64,
    int32,
    int64,
    where,
)
from gt4py.next.ffront.foast_passes.type_deduction import FieldOperatorTypeDeductionError
from gt4py.next.ffront.func_to_foast import FieldOperatorParser, FieldOperatorSyntaxError
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.builtins import (
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
    xor_,
)
from gt4py.next.type_system import type_specifications as ts
from gt4py.next.type_system.type_translation import TypingError


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
XOR = itir.SymRef(id=xor_.fun.__name__)
LIFT = itir.SymRef(id=lift.fun.__name__)

TDim = Dimension("TDim")  # Meaningless dimension, used for tests.


# --- Parsing ---
def test_untyped_arg():
    """Field operator parameters must be type annotated."""

    def untyped(inp):
        return inp

    with pytest.raises(
        FieldOperatorSyntaxError,
        match="Untyped parameters not allowed!",
    ):
        _ = FieldOperatorParser.apply_to_function(untyped)


def test_mistyped_arg():
    """Field operator parameters must be type annotated."""

    def mistyped(inp: Field):
        return inp

    with pytest.raises(
        TypingError,
        match="Field type requires two arguments, got 0!",
    ):
        _ = FieldOperatorParser.apply_to_function(mistyped)


def test_return_type():
    """Return type annotation should be stored on the FieldOperator."""

    def rettype(inp: Field[[TDim], float64]) -> Field[[TDim], float64]:
        return inp

    parsed = FieldOperatorParser.apply_to_function(rettype)

    assert parsed.body.stmts[-1].value.type == ts.FieldType(
        dims=[TDim],
        dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64, shape=None),
    )


def test_invalid_syntax_no_return():
    """Field operators must end with a return statement."""

    def no_return(inp: Field[[TDim], "float64"]):
        tmp = inp  # noqa

    with pytest.raises(
        FieldOperatorSyntaxError,
        match="Function must return a value, but no return statement was found\.",
    ):
        _ = FieldOperatorParser.apply_to_function(no_return)


def test_invalid_assign_to_expr():
    """Assigning to subscripts disallowed until a usecase can be found."""

    def invalid_assign_to_expr(inp1: Field[[TDim], "float64"], inp2: Field[[TDim], "float64"]):
        tmp = inp1
        tmp[-1] = inp2
        return tmp

    with pytest.raises(FieldOperatorSyntaxError, match=r"Can only assign to names! \(.*\)"):
        _ = FieldOperatorParser.apply_to_function(invalid_assign_to_expr)


def test_temp_assignment():
    def copy_field(inp: Field[[TDim], "float64"]):
        tmp = inp
        inp = tmp
        tmp2 = inp
        return tmp2

    parsed = FieldOperatorParser.apply_to_function(copy_field)

    assert parsed.body.annex.symtable["tmp__0"].type == ts.FieldType(
        dims=[TDim],
        dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64, shape=None),
    )


def test_clashing_annotated_assignment():
    pytest.skip("Annotated assignments are not properly supported at the moment.")

    def clashing(inp: Field[[TDim], "float64"]):
        tmp: Field[[TDim], "int64"] = inp
        return tmp

    with pytest.raises(FieldOperatorTypeDeductionError, match="type inconsistency"):
        _ = FieldOperatorParser.apply_to_function(clashing)


def test_binary_pow():
    def power(inp: Field[[TDim], "float64"]):
        return inp**3

    parsed = FieldOperatorParser.apply_to_function(power)

    assert parsed.body.stmts[-1].value.type == ts.FieldType(
        dims=[TDim],
        dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64, shape=None),
    )


def test_binary_mod():
    def modulo(inp: Field[[TDim], "int64"]):
        return inp % 3

    parsed = FieldOperatorParser.apply_to_function(modulo)

    assert parsed.body.stmts[-1].value.type == ts.FieldType(
        dims=[TDim],
        dtype=ts.ScalarType(kind=ts.ScalarKind.INT64, shape=None),
    )


def test_bool_and():
    def bool_and(a: Field[[TDim], "bool"], b: Field[[TDim], "bool"]):
        return a and b

    with pytest.raises(
        FieldOperatorSyntaxError,
        match=(r"`and`/`or` operator not allowed!"),
    ):
        _ = FieldOperatorParser.apply_to_function(bool_and)


def test_bool_or():
    def bool_or(a: Field[[TDim], "bool"], b: Field[[TDim], "bool"]):
        return a or b

    with pytest.raises(
        FieldOperatorSyntaxError,
        match=(r"`and`/`or` operator not allowed!"),
    ):
        _ = FieldOperatorParser.apply_to_function(bool_or)


def test_bool_xor():
    def bool_xor(a: Field[[TDim], "bool"], b: Field[[TDim], "bool"]):
        return a ^ b

    parsed = FieldOperatorParser.apply_to_function(bool_xor)

    assert parsed.body.stmts[-1].value.type == ts.FieldType(
        dims=[TDim],
        dtype=ts.ScalarType(kind=ts.ScalarKind.BOOL, shape=None),
    )


def test_unary_tilde():
    def unary_tilde(a: Field[[TDim], "bool"]):
        return ~a

    parsed = FieldOperatorParser.apply_to_function(unary_tilde)

    assert parsed.body.stmts[-1].value.type == ts.FieldType(
        dims=[TDim],
        dtype=ts.ScalarType(kind=ts.ScalarKind.BOOL, shape=None),
    )


def test_scalar_cast():
    def cast_scalar_temp():
        tmp = int64(1)
        return int32(tmp)

    with pytest.raises(FieldOperatorSyntaxError, match=(r"only takes literal arguments!")):
        _ = FieldOperatorParser.apply_to_function(cast_scalar_temp)


def test_conditional_wrong_mask_type():
    def conditional_wrong_mask_type(
        a: Field[[TDim], float64],
    ) -> Field[[TDim], float64]:
        return where(a, a, a)

    msg = r"Expected a field with dtype bool."
    with pytest.raises(FieldOperatorTypeDeductionError, match=msg):
        _ = FieldOperatorParser.apply_to_function(conditional_wrong_mask_type)


def test_conditional_wrong_arg_type():
    def conditional_wrong_arg_type(
        mask: Field[[TDim], bool],
        a: Field[[TDim], float32],
        b: Field[[TDim], float64],
    ) -> Field[[TDim], float64]:
        return where(mask, a, b)

    msg = r"Could not promote scalars of different dtype \(not implemented\)."
    with pytest.raises(FieldOperatorTypeDeductionError) as exc_info:
        _ = FieldOperatorParser.apply_to_function(conditional_wrong_arg_type)

    assert re.search(msg, exc_info.value.__cause__.args[0]) is not None


def test_ternary_with_field_condition():
    def ternary_with_field_condition(cond: Field[[], bool]):
        return 1 if cond else 2

    with pytest.raises(FieldOperatorTypeDeductionError, match=r"should be .* `bool`"):
        _ = FieldOperatorParser.apply_to_function(ternary_with_field_condition)


def test_correct_return_type_annotation():
    """See ADR 13."""

    def correct_return_type_annotation() -> float:
        return 1.0

    FieldOperatorParser.apply_to_function(correct_return_type_annotation)


def test_adr13_wrong_return_type_annotation():
    """See ADR 13."""

    def wrong_return_type_annotation() -> Field[[], float]:
        return 1.0

    with pytest.raises(GTTypeError, match=r"Expected `float.*`"):
        _ = FieldOperatorParser.apply_to_function(wrong_return_type_annotation)


def test_adr13_fixed_return_type_annotation():
    """See ADR 13."""

    def fixed_return_type_annotation() -> Field[[], float]:
        return broadcast(1.0, ())

    FieldOperatorParser.apply_to_function(fixed_return_type_annotation)


def test_no_implicit_broadcast_in_field_op_call():
    """See ADR 13."""

    def no_implicit_broadcast_in_field_op_call(scalar: float) -> float:
        return scalar

    def no_implicit_broadcast_in_field_op_call_caller() -> float:
        return no_implicit_broadcast_in_field_op_call(1.0)

    FieldOperatorParser.apply_to_function(no_implicit_broadcast_in_field_op_call_caller)


def test_astype():
    def astype_fieldop(a: Field[[TDim], "int64"]) -> Field[[TDim], float64]:
        return astype(a, float64)

    parsed = FieldOperatorParser.apply_to_function(astype_fieldop)

    assert parsed.body.stmts[-1].value.type == ts.FieldType(
        dims=[TDim],
        dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64, shape=None),
    )


# --- External symbols ---
def test_closure_symbols():
    import numpy as np

    from gt4py.eve.utils import FrozenNamespace

    nonlocals_unreferenced = FrozenNamespace()
    nonlocals = FrozenNamespace(float_value=2.3, np_value=np.float32(3.4))

    def operator_with_refs(inp: Field[[TDim], "float64"], inp2: Field[[TDim], "float32"]):
        a = inp + nonlocals.float_value
        b = inp2 + nonlocals.np_value
        return a, b

    parsed = FieldOperatorParser.apply_to_function(operator_with_refs)
    assert "nonlocals_unreferenced" not in {**parsed.annex.symtable, **parsed.body.annex.symtable}
    assert "nonlocals" not in {**parsed.annex.symtable, **parsed.body.annex.symtable}

    pattern_node = P(
        foast.FunctionDefinition,
        body=P(
            foast.BlockStmt,
            stmts=[
                P(
                    foast.Assign,
                    value=P(foast.BinOp, right=P(foast.Constant, value=nonlocals.float_value)),
                ),
                P(
                    foast.Assign,
                    value=P(foast.BinOp, right=P(foast.Constant, value=nonlocals.np_value)),
                ),
                P(foast.Return),
            ],
        ),
    )
    assert pattern_node.match(parsed, raise_exception=True)


def test_wrong_return_type_annotation():
    ADim = Dimension("ADim")
    BDim = Dimension("BDim")

    def wrong_return_type_annotation(a: Field[[ADim], float64]) -> Field[[BDim], float64]:
        return a

    with pytest.raises(
        GTTypeError,
        match=r"Annotated return type does not match deduced return type",
    ):
        _ = FieldOperatorParser.apply_to_function(wrong_return_type_annotation)


def test_empty_dims_type():
    def empty_dims() -> Field[[], float]:
        return 1.0

    with pytest.raises(
        GTTypeError,
        match=r"Annotated return type does not match deduced return type",
    ):
        _ = FieldOperatorParser.apply_to_function(empty_dims)


def test_zero_dims_ternary():
    ADim = Dimension("ADim")

    def zero_dims_ternary(
        cond: Field[[], float64], a: Field[[ADim], float64], b: Field[[ADim], float64]
    ):
        return a if cond == 1 else b

    msg = r"Could not deduce type"
    with pytest.raises(FieldOperatorTypeDeductionError) as exc_info:
        _ = FieldOperatorParser.apply_to_function(zero_dims_ternary)

    assert re.search(msg, exc_info.value.args[0]) is not None
