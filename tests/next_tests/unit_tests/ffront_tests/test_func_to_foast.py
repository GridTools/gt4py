# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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

import gt4py.next as gtx
from gt4py.eve.pattern_matching import ObjectPattern as P
from gt4py.next import (
    astype,
    broadcast,
    errors,
    float32,
    float64,
    int32,
    int64,
    where,
)
from gt4py.next.ffront.experimental import concat_where
from gt4py.next.ffront import field_operator_ast as foast
from gt4py.next.ffront.ast_passes import single_static_assign as ssa
from gt4py.next.ffront.func_to_foast import FieldOperatorParser
from gt4py.next.iterator import builtins as itb, ir as itir
from gt4py.next.type_system import type_specifications as ts


DEREF = itir.SymRef(id=itb.deref.fun.__name__)
PLUS = itir.SymRef(id=itb.plus.fun.__name__)
MINUS = itir.SymRef(id=itb.minus.fun.__name__)
MULTIPLIES = itir.SymRef(id=itb.multiplies.fun.__name__)
DIVIDES = itir.SymRef(id=itb.divides.fun.__name__)
MAKE_TUPLE = itir.SymRef(id=itb.make_tuple.fun.__name__)
TUPLE_GET = itir.SymRef(id=itb.tuple_get.fun.__name__)
IF = itir.SymRef(id=itb.if_.fun.__name__)
NOT = itir.SymRef(id=itb.not_.fun.__name__)
GREATER = itir.SymRef(id=itb.greater.fun.__name__)
LESS = itir.SymRef(id=itb.less.fun.__name__)
EQ = itir.SymRef(id=itb.eq.fun.__name__)
AND = itir.SymRef(id=itb.and_.fun.__name__)
OR = itir.SymRef(id=itb.or_.fun.__name__)
XOR = itir.SymRef(id=itb.xor_.fun.__name__)
LIFT = itir.SymRef(id=itb.lift.fun.__name__)

TDim = gtx.Dimension("TDim")  # Meaningless dimension, used for tests.


# --- Parsing ---
def test_untyped_arg():
    """Field operator parameters must be type annotated."""

    def untyped(inp):
        return inp

    with pytest.raises(errors.MissingParameterAnnotationError):
        _ = FieldOperatorParser.apply_to_function(untyped)


def test_mistyped_arg():
    """Field operator parameters must be type annotated."""

    def mistyped(inp: gtx.Field):
        return inp

    with pytest.raises(ValueError, match="Field type requires two arguments, got 0."):
        _ = FieldOperatorParser.apply_to_function(mistyped)


def test_return_type():
    """Return type annotation should be stored on the FieldOperator."""

    def rettype(inp: gtx.Field[[TDim], float64]) -> gtx.Field[[TDim], float64]:
        return inp

    parsed = FieldOperatorParser.apply_to_function(rettype)

    assert parsed.body.stmts[-1].value.type == ts.FieldType(
        dims=[TDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64, shape=None)
    )


def test_invalid_syntax_no_return():
    """Field operators must end with a return statement."""

    def no_return(inp: gtx.Field[[TDim], "float64"]):
        tmp = inp  # noqa

    with pytest.raises(errors.DSLError, match=".*return.*"):
        _ = FieldOperatorParser.apply_to_function(no_return)


def test_invalid_assign_to_expr():
    """Assigning to subscripts disallowed until a usecase can be found."""

    def invalid_assign_to_expr(
        inp1: gtx.Field[[TDim], "float64"], inp2: gtx.Field[[TDim], "float64"]
    ):
        tmp = inp1
        tmp[-1] = inp2
        return tmp

    with pytest.raises(errors.DSLError, match=r".*assign.*"):
        _ = FieldOperatorParser.apply_to_function(invalid_assign_to_expr)


def test_temp_assignment():
    def copy_field(inp: gtx.Field[[TDim], "float64"]):
        tmp = inp
        inp = tmp
        tmp2 = inp
        return tmp2

    parsed = FieldOperatorParser.apply_to_function(copy_field)

    assert parsed.body.annex.symtable[ssa.unique_name("tmp", 0)].type == ts.FieldType(
        dims=[TDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64, shape=None)
    )


def test_clashing_annotated_assignment():
    pytest.skip("Annotated assignments are not properly supported at the moment.")

    def clashing(inp: gtx.Field[[TDim], "float64"]):
        tmp: gtx.Field[[TDim], "int64"] = inp
        return tmp

    with pytest.raises(errors.DSLError, match="type inconsistency"):
        _ = FieldOperatorParser.apply_to_function(clashing)


def test_binary_pow():
    def power(inp: gtx.Field[[TDim], "float64"]):
        return inp**3

    parsed = FieldOperatorParser.apply_to_function(power)

    assert parsed.body.stmts[-1].value.type == ts.FieldType(
        dims=[TDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64, shape=None)
    )


def test_binary_mod():
    def modulo(inp: gtx.Field[[TDim], "int32"]):
        return inp % 3

    parsed = FieldOperatorParser.apply_to_function(modulo)

    assert parsed.body.stmts[-1].value.type == ts.FieldType(
        dims=[TDim], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32, shape=None)
    )


def test_boolean_and_op_unsupported():
    def bool_and(a: gtx.Field[[TDim], "bool"], b: gtx.Field[[TDim], "bool"]):
        return a and b

    with pytest.raises(errors.UnsupportedPythonFeatureError, match=r".*and.*or.*"):
        _ = FieldOperatorParser.apply_to_function(bool_and)


def test_boolean_or_op_unsupported():
    def bool_or(a: gtx.Field[[TDim], "bool"], b: gtx.Field[[TDim], "bool"]):
        return a or b

    with pytest.raises(errors.UnsupportedPythonFeatureError, match=r".*and.*or.*"):
        _ = FieldOperatorParser.apply_to_function(bool_or)


def test_bool_xor():
    def bool_xor(a: gtx.Field[[TDim], "bool"], b: gtx.Field[[TDim], "bool"]):
        return a ^ b

    parsed = FieldOperatorParser.apply_to_function(bool_xor)

    assert parsed.body.stmts[-1].value.type == ts.FieldType(
        dims=[TDim], dtype=ts.ScalarType(kind=ts.ScalarKind.BOOL, shape=None)
    )


def test_unary_tilde():
    def unary_tilde(a: gtx.Field[[TDim], "bool"]):
        return ~a

    parsed = FieldOperatorParser.apply_to_function(unary_tilde)

    assert parsed.body.stmts[-1].value.type == ts.FieldType(
        dims=[TDim], dtype=ts.ScalarType(kind=ts.ScalarKind.BOOL, shape=None)
    )


def test_scalar_cast_disallow_non_literals():
    def cast_scalar_temp():
        tmp = int64(1)
        return int32(tmp)

    with pytest.raises(errors.DSLError, match=r".*literal.*"):
        _ = FieldOperatorParser.apply_to_function(cast_scalar_temp)


def test_conditional_wrong_mask_type():
    def conditional_wrong_mask_type(a: gtx.Field[[TDim], float64]) -> gtx.Field[[TDim], float64]:
        return where(a, a, a)

    msg = r"expected a field with dtype 'bool'"
    with pytest.raises(errors.DSLError, match=msg):
        _ = FieldOperatorParser.apply_to_function(conditional_wrong_mask_type)


def test_conditional_wrong_arg_type():
    def conditional_wrong_arg_type(
        mask: gtx.Field[[TDim], bool], a: gtx.Field[[TDim], float32], b: gtx.Field[[TDim], float64]
    ) -> gtx.Field[[TDim], float64]:
        return where(mask, a, b)

    msg = r"Could not promote scalars of different dtype \(not implemented\)."
    with pytest.raises(errors.DSLError) as exc_info:
        _ = FieldOperatorParser.apply_to_function(conditional_wrong_arg_type)

    assert re.search(msg, exc_info.value.__cause__.args[0]) is not None


def test_ternary_with_field_condition():
    def ternary_with_field_condition(cond: gtx.Field[[], bool]):
        return 1 if cond else 2

    with pytest.raises(errors.DSLError, match=r"should be .* 'bool'"):
        _ = FieldOperatorParser.apply_to_function(ternary_with_field_condition)


def test_correct_return_type_annotation():
    """See ADR 13."""

    def correct_return_type_annotation() -> float:
        return 1.0

    FieldOperatorParser.apply_to_function(correct_return_type_annotation)


def test_adr13_wrong_return_type_annotation():
    """See ADR 13."""

    def wrong_return_type_annotation() -> gtx.Field[[], float]:
        return 1.0

    with pytest.raises(errors.DSLError, match=r"expected 'float.*'"):
        _ = FieldOperatorParser.apply_to_function(wrong_return_type_annotation)


def test_adr13_fixed_return_type_annotation():
    """See ADR 13."""

    def fixed_return_type_annotation() -> gtx.Field[[], float]:
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
    def astype_fieldop(a: gtx.Field[[TDim], "int64"]) -> gtx.Field[[TDim], float64]:
        return astype(a, float64)

    parsed = FieldOperatorParser.apply_to_function(astype_fieldop)

    assert parsed.body.stmts[-1].value.type == ts.FieldType(
        dims=[TDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64, shape=None)
    )


# --- External symbols ---
def test_closure_symbols():
    import numpy as np

    from gt4py.eve.utils import FrozenNamespace

    nonlocals = FrozenNamespace(float_value=2.3, np_value=np.float32(3.4))

    def operator_with_refs(inp: gtx.Field[[TDim], "float64"], inp2: gtx.Field[[TDim], "float32"]):
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
    ADim = gtx.Dimension("ADim")
    BDim = gtx.Dimension("BDim")

    def wrong_return_type_annotation(a: gtx.Field[[ADim], float64]) -> gtx.Field[[BDim], float64]:
        return a

    with pytest.raises(
        errors.DSLError, match=r"Annotated return type does not match deduced return type"
    ):
        _ = FieldOperatorParser.apply_to_function(wrong_return_type_annotation)


def test_empty_dims_type():
    def empty_dims() -> gtx.Field[[], float]:
        return 1.0

    with pytest.raises(
        errors.DSLError, match=r"Annotated return type does not match deduced return type"
    ):
        _ = FieldOperatorParser.apply_to_function(empty_dims)


def test_zero_dims_ternary():
    ADim = gtx.Dimension("ADim")

    def zero_dims_ternary(
        cond: gtx.Field[[], float64], a: gtx.Field[[ADim], float64], b: gtx.Field[[ADim], float64]
    ):
        return a if cond == 1 else b

    msg = r"Incompatible datatypes in operator '=='"
    with pytest.raises(errors.DSLError, match=msg):
        _ = FieldOperatorParser.apply_to_function(zero_dims_ternary)


def test_domain_chained_comparison_failure():
    def domain_comparison(a: gtx.Field[[TDim], float], b: gtx.Field[[TDim], float]):
        return concat_where(0 < TDim < 42, a, b)

    with pytest.raises(
        errors.DSLError,
        match=r".*chain.*not.*allowed(?s:.)*\(0 < TDim\) & \(TDim < 42\).*",
    ):
        _ = FieldOperatorParser.apply_to_function(domain_comparison)


def test_field_chained_comparison_failure():
    def comparison(
        cond: gtx.Field[[TDim], float], a: gtx.Field[[TDim], float], b: gtx.Field[[TDim], float]
    ):
        return where(0.0 < cond < 42.0, a, b)

    with pytest.raises(
        errors.DSLError,
        match=r".*chain.*not.*allowed(?s:.)*\(0.0 < cond\) & \(cond < 42.0\).*",
    ):
        _ = FieldOperatorParser.apply_to_function(comparison)
