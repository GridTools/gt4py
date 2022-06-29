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

from functional.common import Field
from functional.ffront import common_types
from functional.ffront.fbuiltins import float32, float64, int32, int64, where
from functional.ffront.foast_passes.type_deduction import FieldOperatorTypeDeductionError
from functional.ffront.func_to_foast import FieldOperatorParser, FieldOperatorSyntaxError
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

    def rettype(inp: Field[..., float64]) -> Field[..., float64]:
        return inp

    parsed = FieldOperatorParser.apply_to_function(rettype)

    assert parsed.body[-1].value.type == common_types.FieldType(
        dims=Ellipsis,
        dtype=common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64, shape=None),
    )


def test_invalid_syntax_no_return():
    """Field operators must end with a return statement."""

    def no_return(inp: Field[..., "float64"]):
        tmp = inp  # noqa

    with pytest.raises(
        FieldOperatorSyntaxError,
        match="Field operator must return a field expression on the last line!",
    ):
        _ = FieldOperatorParser.apply_to_function(no_return)


def test_invalid_assign_to_expr():
    """Assigning to subscripts disallowed until a usecase can be found."""

    def invalid_assign_to_expr(inp1: Field[..., "float64"], inp2: Field[..., "float64"]):
        tmp = inp1
        tmp[-1] = inp2
        return tmp

    with pytest.raises(FieldOperatorSyntaxError, match=r"Can only assign to names! \(.*\)"):
        _ = FieldOperatorParser.apply_to_function(invalid_assign_to_expr)


def test_temp_assignment():
    def copy_field(inp: Field[..., "float64"]):
        tmp = inp
        inp = tmp
        tmp2 = inp
        return tmp2

    parsed = FieldOperatorParser.apply_to_function(copy_field)

    assert parsed.annex.symtable["tmp__0"].type == common_types.FieldType(
        dims=Ellipsis,
        dtype=common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64, shape=None),
    )


def test_clashing_annotated_assignment():
    def clashing(inp: Field[..., "float64"]):
        tmp: Field[..., "int64"] = inp
        return tmp

    with pytest.raises(FieldOperatorTypeDeductionError, match="type inconsistency"):
        _ = FieldOperatorParser.apply_to_function(clashing)


def test_binary_pow():
    def power(inp: Field[..., "float64"]):
        return inp**3

    parsed = FieldOperatorParser.apply_to_function(power)

    assert parsed.body[-1].value.type == common_types.FieldType(
        dims=Ellipsis,
        dtype=common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64, shape=None),
    )


def test_binary_mod():
    def power(inp: Field[..., "int64"]):
        return inp % 3

    with pytest.raises(
        FieldOperatorSyntaxError,
        match=(r"`%` operator not supported!"),
    ):
        _ = FieldOperatorParser.apply_to_function(power)


def test_bool_and():
    def bool_and(a: Field[..., "bool"], b: Field[..., "bool"]):
        return a and b

    with pytest.raises(
        FieldOperatorSyntaxError,
        match=(r"`and`/`or` operator not allowed!"),
    ):
        _ = FieldOperatorParser.apply_to_function(bool_and)


def test_bool_or():
    def bool_or(a: Field[..., "bool"], b: Field[..., "bool"]):
        return a or b

    with pytest.raises(
        FieldOperatorSyntaxError,
        match=(r"`and`/`or` operator not allowed!"),
    ):
        _ = FieldOperatorParser.apply_to_function(bool_or)


def test_scalar_cast():
    def cast_scalar_temp():
        tmp = int64(1)
        return int32(tmp)

    with pytest.raises(FieldOperatorSyntaxError, match=(r"only takes literal arguments!")):
        _ = FieldOperatorParser.apply_to_function(cast_scalar_temp)


def test_conditional_wrong_mask_type():
    def conditional_wrong_mask_type(
        a: Field[..., float64],
    ) -> Field[..., float64]:
        return where(a, a, a)

    msg = r"Expected a field with dtype bool."
    with pytest.raises(FieldOperatorTypeDeductionError, match=msg):
        _ = FieldOperatorParser.apply_to_function(conditional_wrong_mask_type)


def test_conditional_wrong_arg_type():
    def conditional_wrong_arg_type(
        mask: Field[..., bool],
        a: Field[..., float32],
        b: Field[..., float64],
    ) -> Field[..., float64]:
        return where(mask, a, b)

    msg = r"Could not promote scalars of different dtype \(not implemented\)."
    with pytest.raises(FieldOperatorTypeDeductionError) as exc_info:
        _ = FieldOperatorParser.apply_to_function(conditional_wrong_arg_type)

    assert re.search(msg, exc_info.value.__context__.args[0]) is not None


# --- External symbols ---
def test_closure_symbols():
    import numpy as np

    nonlocal_unused = 0  # noqa: F841
    nonlocal_float = 2.3
    nonlocal_np_scalar = np.float32(3.4)

    def operator_with_refs(inp: Field[..., "float64"], inp2: Field[..., "float32"]):
        a = inp + nonlocal_float
        b = inp2 + nonlocal_np_scalar
        return a, b

    parsed = FieldOperatorParser.apply_to_function(operator_with_refs)
    assert parsed.annex.symtable["nonlocal_float"].type == common_types.ScalarType(
        kind=common_types.ScalarKind.FLOAT64, shape=None
    )
    assert parsed.annex.symtable["nonlocal_np_scalar"].type == common_types.ScalarType(
        kind=common_types.ScalarKind.FLOAT32, shape=None
    )
    assert "nonlocal_unused" not in parsed.annex.symtable


def test_external_symbols():
    import numpy as np

    def operator_with_externals(inp: Field[..., "float64"], inp2: Field[..., "float32"]):
        from __externals__ import ext_float, ext_np_scalar

        a = inp + ext_float
        b = inp2 + ext_np_scalar
        return a, b

    parsed = FieldOperatorParser.apply_to_function(
        operator_with_externals,
        externals=dict(ext_float=2.3, ext_np_scalar=np.float32(3.4), ext_unused=0),
    )
    assert parsed.annex.symtable["ext_float"].type == common_types.ScalarType(
        kind=common_types.ScalarKind.FLOAT64, shape=None
    )
    assert parsed.annex.symtable["ext_np_scalar"].type == common_types.ScalarType(
        kind=common_types.ScalarKind.FLOAT32, shape=None
    )
    assert "ext_unused" not in parsed.annex.symtable
