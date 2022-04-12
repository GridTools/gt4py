# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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
from typing import Optional

import pytest

from functional.common import Dimension, GTTypeError
from functional.ffront import common_types as ct
from functional.ffront.fbuiltins import Field, float32, float64, int64
from functional.ffront.foast_passes.type_deduction import FieldOperatorTypeDeductionError
from functional.ffront.func_to_foast import FieldOperatorParser
from functional.ffront.type_info import TypeInfo


def type_info_cases() -> list[tuple[Optional[ct.SymbolType], dict]]:
    return [
        (
            None,
            {
                "is_complete": False,
                "is_any_type": True,
                "constraint": None,
                "is_field_type": False,
                "is_scalar": False,
                "is_arithmetic_compatible": False,
                "is_logics_compatible": False,
                "is_callable": False,
            },
        ),
        (
            ct.DeferredSymbolType(constraint=None),
            {
                "is_complete": False,
                "is_any_type": True,
                "constraint": None,
                "is_field_type": False,
                "is_scalar": False,
                "is_arithmetic_compatible": False,
                "is_logics_compatible": False,
                "is_callable": False,
            },
        ),
        (
            ct.DeferredSymbolType(constraint=ct.ScalarType),
            {
                "is_complete": False,
                "is_any_type": False,
                "constraint": ct.ScalarType,
                "is_field_type": False,
                "is_scalar": True,
                "is_arithmetic_compatible": False,
                "is_logics_compatible": False,
                "is_callable": False,
            },
        ),
        (
            ct.DeferredSymbolType(constraint=ct.FieldType),
            {
                "is_complete": False,
                "is_any_type": False,
                "constraint": ct.FieldType,
                "is_field_type": True,
                "is_scalar": False,
                "is_arithmetic_compatible": False,
                "is_logics_compatible": False,
                "is_callable": False,
            },
        ),
        (
            ct.ScalarType(kind=ct.ScalarKind.INT64),
            {
                "is_complete": True,
                "is_any_type": False,
                "constraint": ct.ScalarType,
                "is_field_type": False,
                "is_scalar": True,
                "is_arithmetic_compatible": True,
                "is_logics_compatible": False,
                "is_callable": False,
            },
        ),
        (
            ct.FieldType(dims=Ellipsis, dtype=ct.ScalarType(kind=ct.ScalarKind.BOOL)),
            {
                "is_complete": True,
                "is_any_type": False,
                "constraint": ct.FieldType,
                "is_field_type": True,
                "is_scalar": False,
                "is_arithmetic_compatible": False,
                "is_logics_compatible": True,
                "is_callable": False,
            },
        ),
    ]


def type_info_is_callable_for_args_cases():
    # reuse all the other test cases
    not_callable = [
        (symbol_type, [], {}, [r"Expected a function type, but got "])
        for symbol_type, attributes in type_info_cases()
        if not attributes["is_callable"]
    ]

    bool_type = ct.ScalarType(kind=ct.ScalarKind.BOOL)
    float_type = ct.ScalarType(kind=ct.ScalarKind.FLOAT64)
    nullary_func_type = ct.FunctionType(args=[], kwargs={}, returns=ct.VoidType())
    unary_func_type = ct.FunctionType(args=[bool_type], kwargs={}, returns=ct.VoidType())
    kwarg_func_type = ct.FunctionType(args=[], kwargs={"foo": bool_type}, returns=ct.VoidType())

    return [
        # func_type, args, kwargs, expected incompatibilities
        *not_callable,
        (nullary_func_type, [], {}, []),
        (nullary_func_type, [bool_type], {}, [r"Function takes 0 arguments, but 1 were given."]),
        (
            nullary_func_type,
            [],
            {"foo": bool_type},
            [r"Got unexpected keyword argument\(s\) `foo`."],
        ),
        (unary_func_type, [], {}, [r"Function takes 1 arguments, but 0 were given."]),
        (unary_func_type, [bool_type], {}, []),
        (
            unary_func_type,
            [float_type],
            {},
            [r"Expected 0-th argument to be of type bool, but got float64."],
        ),
        (kwarg_func_type, [], {}, [r"Missing required keyword argument\(s\) `foo`."]),
        (kwarg_func_type, [], {"foo": bool_type}, []),
        (
            kwarg_func_type,
            [],
            {"foo": float_type},
            [r"Expected keyword argument foo to be of type bool, but got float64."],
        ),
        (kwarg_func_type, [], {"bar": bool_type}, [r"Got unexpected keyword argument\(s\) `bar`."]),
    ]


@pytest.mark.parametrize("symbol_type,expected", type_info_cases())
def test_type_info_basic(symbol_type, expected):
    typeinfo = TypeInfo(symbol_type)
    for key in expected:
        assert getattr(typeinfo, key) == expected[key]


def test_type_info_refinable_complete_complete():
    complete_type = ct.ScalarType(kind=ct.ScalarKind.INT64)
    other_complete_type = ct.ScalarType(kind=ct.ScalarKind.FLOAT64)
    type_info_a = TypeInfo(complete_type)
    type_info_b = TypeInfo(other_complete_type)
    assert type_info_a.can_be_refined_to(TypeInfo(complete_type))
    assert not type_info_a.can_be_refined_to(type_info_b)


def test_type_info_refinable_incomplete_complete():
    complete_type = TypeInfo(
        ct.FieldType(dtype=ct.ScalarType(kind=ct.ScalarKind.BOOL), dims=Ellipsis)
    )
    assert TypeInfo(None).can_be_refined_to(complete_type)
    assert TypeInfo(ct.DeferredSymbolType(constraint=None)).can_be_refined_to(complete_type)
    assert TypeInfo(ct.DeferredSymbolType(constraint=ct.FieldType)).can_be_refined_to(complete_type)
    assert not TypeInfo(ct.DeferredSymbolType(constraint=ct.OffsetType)).can_be_refined_to(
        complete_type
    )


def test_type_info_refinable_incomplete_incomplete():
    target_type = TypeInfo(ct.DeferredSymbolType(constraint=ct.ScalarType))
    assert TypeInfo(None).can_be_refined_to(target_type)
    assert TypeInfo(ct.DeferredSymbolType(constraint=None)).can_be_refined_to(target_type)
    assert TypeInfo(ct.DeferredSymbolType(constraint=ct.ScalarType)).can_be_refined_to(target_type)
    assert not TypeInfo(ct.DeferredSymbolType(constraint=ct.FieldType)).can_be_refined_to(
        target_type
    )


@pytest.mark.parametrize("func_type,args,kwargs,expected", type_info_is_callable_for_args_cases())
def test_type_info_is_callable_for_args_cases(
    func_type: ct.SymbolType,
    args: list[ct.SymbolType],
    kwargs: dict[str, ct.SymbolType],
    expected: list,
):
    typeinfo = TypeInfo(func_type)
    is_callable = len(expected) == 0
    assert typeinfo.is_callable_for_args(args, kwargs) == is_callable

    if len(expected) > 0:
        with pytest.raises(
            GTTypeError,
        ) as exc_info:
            typeinfo.is_callable_for_args(args, kwargs, raise_exception=True)

        for expected_msg in expected:
            assert exc_info.match(expected_msg)


def test_unpack_assign():
    def unpack_explicit_tuple(
        a: Field[..., float64], b: Field[..., float64]
    ) -> tuple[Field[..., float64], Field[..., float64]]:
        tmp_a, tmp_b = (a, b)
        return tmp_a, tmp_b

    parsed = FieldOperatorParser.apply_to_function(unpack_explicit_tuple)

    assert parsed.symtable_["tmp_a__0"].type == ct.FieldType(
        dims=Ellipsis,
        dtype=ct.ScalarType(kind=ct.ScalarKind.FLOAT64, shape=None),
    )
    assert parsed.symtable_["tmp_b__0"].type == ct.FieldType(
        dims=Ellipsis,
        dtype=ct.ScalarType(kind=ct.ScalarKind.FLOAT64, shape=None),
    )


def test_assign_tuple():
    def temp_tuple(a: Field[..., float64], b: Field[..., int64]):
        tmp = a, b
        return tmp

    parsed = FieldOperatorParser.apply_to_function(temp_tuple)

    assert parsed.symtable_["tmp__0"].type == ct.TupleType(
        types=[
            ct.FieldType(
                dims=Ellipsis,
                dtype=ct.ScalarType(kind=ct.ScalarKind.FLOAT64, shape=None),
            ),
            ct.FieldType(
                dims=Ellipsis,
                dtype=ct.ScalarType(kind=ct.ScalarKind.INT64, shape=None),
            ),
        ]
    )


def test_adding_bool():
    """Expect an error when using arithmetic on bools."""

    def add_bools(a: Field[..., bool], b: Field[..., bool]):
        return a + b

    with pytest.raises(
        FieldOperatorTypeDeductionError,
        match=(
            r"Incompatible type\(s\) for operator '\+': "
            r"Field\[\.\.\., dtype=bool\], Field\[\.\.\., dtype=bool\]!"
        ),
    ):
        _ = FieldOperatorParser.apply_to_function(add_bools)


def test_binop_nonmatching_dims():
    """Binary operations can only work when both fields have the same dimensions."""
    X = Dimension("X")
    Y = Dimension("Y")

    def nonmatching(a: Field[[X], float64], b: Field[[Y], float64]):
        return a + b

    with pytest.raises(
        FieldOperatorTypeDeductionError,
        match=(
            r"Incompatible type\(s\) for operator '\+': "
            r"Field\[\[X\], dtype=float64\], Field\[\[Y\], dtype=float64\]!"
        ),
    ):
        _ = FieldOperatorParser.apply_to_function(nonmatching)


def test_bitopping_float():
    def float_bitop(a: Field[..., float], b: Field[..., float]):
        return a & b

    with pytest.raises(
        FieldOperatorTypeDeductionError,
        match=(
            r"Incompatible type\(s\) for operator '\&': "
            r"Field\[\.\.\., dtype=float64\], Field\[\.\.\., dtype=float64\]!"
        ),
    ):
        _ = FieldOperatorParser.apply_to_function(float_bitop)


def test_signing_bool():
    def sign_bool(a: Field[..., bool]):
        return -a

    with pytest.raises(
        FieldOperatorTypeDeductionError,
        match=r"Incompatible type for unary operator '\-': Field\[\.\.\., dtype=bool\]!",
    ):
        _ = FieldOperatorParser.apply_to_function(sign_bool)


def test_notting_int():
    def not_int(a: Field[..., int64]):
        return not a

    with pytest.raises(
        FieldOperatorTypeDeductionError,
        match=r"Incompatible type for unary operator 'not': Field\[\.\.\., dtype=int64\]!",
    ):
        _ = FieldOperatorParser.apply_to_function(not_int)


def test_scalar_arg():
    def scalar_arg(bar: Field[..., int64], alpha: int64) -> Field[..., int64]:
        return alpha * bar

    parsed = FieldOperatorParser.apply_to_function(scalar_arg)

    assert parsed.params[1].id == "alpha"
    assert parsed.params[1].type == ct.FieldType(
        dims=[], dtype=ct.ScalarType(kind=ct.ScalarKind.INT64)
    )


def test_mismatched_literals():
    def mismatched_lit() -> Field[..., "float32"]:
        return float32("1.0") + float64("1.0")

    with pytest.raises(
        FieldOperatorTypeDeductionError,
        match=(r"Incompatible type\(s\) for operator '\+': float32, float64"),
    ):
        _ = FieldOperatorParser.apply_to_function(mismatched_lit)
