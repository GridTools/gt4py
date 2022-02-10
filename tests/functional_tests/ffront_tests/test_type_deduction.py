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

import pytest

from functional.common import Dimension
from functional.ffront import common_types
from functional.ffront import field_operator_ast as foast
from functional.ffront.fbuiltins import Field, float64, int64
from functional.ffront.foast_passes.type_deduction import FieldOperatorTypeDeductionError
from functional.ffront.func_to_foast import FieldOperatorParser
from functional.ffront.type_info import TypeInfo


def type_info_cases():
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
            },
        ),
        (
            common_types.DeferredSymbolType(constraint=None),
            {
                "is_complete": False,
                "is_any_type": True,
                "constraint": None,
                "is_field_type": False,
                "is_scalar": False,
                "is_arithmetic_compatible": False,
                "is_logics_compatible": False,
            },
        ),
        (
            common_types.DeferredSymbolType(constraint=common_types.ScalarType),
            {
                "is_complete": False,
                "is_any_type": False,
                "constraint": common_types.ScalarType,
                "is_field_type": False,
                "is_scalar": True,
                "is_arithmetic_compatible": False,
                "is_logics_compatible": False,
            },
        ),
        (
            common_types.DeferredSymbolType(constraint=common_types.FieldType),
            {
                "is_complete": False,
                "is_any_type": False,
                "constraint": common_types.FieldType,
                "is_field_type": True,
                "is_scalar": False,
                "is_arithmetic_compatible": False,
                "is_logics_compatible": False,
            },
        ),
        (
            common_types.ScalarType(kind=common_types.ScalarKind.INT64),
            {
                "is_complete": True,
                "is_any_type": False,
                "constraint": common_types.ScalarType,
                "is_field_type": False,
                "is_scalar": True,
                "is_arithmetic_compatible": True,
                "is_logics_compatible": False,
            },
        ),
        (
            common_types.FieldType(
                dims=Ellipsis, dtype=common_types.ScalarType(kind=common_types.ScalarKind.BOOL)
            ),
            {
                "is_complete": True,
                "is_any_type": False,
                "constraint": common_types.FieldType,
                "is_field_type": True,
                "is_scalar": False,
                "is_arithmetic_compatible": False,
                "is_logics_compatible": True,
            },
        ),
    ]


@pytest.mark.parametrize("symbol_type,expected", type_info_cases())
def test_type_info_basic(symbol_type, expected):
    typeinfo = TypeInfo(symbol_type)
    for key in expected:
        assert getattr(typeinfo, key) == expected[key]


def test_type_info_refinable_complete_complete():
    complete_type = common_types.ScalarType(kind=common_types.ScalarKind.INT64)
    other_complete_type = common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64)
    type_info_a = TypeInfo(complete_type)
    type_info_b = TypeInfo(other_complete_type)
    assert type_info_a.can_be_refined_to(TypeInfo(complete_type))
    assert not type_info_a.can_be_refined_to(type_info_b)


def test_type_info_refinable_incomplete_complete():
    complete_type = TypeInfo(
        common_types.FieldType(
            dtype=common_types.ScalarType(kind=common_types.ScalarKind.BOOL), dims=Ellipsis
        )
    )
    assert TypeInfo(None).can_be_refined_to(complete_type)
    assert TypeInfo(common_types.DeferredSymbolType(constraint=None)).can_be_refined_to(
        complete_type
    )
    assert TypeInfo(
        common_types.DeferredSymbolType(constraint=common_types.FieldType)
    ).can_be_refined_to(complete_type)
    assert not TypeInfo(
        common_types.DeferredSymbolType(constraint=common_types.OffsetType)
    ).can_be_refined_to(complete_type)


def test_type_info_refinable_incomplete_incomplete():
    target_type = TypeInfo(common_types.DeferredSymbolType(constraint=common_types.ScalarType))
    assert TypeInfo(None).can_be_refined_to(target_type)
    assert TypeInfo(common_types.DeferredSymbolType(constraint=None)).can_be_refined_to(target_type)
    assert TypeInfo(
        common_types.DeferredSymbolType(constraint=common_types.ScalarType)
    ).can_be_refined_to(target_type)
    assert not TypeInfo(
        common_types.DeferredSymbolType(constraint=common_types.FieldType)
    ).can_be_refined_to(target_type)


def test_unpack_assign():
    def unpack_explicit_tuple(
        a: Field[..., float64], b: Field[..., float64]
    ) -> tuple[Field[..., float64], Field[..., float64]]:
        tmp_a, tmp_b = (a, b)
        return tmp_a, tmp_b

    parsed = FieldOperatorParser.apply_to_function(unpack_explicit_tuple)

    assert parsed.symtable_["tmp_a__0"].type == common_types.FieldType(
        dims=Ellipsis,
        dtype=common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64, shape=None),
    )
    assert parsed.symtable_["tmp_b__0"].type == common_types.FieldType(
        dims=Ellipsis,
        dtype=common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64, shape=None),
    )


def test_assign_tuple():
    def temp_tuple(a: Field[..., float64], b: Field[..., int64]):
        tmp = a, b
        return tmp

    parsed = FieldOperatorParser.apply_to_function(temp_tuple)

    assert parsed.symtable_["tmp__0"].type == common_types.TupleType(
        types=[
            common_types.FieldType(
                dims=Ellipsis,
                dtype=common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64, shape=None),
            ),
            common_types.FieldType(
                dims=Ellipsis,
                dtype=common_types.ScalarType(kind=common_types.ScalarKind.INT64, shape=None),
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
