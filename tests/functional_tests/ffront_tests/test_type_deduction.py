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

from functional.ffront import field_operator_ast as foast
from functional.ffront.builtins import Field, float64, int64
from functional.ffront.foast_passes.type_deduction import FieldOperatorTypeDeductionError
from functional.ffront.func_to_foast import FieldOperatorParser


def test_unpack_assign():
    def unpack_explicit_tuple(
        a: Field[..., float64], b: Field[..., float64]
    ) -> tuple[Field[..., float64], Field[..., float64]]:
        tmp_a, tmp_b = (a, b)
        return tmp_a, tmp_b

    parsed = FieldOperatorParser.apply_to_func(unpack_explicit_tuple)

    assert parsed.symtable_["tmp_a$0"].type == foast.FieldType(
        dims=Ellipsis, dtype=foast.ScalarType(kind=foast.ScalarKind.FLOAT64, shape=None)
    )
    assert parsed.symtable_["tmp_b$0"].type == foast.FieldType(
        dims=Ellipsis, dtype=foast.ScalarType(kind=foast.ScalarKind.FLOAT64, shape=None)
    )


def test_assign_tuple():
    def temp_tuple(a: Field[..., float64], b: Field[..., int64]):
        tmp = a, b
        return tmp

    parsed = FieldOperatorParser.apply_to_func(temp_tuple)

    assert parsed.symtable_["tmp$0"].type == foast.TupleType(
        types=[
            foast.FieldType(
                dims=Ellipsis, dtype=foast.ScalarType(kind=foast.ScalarKind.FLOAT64, shape=None)
            ),
            foast.FieldType(
                dims=Ellipsis, dtype=foast.ScalarType(kind=foast.ScalarKind.INT64, shape=None)
            ),
        ]
    )


def test_adding_bool():
    """Expect an error (or at least a warnign) when using arithmetic on bools."""

    def add_bools(a: Field[..., bool], b: Field[..., bool]):
        return a + b

    with pytest.raises(
        FieldOperatorTypeDeductionError,
        match=(
            r"Incompatible type\(s\) for operator '\+': "
            r"Field\[\.\.\., dtype=bool\], Field\[\.\.\., dtype=bool\]!"
        ),
    ):
        _ = FieldOperatorParser.apply_to_func(add_bools)


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
        _ = FieldOperatorParser.apply_to_func(float_bitop)


def test_signing_bool():
    def sign_bool(a: Field[..., bool]):
        return -a

    with pytest.raises(
        FieldOperatorTypeDeductionError,
        match=r"Incompatible type for unary operator '\-': Field\[\.\.\., dtype=bool\]!",
    ):
        _ = FieldOperatorParser.apply_to_func(sign_bool)


def test_notting_int():
    def not_int(a: Field[..., int64]):
        return not a

    with pytest.raises(
        FieldOperatorTypeDeductionError,
        match=r"Incompatible type for unary operator 'not': Field\[\.\.\., dtype=int64\]!",
    ):
        _ = FieldOperatorParser.apply_to_func(not_int)
