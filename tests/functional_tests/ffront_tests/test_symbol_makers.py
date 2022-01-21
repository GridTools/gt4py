# -*- coding: utf-8 -*-
#
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

import typing

import numpy as np
import pytest

from eve import typingx
from functional import common
from functional.ffront import field_operator_ast as foast
from functional.ffront import symbol_makers
from functional.ffront.fbuiltins import Field, float64


class CustomInt32DType:
    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.int32)


IDim = common.Dimension("IDim")
JDim = common.Dimension("JDim")


@pytest.mark.parametrize(
    "value,expected",
    [
        ("float", foast.ScalarKind.FLOAT64),
        (float, foast.ScalarKind.FLOAT64),
        (np.float64, foast.ScalarKind.FLOAT64),
        (np.dtype(float), foast.ScalarKind.FLOAT64),
        (CustomInt32DType(), foast.ScalarKind.INT32),
    ],
)
def test_valid_scalar_kind(value, expected):
    assert symbol_makers.make_scalar_kind(value) == expected


def test_invalid_scalar_kind():
    with pytest.raises(common.GTTypeError, match="Non-trivial dtypes"):
        symbol_makers.make_scalar_kind(np.dtype("i4, (2,3)f8, f4"))


@pytest.mark.parametrize(
    "value,expected",
    [
        (bool, foast.ScalarType(kind=foast.ScalarKind.BOOL)),
        (np.int32, foast.ScalarType(kind=foast.ScalarKind.INT32)),
        (float, foast.ScalarType(kind=foast.ScalarKind.FLOAT64)),
        ("float", foast.ScalarType(kind=foast.ScalarKind.FLOAT64)),
        (
            typing.Tuple[int, float],
            foast.TupleType(
                types=[
                    foast.ScalarType(kind=foast.ScalarKind.INT64),
                    foast.ScalarType(kind=foast.ScalarKind.FLOAT64),
                ]
            ),
        ),
        (
            tuple[bool, typing.Tuple[int, float]],
            foast.TupleType(
                types=[
                    foast.ScalarType(kind=foast.ScalarKind.BOOL),
                    foast.TupleType(
                        types=[
                            foast.ScalarType(kind=foast.ScalarKind.INT64),
                            foast.ScalarType(kind=foast.ScalarKind.FLOAT64),
                        ]
                    ),
                ]
            ),
        ),
        (
            common.Field[..., float],
            foast.FieldType(dims=..., dtype=foast.ScalarType(kind=foast.ScalarKind.FLOAT64)),
        ),
        (
            common.Field[[IDim, JDim], float],
            foast.FieldType(
                dims=[foast.Dimension(name="IDim"), foast.Dimension(name="JDim")],
                dtype=foast.ScalarType(kind=foast.ScalarKind.FLOAT64),
            ),
        ),
        (
            typing.Annotated[
                typing.Callable[["float", int], int], typingx.CallableKwargsInfo(data={})
            ],
            foast.FunctionType(
                args=[
                    foast.ScalarType(kind=foast.ScalarKind.FLOAT64),
                    foast.ScalarType(kind=foast.ScalarKind.INT64),
                ],
                kwargs={},
                returns=foast.ScalarType(kind=foast.ScalarKind.INT64),
            ),
        ),
        (typing.ForwardRef("float"), foast.ScalarType(kind=foast.ScalarKind.FLOAT64)),
        (typing.Annotated[float, "foo"], foast.ScalarType(kind=foast.ScalarKind.FLOAT64)),
        (typing.Annotated["float", "foo", "bar"], foast.ScalarType(kind=foast.ScalarKind.FLOAT64)),
        (
            typing.Annotated[typing.ForwardRef("float"), "foo"],
            foast.ScalarType(kind=foast.ScalarKind.FLOAT64),
        ),
    ],
)
def test_make_symbol_type_from_typing(value, expected):
    assert symbol_makers.make_symbol_type_from_typing(value) == expected


def test_invalid_symbol_types():
    # Forward references
    with pytest.raises(symbol_makers.FieldOperatorTypeError, match="undefined forward references"):
        symbol_makers.make_symbol_type_from_typing("foo")

    # Tuples
    with pytest.raises(symbol_makers.FieldOperatorTypeError, match="least one argument"):
        symbol_makers.make_symbol_type_from_typing(typing.Tuple)
    with pytest.raises(symbol_makers.FieldOperatorTypeError, match="least one argument"):
        symbol_makers.make_symbol_type_from_typing(tuple)

    with pytest.raises(symbol_makers.FieldOperatorTypeError, match="Unbound tuples"):
        symbol_makers.make_symbol_type_from_typing(tuple[int, ...])
    with pytest.raises(symbol_makers.FieldOperatorTypeError, match="Unbound tuples"):
        symbol_makers.make_symbol_type_from_typing(typing.Tuple["float", ...])

    # Fields
    with pytest.raises(
        symbol_makers.FieldOperatorTypeError, match="Field type requires two arguments"
    ):
        symbol_makers.make_symbol_type_from_typing(common.Field)
    with pytest.raises(symbol_makers.FieldOperatorTypeError, match="Invalid field dimensions"):
        symbol_makers.make_symbol_type_from_typing(common.Field[int, int])
    with pytest.raises(symbol_makers.FieldOperatorTypeError, match="Invalid field dimension"):
        symbol_makers.make_symbol_type_from_typing(common.Field[[int, int], int])

    with pytest.raises(symbol_makers.FieldOperatorTypeError, match="Field dtype argument"):
        symbol_makers.make_symbol_type_from_typing(common.Field[..., str])
    with pytest.raises(symbol_makers.FieldOperatorTypeError, match="Field dtype argument"):
        symbol_makers.make_symbol_type_from_typing(common.Field[..., None])

    # Functions
    with pytest.raises(
        symbol_makers.FieldOperatorTypeError, match="Not annotated functions are not supported"
    ):
        symbol_makers.make_symbol_type_from_typing(typing.Callable)

    with pytest.raises(symbol_makers.FieldOperatorTypeError, match="Invalid callable annotations"):
        symbol_makers.make_symbol_type_from_typing(typing.Callable[..., float])
    with pytest.raises(symbol_makers.FieldOperatorTypeError, match="Invalid callable annotations"):
        symbol_makers.make_symbol_type_from_typing(typing.Callable[[int], str])
    with pytest.raises(symbol_makers.FieldOperatorTypeError, match="Invalid callable annotations"):
        symbol_makers.make_symbol_type_from_typing(typing.Callable[[int], float])

    with pytest.raises(
        symbol_makers.FieldOperatorTypeError, match="'<class 'str'>' type is not supported"
    ):
        symbol_makers.make_symbol_type_from_typing(
            typing.Annotated[
                typing.Callable[["float", int], str], typingx.CallableKwargsInfo(data={})
            ]
        )
