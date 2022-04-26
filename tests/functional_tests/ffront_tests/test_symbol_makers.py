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

from eve import extended_typing as xtyping
from functional import common
from functional.ffront import common_types
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
        ("float", common_types.ScalarKind.FLOAT64),
        (float, common_types.ScalarKind.FLOAT64),
        (np.float64, common_types.ScalarKind.FLOAT64),
        (np.dtype(float), common_types.ScalarKind.FLOAT64),
        (CustomInt32DType(), common_types.ScalarKind.INT32),
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
        (bool, common_types.ScalarType(kind=common_types.ScalarKind.BOOL)),
        (np.int32, common_types.ScalarType(kind=common_types.ScalarKind.INT32)),
        (float, common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64)),
        ("float", common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64)),
        (
            typing.Tuple[int, float],
            common_types.TupleType(
                types=[
                    common_types.ScalarType(kind=common_types.ScalarKind.INT),
                    common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64),
                ]
            ),
        ),
        (
            tuple[bool, typing.Tuple[int, float]],
            common_types.TupleType(
                types=[
                    common_types.ScalarType(kind=common_types.ScalarKind.BOOL),
                    common_types.TupleType(
                        types=[
                            common_types.ScalarType(kind=common_types.ScalarKind.INT),
                            common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64),
                        ]
                    ),
                ]
            ),
        ),
        (
            common.Field[..., float],
            common_types.FieldType(
                dims=..., dtype=common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64)
            ),
        ),
        (
            common.Field[[IDim, JDim], float],
            common_types.FieldType(
                dims=[common.Dimension("IDim"), common.Dimension("JDim")],
                dtype=common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64),
            ),
        ),
        (
            typing.Annotated[
                typing.Callable[["float", int], int], xtyping.CallableKwargsInfo(data={})
            ],
            common_types.FunctionType(
                args=[
                    common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64),
                    common_types.ScalarType(kind=common_types.ScalarKind.INT),
                ],
                kwargs={},
                returns=common_types.ScalarType(kind=common_types.ScalarKind.INT),
            ),
        ),
        (typing.ForwardRef("float"), common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64)),
        (
            typing.Annotated[float, "foo"],
            common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64),
        ),
        (
            typing.Annotated["float", "foo", "bar"],
            common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64),
        ),
        (
            typing.Annotated[typing.ForwardRef("float"), "foo"],
            common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64),
        ),
    ],
)
def test_make_symbol_type_from_typing(value, expected):
    assert symbol_makers.make_symbol_type_from_typing(value) == expected


def test_invalid_symbol_types():
    # Forward references
    with pytest.raises(symbol_makers.TypingError, match="undefined forward references"):
        symbol_makers.make_symbol_type_from_typing("foo")

    # Tuples
    with pytest.raises(symbol_makers.TypingError, match="least one argument"):
        symbol_makers.make_symbol_type_from_typing(typing.Tuple)
    with pytest.raises(symbol_makers.TypingError, match="least one argument"):
        symbol_makers.make_symbol_type_from_typing(tuple)

    with pytest.raises(symbol_makers.TypingError, match="Unbound tuples"):
        symbol_makers.make_symbol_type_from_typing(tuple[int, ...])
    with pytest.raises(symbol_makers.TypingError, match="Unbound tuples"):
        symbol_makers.make_symbol_type_from_typing(typing.Tuple["float", ...])

    # Fields
    with pytest.raises(symbol_makers.TypingError, match="Field type requires two arguments"):
        symbol_makers.make_symbol_type_from_typing(common.Field)
    with pytest.raises(symbol_makers.TypingError, match="Invalid field dimensions"):
        symbol_makers.make_symbol_type_from_typing(common.Field[int, int])
    with pytest.raises(symbol_makers.TypingError, match="Invalid field dimension"):
        symbol_makers.make_symbol_type_from_typing(common.Field[[int, int], int])

    with pytest.raises(symbol_makers.TypingError, match="Field dtype argument"):
        symbol_makers.make_symbol_type_from_typing(common.Field[..., str])
    with pytest.raises(symbol_makers.TypingError, match="Field dtype argument"):
        symbol_makers.make_symbol_type_from_typing(common.Field[..., None])

    # Functions
    with pytest.raises(
        symbol_makers.TypingError, match="Not annotated functions are not supported"
    ):
        symbol_makers.make_symbol_type_from_typing(typing.Callable)

    with pytest.raises(symbol_makers.TypingError, match="Invalid callable annotations"):
        symbol_makers.make_symbol_type_from_typing(typing.Callable[..., float])
    with pytest.raises(symbol_makers.TypingError, match="Invalid callable annotations"):
        symbol_makers.make_symbol_type_from_typing(typing.Callable[[int], str])
    with pytest.raises(symbol_makers.TypingError, match="Invalid callable annotations"):
        symbol_makers.make_symbol_type_from_typing(typing.Callable[[int], float])

    with pytest.raises(symbol_makers.TypingError, match="'<class 'str'>' type is not supported"):
        symbol_makers.make_symbol_type_from_typing(
            typing.Annotated[
                typing.Callable[["float", int], str], xtyping.CallableKwargsInfo(data={})
            ]
        )
