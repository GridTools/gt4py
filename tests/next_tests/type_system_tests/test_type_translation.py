# -*- coding: utf-8 -*-
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

#
import typing

import numpy as np
import pytest

from gt4py.eve import extended_typing as xtyping
from gt4py.next import common
from gt4py.next.ffront.fbuiltins import Dimension
from gt4py.next.type_system import type_specifications as ts, type_translation


class CustomInt32DType:
    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.int32)


IDim = Dimension("IDim")
JDim = Dimension("JDim")


@pytest.mark.parametrize(
    "value,expected",
    [
        ("float", ts.ScalarKind.FLOAT64),
        (float, ts.ScalarKind.FLOAT64),
        (np.float64, ts.ScalarKind.FLOAT64),
        (np.dtype(float), ts.ScalarKind.FLOAT64),
        (CustomInt32DType(), ts.ScalarKind.INT32),
    ],
)
def test_valid_scalar_kind(value, expected):
    assert type_translation.get_scalar_kind(value) == expected


def test_invalid_scalar_kind():
    with pytest.raises(common.GTTypeError, match="Non-trivial dtypes"):
        type_translation.get_scalar_kind(np.dtype("i4, (2,3)f8, f4"))


@pytest.mark.parametrize(
    "value,expected",
    [
        (bool, ts.ScalarType(kind=ts.ScalarKind.BOOL)),
        (np.int32, ts.ScalarType(kind=ts.ScalarKind.INT32)),
        (float, ts.ScalarType(kind=ts.ScalarKind.FLOAT64)),
        ("float", ts.ScalarType(kind=ts.ScalarKind.FLOAT64)),
        (
            typing.Tuple[int, float],
            ts.TupleType(
                types=[
                    ts.ScalarType(kind=ts.ScalarKind.INT),
                    ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
                ]
            ),
        ),
        (
            tuple[bool, typing.Tuple[int, float]],
            ts.TupleType(
                types=[
                    ts.ScalarType(kind=ts.ScalarKind.BOOL),
                    ts.TupleType(
                        types=[
                            ts.ScalarType(kind=ts.ScalarKind.INT),
                            ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
                        ]
                    ),
                ]
            ),
        ),
        (
            common.Field[[IDim], float],
            ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)),
        ),
        (
            common.Field[[IDim, JDim], float],
            ts.FieldType(
                dims=[Dimension("IDim"), Dimension("JDim")],
                dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
            ),
        ),
        (
            typing.Annotated[
                typing.Callable[["float", int], int], xtyping.CallableKwargsInfo(data={})
            ],
            ts.FunctionType(
                args=[
                    ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
                    ts.ScalarType(kind=ts.ScalarKind.INT),
                ],
                kwargs={},
                returns=ts.ScalarType(kind=ts.ScalarKind.INT),
            ),
        ),
        (typing.ForwardRef("float"), ts.ScalarType(kind=ts.ScalarKind.FLOAT64)),
        (
            typing.Annotated[float, "foo"],
            ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
        ),
        (
            typing.Annotated["float", "foo", "bar"],
            ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
        ),
        (
            typing.Annotated[typing.ForwardRef("float"), "foo"],
            ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
        ),
    ],
)
def test_make_symbol_type_from_typing(value, expected):
    assert type_translation.from_type_hint(value) == expected


def test_invalid_symbol_types():
    # Forward references
    with pytest.raises(type_translation.TypingError, match="undefined forward references"):
        type_translation.from_type_hint("foo")

    # Tuples
    with pytest.raises(type_translation.TypingError, match="least one argument"):
        type_translation.from_type_hint(typing.Tuple)
    with pytest.raises(type_translation.TypingError, match="least one argument"):
        type_translation.from_type_hint(tuple)

    with pytest.raises(type_translation.TypingError, match="Unbound tuples"):
        type_translation.from_type_hint(tuple[int, ...])
    with pytest.raises(type_translation.TypingError, match="Unbound tuples"):
        type_translation.from_type_hint(typing.Tuple["float", ...])

    # Fields
    with pytest.raises(type_translation.TypingError, match="Field type requires two arguments"):
        type_translation.from_type_hint(common.Field)
    with pytest.raises(type_translation.TypingError, match="Invalid field dimensions"):
        type_translation.from_type_hint(common.Field[int, int])
    with pytest.raises(type_translation.TypingError, match="Invalid field dimension"):
        type_translation.from_type_hint(common.Field[[int, int], int])

    with pytest.raises(type_translation.TypingError, match="Field dtype argument"):
        type_translation.from_type_hint(common.Field[[IDim], str])
    with pytest.raises(type_translation.TypingError, match="Field dtype argument"):
        type_translation.from_type_hint(common.Field[[IDim], None])

    # Functions
    with pytest.raises(
        type_translation.TypingError, match="Not annotated functions are not supported"
    ):
        type_translation.from_type_hint(typing.Callable)

    with pytest.raises(type_translation.TypingError, match="Invalid callable annotations"):
        type_translation.from_type_hint(typing.Callable[..., float])
    with pytest.raises(type_translation.TypingError, match="Invalid callable annotations"):
        type_translation.from_type_hint(typing.Callable[[int], str])
    with pytest.raises(type_translation.TypingError, match="Invalid callable annotations"):
        type_translation.from_type_hint(typing.Callable[[int], float])
