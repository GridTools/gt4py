# -*- coding: utf-8 -*-
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

import gt4py.next as gtx
from gt4py.eve import extended_typing as xtyping
from gt4py.next import common
from gt4py.next.type_system import type_specifications as ts, type_translation


class CustomInt32DType:
    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.int32)


IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")


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
    with pytest.raises(ValueError, match="Non-trivial dtypes"):
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
                    ts.ScalarType(kind=ts.ScalarKind.INT64),
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
                            ts.ScalarType(kind=ts.ScalarKind.INT64),
                            ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
                        ]
                    ),
                ]
            ),
        ),
        (
            gtx.Field[[IDim], float],
            ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)),
        ),
        (
            gtx.Field[[IDim, JDim], float],
            ts.FieldType(
                dims=[gtx.Dimension("IDim"), gtx.Dimension("JDim")],
                dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
            ),
        ),
        (
            typing.Annotated[
                typing.Callable[["float", int], int], xtyping.CallableKwargsInfo(data={})
            ],
            ts.FunctionType(
                pos_only_args=[
                    ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
                    ts.ScalarType(kind=ts.ScalarKind.INT64),
                ],
                pos_or_kw_args={},
                kw_only_args={},
                returns=ts.ScalarType(kind=ts.ScalarKind.INT64),
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
    with pytest.raises(ValueError, match="undefined forward references"):
        type_translation.from_type_hint("foo")

    # Tuples
    with pytest.raises(ValueError, match="least one argument"):
        type_translation.from_type_hint(typing.Tuple)
    with pytest.raises(ValueError, match="least one argument"):
        type_translation.from_type_hint(tuple)

    with pytest.raises(ValueError, match="Unbound tuples"):
        type_translation.from_type_hint(tuple[int, ...])
    with pytest.raises(ValueError, match="Unbound tuples"):
        type_translation.from_type_hint(typing.Tuple["float", ...])

    # Fields
    with pytest.raises(ValueError, match="Field type requires two arguments"):
        type_translation.from_type_hint(common.Field)
    with pytest.raises(ValueError, match="Invalid field dimensions"):
        type_translation.from_type_hint(common.Field[int, int])
    with pytest.raises(ValueError, match="Invalid field dimension"):
        type_translation.from_type_hint(common.Field[[int, int], int])

    with pytest.raises(ValueError, match="Field dtype argument"):
        type_translation.from_type_hint(common.Field[[IDim], str])
    with pytest.raises(ValueError, match="Field dtype argument"):
        type_translation.from_type_hint(common.Field[[IDim], None])

    # Functions
    with pytest.raises(ValueError, match="Not annotated functions are not supported"):
        type_translation.from_type_hint(typing.Callable)

    with pytest.raises(ValueError, match="Invalid callable annotations"):
        type_translation.from_type_hint(typing.Callable[..., float])
    with pytest.raises(ValueError, match="Invalid callable annotations"):
        type_translation.from_type_hint(typing.Callable[[int], str])
    with pytest.raises(ValueError, match="Invalid callable annotations"):
        type_translation.from_type_hint(typing.Callable[[int], float])
