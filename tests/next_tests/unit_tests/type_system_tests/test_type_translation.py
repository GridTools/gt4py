# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import typing

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py import eve
from gt4py._core import definitions as core_defs
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
        (typing.Annotated[float, "foo"], ts.ScalarType(kind=ts.ScalarKind.FLOAT64)),
        (typing.Annotated["float", "foo", "bar"], ts.ScalarType(kind=ts.ScalarKind.FLOAT64)),
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
    with pytest.raises(ValueError, match="Unannotated functions are not supported"):
        type_translation.from_type_hint(typing.Callable)

    with pytest.raises(ValueError, match="Invalid callable annotations"):
        type_translation.from_type_hint(typing.Callable[..., float])
    with pytest.raises(ValueError, match="Invalid callable annotations"):
        type_translation.from_type_hint(typing.Callable[[int], str])
    with pytest.raises(ValueError, match="Invalid callable annotations"):
        type_translation.from_type_hint(typing.Callable[[int], float])


@pytest.mark.parametrize(
    "value, expected_dims",
    [
        (common.Dims[IDim, JDim], [IDim, JDim]),
        (common.Dims[IDim, np.float64], ValueError),
        (common.Dims["IDim"], ValueError),
    ],
)
def test_generic_variadic_dims(value, expected_dims):
    if expected_dims == ValueError:
        with pytest.raises(ValueError, match="Invalid field dimension definition"):
            type_translation.from_type_hint(gtx.Field[value, np.int32])
    else:
        assert type_translation.from_type_hint(gtx.Field[value, np.int32]).dims == expected_dims


@pytest.mark.parametrize(
    "dtype",
    [
        core_defs.BoolDType(),
        core_defs.Int32DType(),
        core_defs.Int64DType(),
        core_defs.Float32DType(),
        core_defs.Float64DType(),
    ],
)
def test_as_from_dtype(dtype):
    assert type_translation.as_dtype(type_translation.from_dtype(dtype)) == dtype


def test_from_value_module():
    import next_tests.artifacts.dummy_package as dummy_package

    # TODO(egparedes): the following import should not be necessary
    #  but it seems to fail if the import is not here. It should be
    #  investigated.
    import next_tests.artifacts.dummy_package.dummy_module

    assert isinstance(
        type_translation.from_value(dummy_package), type_translation.UnknownPythonObject
    )
    assert type_translation.from_value(dummy_package).dummy_module.dummy_int == ts.ScalarType(
        kind=ts.ScalarKind.INT32
    )
    assert type_translation.from_value(dummy_package.dummy_module.dummy_int) == ts.ScalarType(
        kind=ts.ScalarKind.INT32
    )


class SomeEnum(eve.IntEnum):
    FOO = 1


@pytest.mark.parametrize(
    "value, type_, expected",
    [
        (gtx.int32(1), ts.ScalarType(kind=ts.ScalarKind.INT32), gtx.int32(1)),
        (gtx.int64(1), ts.ScalarType(kind=ts.ScalarKind.INT64), gtx.int64(1)),
        (1.0, ts.ScalarType(kind=ts.ScalarKind.INT64), gtx.int64(1)),
        (1, ts.ScalarType(kind=ts.ScalarKind.INT32), gtx.int32(1)),
        (True, ts.ScalarType(kind=ts.ScalarKind.BOOL), np.bool_(True)),
        (False, ts.ScalarType(kind=ts.ScalarKind.BOOL), np.bool_(False)),
        (SomeEnum.FOO, ts.ScalarType(kind=ts.ScalarKind.INT32), gtx.int32(1)),
        (
            (1, (2.0, gtx.float32(3.0))),
            ts.TupleType(
                types=[
                    ts.ScalarType(kind=ts.ScalarKind.INT32),
                    ts.TupleType(
                        types=[
                            ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
                            ts.ScalarType(kind=ts.ScalarKind.FLOAT32),
                        ]
                    ),
                ]
            ),
            (gtx.float32(1), (gtx.float64(2.0), gtx.float32(3.0))),
        ),
    ],
)
def test_unsafe_cast_to(value, type_, expected):
    result = type_translation.unsafe_cast_to(value, type_)
    assert result == expected
    assert type(result) is type(expected)
