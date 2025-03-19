# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import re
from typing import Optional, Pattern

import pytest

from gt4py.next import (
    Dimension,
    DimensionKind,
)
from gt4py.next.type_system import type_info, type_specifications as ts
from gt4py.next.ffront import type_specifications as ts_ffront
from gt4py.next.iterator.type_system import type_specifications as ts_it

TDim = Dimension("TDim")  # Meaningless dimension, used for tests.


def type_info_cases() -> list[tuple[Optional[ts.TypeSpec], dict]]:
    return [
        (ts.DeferredType(constraint=None), {"is_concrete": False}),
        (
            ts.DeferredType(constraint=ts.ScalarType),
            {"is_concrete": False, "type_class": ts.ScalarType},
        ),
        (
            ts.DeferredType(constraint=ts.FieldType),
            {"is_concrete": False, "type_class": ts.FieldType},
        ),
        (
            ts.ScalarType(kind=ts.ScalarKind.INT64),
            {
                "is_concrete": True,
                "type_class": ts.ScalarType,
                "is_arithmetic": True,
                "is_logical": False,
            },
        ),
        (
            ts.FieldType(dims=[TDim], dtype=ts.ScalarType(kind=ts.ScalarKind.BOOL)),
            {
                "is_concrete": True,
                "type_class": ts.FieldType,
                "is_arithmetic": False,
                "is_logical": True,
            },
        ),
    ]


def callable_type_info_cases():
    # reuse all the other test cases
    not_callable = [
        (symbol_type, [], {}, [r"Expected a callable type, got "], None)
        for symbol_type, attributes in type_info_cases()
        if not isinstance(symbol_type, ts.CallableType)
    ]

    IDim = Dimension("I")
    JDim = Dimension("J")
    KDim = Dimension("K", kind=DimensionKind.VERTICAL)

    bool_type = ts.ScalarType(kind=ts.ScalarKind.BOOL)
    float_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    int_type = ts.ScalarType(kind=ts.ScalarKind.INT64)
    field_type = ts.FieldType(dims=[Dimension("I")], dtype=float_type)
    tuple_type = ts.TupleType(types=[bool_type, field_type])
    nullary_func_type = ts.FunctionType(
        pos_only_args=[], pos_or_kw_args={}, kw_only_args={}, returns=ts.VoidType()
    )
    unary_func_type = ts.FunctionType(
        pos_only_args=[bool_type], pos_or_kw_args={}, kw_only_args={}, returns=ts.VoidType()
    )
    kw_only_arg_func_type = ts.FunctionType(
        pos_only_args=[], pos_or_kw_args={}, kw_only_args={"foo": bool_type}, returns=ts.VoidType()
    )
    kw_or_pos_arg_func_type = ts.FunctionType(
        pos_only_args=[], pos_or_kw_args={"foo": bool_type}, kw_only_args={}, returns=ts.VoidType()
    )
    pos_arg_and_kw_or_pos_arg_and_kw_only_arg_func_type = ts.FunctionType(
        pos_only_args=[bool_type],
        pos_or_kw_args={"foo": int_type},
        kw_only_args={"bar": float_type},
        returns=ts.VoidType(),
    )
    unary_tuple_arg_func_type = ts.FunctionType(
        pos_only_args=[tuple_type], pos_or_kw_args={}, kw_only_args={}, returns=ts.VoidType()
    )
    fieldop_type = ts_ffront.FieldOperatorType(
        definition=ts.FunctionType(
            pos_only_args=[field_type, float_type],
            pos_or_kw_args={},
            kw_only_args={},
            returns=field_type,
        )
    )
    scanop_type = ts_ffront.ScanOperatorType(
        axis=KDim,
        definition=ts.FunctionType(
            pos_only_args=[],
            pos_or_kw_args={"carry": float_type, "a": int_type, "b": int_type},
            kw_only_args={},
            returns=float_type,
        ),
    )
    tuple_scanop_type = ts_ffront.ScanOperatorType(
        axis=KDim,
        definition=ts.FunctionType(
            pos_only_args=[],
            pos_or_kw_args={"carry": float_type, "a": ts.TupleType(types=[int_type, int_type])},
            kw_only_args={},
            returns=float_type,
        ),
    )

    return [
        # func_type, pos_only_args, kwargs, expected incompatibilities, return type
        *not_callable,
        (nullary_func_type, [], {}, [], ts.VoidType()),
        (
            nullary_func_type,
            [bool_type],
            {},
            [r"Function takes 0 positional arguments, but 1 were given."],
            None,
        ),
        (
            nullary_func_type,
            [],
            {"foo": bool_type},
            [r"Got unexpected keyword argument 'foo'."],
            None,
        ),
        (
            unary_func_type,
            [],
            {},
            [r"Function takes 1 positional argument, but 0 were given."],
            None,
        ),
        (unary_func_type, [bool_type], {}, [], ts.VoidType()),
        (
            unary_func_type,
            [float_type],
            {},
            [r"Expected 1st argument to be of type 'bool', got 'float64'."],
            None,
        ),
        (
            kw_or_pos_arg_func_type,
            [],
            {},
            [
                r"Missing 1 required positional argument: 'foo'",
                r"Function takes 1 positional argument, but 0 were given.",
            ],
            None,
        ),
        # function with keyword-or-positional argument
        (kw_or_pos_arg_func_type, [], {"foo": bool_type}, [], ts.VoidType()),
        (
            kw_or_pos_arg_func_type,
            [],
            {"foo": float_type},
            [r"Expected argument 'foo' to be of type 'bool', got 'float64'."],
            None,
        ),
        (
            kw_or_pos_arg_func_type,
            [],
            {"bar": bool_type},
            [r"Got unexpected keyword argument 'bar'."],
            None,
        ),
        # function with keyword-only argument
        (kw_only_arg_func_type, [], {}, [r"Missing required keyword argument 'foo'."], None),
        (kw_only_arg_func_type, [], {"foo": bool_type}, [], ts.VoidType()),
        (
            kw_only_arg_func_type,
            [],
            {"foo": float_type},
            [r"Expected keyword argument 'foo' to be of type 'bool', got 'float64'."],
            None,
        ),
        (
            kw_only_arg_func_type,
            [],
            {"bar": bool_type},
            [r"Got unexpected keyword argument 'bar'."],
            None,
        ),
        # function with positional, keyword-or-positional, and keyword-only argument
        (
            pos_arg_and_kw_or_pos_arg_and_kw_only_arg_func_type,
            [],
            {},
            [
                r"Missing 1 required positional argument: 'foo'",
                r"Function takes 2 positional arguments, but 0 were given.",
                r"Missing required keyword argument 'bar'",
            ],
            None,
        ),
        (
            pos_arg_and_kw_or_pos_arg_and_kw_only_arg_func_type,
            [bool_type],
            {},
            [
                r"Function takes 2 positional arguments, but 1 were given.",
                r"Missing required keyword argument 'bar'",
            ],
            None,
        ),
        (
            pos_arg_and_kw_or_pos_arg_and_kw_only_arg_func_type,
            [bool_type],
            {"foo": int_type},
            [r"Missing required keyword argument 'bar'"],
            None,
        ),
        (
            pos_arg_and_kw_or_pos_arg_and_kw_only_arg_func_type,
            [bool_type],
            {"foo": int_type},
            [r"Missing required keyword argument 'bar'"],
            None,
        ),
        (
            pos_arg_and_kw_or_pos_arg_and_kw_only_arg_func_type,
            [bool_type, bool_type],
            {"bar": float_type, "foo": int_type},
            [r"G"],
            None,
        ),
        (
            pos_arg_and_kw_or_pos_arg_and_kw_only_arg_func_type,
            [int_type],
            {"bar": bool_type, "foo": bool_type},
            [
                r"Expected 1st argument to be of type 'bool', got 'int64'",
                r"Expected argument 'foo' to be of type 'int64', got 'bool'",
                r"Expected keyword argument 'bar' to be of type 'float64', got 'bool'",
            ],
            None,
        ),
        (
            pos_arg_and_kw_or_pos_arg_and_kw_only_arg_func_type,
            [bool_type],
            {"bar": float_type, "foo": int_type},
            [],
            ts.VoidType(),
        ),
        (unary_tuple_arg_func_type, [tuple_type], {}, [], ts.VoidType()),
        (
            unary_tuple_arg_func_type,
            [ts.TupleType(types=[float_type, field_type])],
            {},
            [
                r"Expected 1st argument to be of type 'tuple\[bool, Field\[\[I\], float64\]\]', got 'tuple\[float64, Field\[\[I\], float64\]\]'"
            ],
            ts.VoidType(),
        ),
        (
            unary_tuple_arg_func_type,
            [int_type],
            {},
            [
                r"Expected 1st argument to be of type 'tuple\[bool, Field\[\[I\], float64\]\]', got 'int64'"
            ],
            ts.VoidType(),
        ),
        # field operator
        (fieldop_type, [field_type, float_type], {}, [], field_type),
        # scan operator
        (
            scanop_type,
            [],
            {},
            [r"Scan operator takes 2 positional arguments, but 0 were given."],
            ts.FieldType(dims=[KDim], dtype=float_type),
        ),
        (
            scanop_type,
            [
                ts.FieldType(dims=[KDim], dtype=float_type),
                ts.FieldType(dims=[KDim], dtype=float_type),
            ],
            {},
            [
                r"Expected argument 'a' to be of type 'Field\[\[K\], int64\]', got 'Field\[\[K\], float64\]'",
                r"Expected argument 'b' to be of type 'Field\[\[K\], int64\]', got 'Field\[\[K\], float64\]'",
            ],
            ts.FieldType(dims=[KDim], dtype=float_type),
        ),
        (
            scanop_type,
            [
                ts.FieldType(dims=[IDim, JDim], dtype=int_type),
                ts.FieldType(dims=[KDim], dtype=int_type),
            ],
            {},
            [],
            ts.FieldType(dims=[IDim, JDim, KDim], dtype=float_type),
        ),
        (
            scanop_type,
            [ts.FieldType(dims=[KDim], dtype=int_type), ts.FieldType(dims=[KDim], dtype=int_type)],
            {},
            [],
            ts.FieldType(dims=[KDim], dtype=float_type),
        ),
        (
            scanop_type,
            [
                ts.FieldType(dims=[IDim, JDim, KDim], dtype=int_type),
                ts.FieldType(dims=[IDim, JDim], dtype=int_type),
            ],
            {},
            [],
            ts.FieldType(dims=[IDim, JDim, KDim], dtype=float_type),
        ),
        (
            tuple_scanop_type,
            [
                ts.TupleType(
                    types=[
                        ts.FieldType(dims=[IDim, JDim, KDim], dtype=int_type),
                        ts.FieldType(dims=[IDim, JDim], dtype=int_type),
                    ]
                )
            ],
            {},
            [],
            ts.FieldType(dims=[IDim, JDim, KDim], dtype=float_type),
        ),
        (
            tuple_scanop_type,
            [ts.TupleType(types=[ts.FieldType(dims=[IDim, JDim, KDim], dtype=int_type)])],
            {},
            [
                r"Expected argument 'a' to be of type 'tuple\[Field\[\[I, J, K\], int64\], "
                r"Field\[\[\.\.\.\], int64\]\]', got 'tuple\[Field\[\[I, J, K\], int64\]\]'."
            ],
            ts.FieldType(dims=[IDim, JDim, KDim], dtype=float_type),
        ),
        (
            ts.FunctionType(
                pos_only_args=[
                    ts_it.IteratorType(
                        position_dims="unknown", defined_dims=[], element_type=float_type
                    ),
                ],
                pos_or_kw_args={},
                kw_only_args={},
                returns=ts.VoidType(),
            ),
            [ts_it.IteratorType(position_dims=[IDim], defined_dims=[], element_type=float_type)],
            {},
            [],
            ts.VoidType(),
        ),
    ]


@pytest.mark.parametrize("symbol_type,expected", type_info_cases())
def test_type_info_basic(symbol_type, expected):
    for key in expected:
        assert getattr(type_info, key)(symbol_type) == expected[key]


@pytest.mark.parametrize("func_type,args,kwargs,expected,return_type", callable_type_info_cases())
def test_accept_args(
    func_type: ts.TypeSpec,
    args: list[ts.TypeSpec],
    kwargs: dict[str, ts.TypeSpec],
    expected: list,
    return_type: ts.TypeSpec,
):
    accepts_args = len(expected) == 0
    assert accepts_args == type_info.accepts_args(func_type, with_args=args, with_kwargs=kwargs)

    if len(expected) > 0:
        with pytest.raises(ValueError) as exc_info:
            type_info.accepts_args(
                func_type, with_args=args, with_kwargs=kwargs, raise_exception=True
            )

        for expected_msg in expected:
            assert exc_info.match(expected_msg)


@pytest.mark.parametrize("func_type,args,kwargs,expected,return_type", callable_type_info_cases())
def test_return_type(
    func_type: ts.TypeSpec,
    args: list[ts.TypeSpec],
    kwargs: dict[str, ts.TypeSpec],
    expected: list,
    return_type: ts.TypeSpec,
):
    accepts_args = type_info.accepts_args(func_type, with_args=args, with_kwargs=kwargs)
    if accepts_args:
        assert type_info.return_type(func_type, with_args=args, with_kwargs=kwargs) == return_type
