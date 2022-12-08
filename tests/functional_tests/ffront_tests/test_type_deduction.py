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
from typing import Optional, Pattern

import pytest

from functional.common import DimensionKind, GTTypeError
from functional.ffront import common_types as ct, type_info
from functional.ffront.fbuiltins import (
    Dimension,
    Field,
    FieldOffset,
    broadcast,
    float32,
    float64,
    int64,
    neighbor_sum,
    where,
)
from functional.ffront.foast_passes.type_deduction import FieldOperatorTypeDeductionError
from functional.ffront.func_to_foast import FieldOperatorParser


def type_info_cases() -> list[tuple[Optional[ct.SymbolType], dict]]:
    return [
        (
            ct.DeferredSymbolType(constraint=None),
            {
                "is_concrete": False,
            },
        ),
        (
            ct.DeferredSymbolType(constraint=ct.ScalarType),
            {
                "is_concrete": False,
                "type_class": ct.ScalarType,
            },
        ),
        (
            ct.DeferredSymbolType(constraint=ct.FieldType),
            {
                "is_concrete": False,
                "type_class": ct.FieldType,
            },
        ),
        (
            ct.ScalarType(kind=ct.ScalarKind.INT64),
            {
                "is_concrete": True,
                "type_class": ct.ScalarType,
                "is_arithmetic": True,
                "is_logical": False,
            },
        ),
        (
            ct.FieldType(dims=Ellipsis, dtype=ct.ScalarType(kind=ct.ScalarKind.BOOL)),
            {
                "is_concrete": True,
                "type_class": ct.FieldType,
                "is_arithmetic": False,
                "is_logical": True,
            },
        ),
    ]


def callable_type_info_cases():
    # reuse all the other test cases
    not_callable = [
        (symbol_type, [], {}, [r"Expected a callable type, but got "], None)
        for symbol_type, attributes in type_info_cases()
        if not isinstance(symbol_type, ct.CallableType)
    ]

    IDim = Dimension("I")
    JDim = Dimension("J")
    KDim = Dimension("K", kind=DimensionKind.VERTICAL)

    bool_type = ct.ScalarType(kind=ct.ScalarKind.BOOL)
    float_type = ct.ScalarType(kind=ct.ScalarKind.FLOAT64)
    int_type = ct.ScalarType(kind=ct.ScalarKind.INT64)
    field_type = ct.FieldType(dims=[Dimension("I")], dtype=float_type)
    nullary_func_type = ct.FunctionType(args=[], kwargs={}, returns=ct.VoidType())
    unary_func_type = ct.FunctionType(args=[bool_type], kwargs={}, returns=ct.VoidType())
    kwarg_func_type = ct.FunctionType(args=[], kwargs={"foo": bool_type}, returns=ct.VoidType())
    fieldop_type = ct.FieldOperatorType(
        definition=ct.FunctionType(args=[field_type, float_type], kwargs={}, returns=field_type)
    )
    scanop_type = ct.ScanOperatorType(
        axis=KDim,
        definition=ct.FunctionType(
            args=[float_type, int_type, int_type], kwargs={}, returns=float_type
        ),
    )
    tuple_scanop_type = ct.ScanOperatorType(
        axis=KDim,
        definition=ct.FunctionType(
            args=[float_type, ct.TupleType(types=[int_type, int_type])],
            kwargs={},
            returns=float_type,
        ),
    )

    return [
        # func_type, args, kwargs, expected incompatibilities, return type
        *not_callable,
        (nullary_func_type, [], {}, [], ct.VoidType()),
        (
            nullary_func_type,
            [bool_type],
            {},
            [r"Function takes 0 argument\(s\), but 1 were given."],
            None,
        ),
        (
            nullary_func_type,
            [],
            {"foo": bool_type},
            [r"Got unexpected keyword argument\(s\) `foo`."],
            None,
        ),
        (unary_func_type, [], {}, [r"Function takes 1 argument\(s\), but 0 were given."], None),
        (unary_func_type, [bool_type], {}, [], ct.VoidType()),
        (
            unary_func_type,
            [float_type],
            {},
            [r"Expected 0-th argument to be of type bool, but got float64."],
            None,
        ),
        (kwarg_func_type, [], {}, [r"Missing required keyword argument\(s\) `foo`."], None),
        (kwarg_func_type, [], {"foo": bool_type}, [], ct.VoidType()),
        (
            kwarg_func_type,
            [],
            {"foo": float_type},
            [r"Expected keyword argument foo to be of type bool, but got float64."],
            None,
        ),
        (
            kwarg_func_type,
            [],
            {"bar": bool_type},
            [r"Got unexpected keyword argument\(s\) `bar`."],
            None,
        ),
        # field operator
        (fieldop_type, [field_type, float_type], {}, [], field_type),
        # scan operator
        (
            scanop_type,
            [],
            {},
            [r"Scan operator takes 2 arguments, but 0 were given."],
            ct.FieldType(dims=[KDim], dtype=float_type),
        ),
        (
            scanop_type,
            [
                ct.FieldType(dims=[KDim], dtype=float_type),
                ct.FieldType(dims=[KDim], dtype=float_type),
            ],
            {},
            [
                r"Expected 0-th argument to be of type Field\[\[K\], int64\], but got Field\[\[K\], float64\]",
                r"Expected 1-th argument to be of type Field\[\[K\], int64\], but got Field\[\[K\], float64\]",
            ],
            ct.FieldType(dims=[KDim], dtype=float_type),
        ),
        (
            scanop_type,
            [
                ct.FieldType(dims=[IDim, JDim], dtype=int_type),
                ct.FieldType(dims=[KDim], dtype=int_type),
            ],
            {},
            [
                r"Dimensions can not be promoted. Could not determine order of the "
                r"following dimensions: J, K."
            ],
            ct.FieldType(dims=[IDim, JDim, KDim], dtype=float_type),
        ),
        (
            scanop_type,
            [
                ct.FieldType(dims=[KDim], dtype=int_type),
                ct.FieldType(dims=[KDim], dtype=int_type),
            ],
            {},
            [],
            ct.FieldType(dims=[KDim], dtype=float_type),
        ),
        (
            scanop_type,
            [
                ct.FieldType(dims=[IDim, JDim, KDim], dtype=int_type),
                ct.FieldType(dims=[IDim, JDim], dtype=int_type),
            ],
            {},
            [],
            ct.FieldType(dims=[IDim, JDim, KDim], dtype=float_type),
        ),
        (
            tuple_scanop_type,
            [
                ct.TupleType(
                    types=[
                        ct.FieldType(dims=[IDim, JDim, KDim], dtype=int_type),
                        ct.FieldType(dims=[IDim, JDim], dtype=int_type),
                    ]
                )
            ],
            {},
            [],
            ct.FieldType(dims=[IDim, JDim, KDim], dtype=float_type),
        ),
        (
            tuple_scanop_type,
            [
                ct.TupleType(
                    types=[
                        ct.FieldType(dims=[IDim, JDim, KDim], dtype=int_type),
                    ]
                )
            ],
            {},
            [
                r"Expected 0-th argument to be of type tuple\[Field\[\[I, J, K\], int64\], "
                r"Field\[..., int64\]\], but got tuple\[Field\[\[I, J, K\], int64\]\]."
            ],
            ct.FieldType(dims=[IDim, JDim, KDim], dtype=float_type),
        ),
    ]


@pytest.mark.parametrize("symbol_type,expected", type_info_cases())
def test_type_info_basic(symbol_type, expected):
    for key in expected:
        assert getattr(type_info, key)(symbol_type) == expected[key]


@pytest.mark.parametrize("func_type,args,kwargs,expected,return_type", callable_type_info_cases())
def test_accept_args(
    func_type: ct.SymbolType,
    args: list[ct.SymbolType],
    kwargs: dict[str, ct.SymbolType],
    expected: list,
    return_type: ct.SymbolType,
):
    accepts_args = len(expected) == 0
    assert accepts_args == type_info.accepts_args(func_type, with_args=args, with_kwargs=kwargs)

    if len(expected) > 0:
        with pytest.raises(
            GTTypeError,
        ) as exc_info:
            type_info.accepts_args(
                func_type, with_args=args, with_kwargs=kwargs, raise_exception=True
            )

        for expected_msg in expected:
            assert exc_info.match(expected_msg)


@pytest.mark.parametrize("func_type,args,kwargs,expected,return_type", callable_type_info_cases())
def test_return_type(
    func_type: ct.SymbolType,
    args: list[ct.SymbolType],
    kwargs: dict[str, ct.SymbolType],
    expected: list,
    return_type: ct.SymbolType,
):
    accepts_args = type_info.accepts_args(func_type, with_args=args, with_kwargs=kwargs)
    if accepts_args:
        assert type_info.return_type(func_type, with_args=args, with_kwargs=kwargs) == return_type


def test_unpack_assign():
    def unpack_explicit_tuple(
        a: Field[..., float64], b: Field[..., float64]
    ) -> tuple[Field[..., float64], Field[..., float64]]:
        tmp_a, tmp_b = (a, b)
        return tmp_a, tmp_b

    parsed = FieldOperatorParser.apply_to_function(unpack_explicit_tuple)

    assert parsed.body.annex.symtable["tmp_a__0"].type == ct.FieldType(
        dims=Ellipsis,
        dtype=ct.ScalarType(kind=ct.ScalarKind.FLOAT64, shape=None),
    )
    assert parsed.body.annex.symtable["tmp_b__0"].type == ct.FieldType(
        dims=Ellipsis,
        dtype=ct.ScalarType(kind=ct.ScalarKind.FLOAT64, shape=None),
    )


def dimension_promotion_cases() -> list[
    tuple[list[list[Dimension]], list[Dimension] | None, None | Pattern]
]:
    raw_list = [
        # list of list of dimensions, expected result, expected error message
        ([["I", "J"], ["I"]], ["I", "J"], None),
        ([["I", "J"], ["J"]], ["I", "J"], None),
        ([["I", "J"], ["J", "K"]], ["I", "J", "K"], None),
        (
            [["I", "J"], ["J", "I"]],
            None,
            r"The following dimensions appear in contradicting order: I, J.",
        ),
        (
            [["I", "K"], ["J", "K"]],
            None,
            r"Could not determine order of the following dimensions: I, J",
        ),
    ]
    # transform dimension names into Dimension objects
    return [
        (
            [[Dimension(el) for el in arg] for arg in args],
            [Dimension(el) for el in result] if result else result,
            msg,
        )
        for args, result, msg in raw_list
    ]


@pytest.mark.parametrize("dim_list,expected_result,expected_error_msg", dimension_promotion_cases())
def test_dimension_promotion(
    dim_list: list[list[Dimension]],
    expected_result: Optional[list[Dimension]],
    expected_error_msg: Optional[str],
):
    if expected_result:
        assert type_info.promote_dims(*dim_list) == expected_result
    else:
        with pytest.raises(Exception) as exc_info:
            type_info.promote_dims(*dim_list)

        assert exc_info.match(expected_error_msg)


def test_assign_tuple():
    def temp_tuple(a: Field[..., float64], b: Field[..., int64]):
        tmp = a, b
        return tmp

    parsed = FieldOperatorParser.apply_to_function(temp_tuple)

    assert parsed.body.annex.symtable["tmp__0"].type == ct.TupleType(
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
        match=(r"Type Field\[\.\.\., bool\] can not be used in operator `\+`!"),
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
            r"Could not promote `Field\[\[X], float64\]` and `Field\[\[Y\], float64\]` to common type in call to +."
        ),
    ):
        _ = FieldOperatorParser.apply_to_function(nonmatching)


def test_bitopping_float():
    def float_bitop(a: Field[..., float], b: Field[..., float]):
        return a & b

    with pytest.raises(
        FieldOperatorTypeDeductionError,
        match=(r"Type Field\[\.\.\., float64\] can not be used in operator `\&`! "),
    ):
        _ = FieldOperatorParser.apply_to_function(float_bitop)


def test_signing_bool():
    def sign_bool(a: Field[..., bool]):
        return -a

    with pytest.raises(
        FieldOperatorTypeDeductionError,
        match=r"Incompatible type for unary operator `\-`: `Field\[\.\.\., bool\]`!",
    ):
        _ = FieldOperatorParser.apply_to_function(sign_bool)


def test_notting_int():
    def not_int(a: Field[..., int64]):
        return not a

    with pytest.raises(
        FieldOperatorTypeDeductionError,
        match=r"Incompatible type for unary operator `not`: `Field\[\.\.\., int64\]`!",
    ):
        _ = FieldOperatorParser.apply_to_function(not_int)


@pytest.fixture
def remap_setup():
    X = Dimension("X")
    Y = Dimension("Y")
    Y2XDim = Dimension("Y2X", kind=DimensionKind.LOCAL)
    Y2X = FieldOffset("Y2X", source=X, target=(Y, Y2XDim))
    return X, Y, Y2XDim, Y2X


def test_remap(remap_setup):
    X, Y, Y2XDim, Y2X = remap_setup

    def remap_fo(bar: Field[[X], int64]) -> Field[[Y], int64]:
        return bar(Y2X[0])

    parsed = FieldOperatorParser.apply_to_function(remap_fo)

    assert parsed.body.stmts[0].value.type == ct.FieldType(
        dims=[Y], dtype=ct.ScalarType(kind=ct.ScalarKind.INT64)
    )


def test_remap_nbfield(remap_setup):
    X, Y, Y2XDim, Y2X = remap_setup

    def remap_fo(bar: Field[[X], int64]) -> Field[[Y, Y2XDim], int64]:
        return bar(Y2X)

    parsed = FieldOperatorParser.apply_to_function(remap_fo)

    assert parsed.body.stmts[0].value.type == ct.FieldType(
        dims=[Y, Y2XDim], dtype=ct.ScalarType(kind=ct.ScalarKind.INT64)
    )


def test_remap_reduce(remap_setup):
    X, Y, Y2XDim, Y2X = remap_setup

    def remap_fo(bar: Field[[X], int64]) -> Field[[Y], int64]:
        return 2 * neighbor_sum(bar(Y2X), axis=Y2XDim)

    parsed = FieldOperatorParser.apply_to_function(remap_fo)

    assert parsed.body.stmts[0].value.type == ct.FieldType(
        dims=[Y], dtype=ct.ScalarType(kind=ct.ScalarKind.INT64)
    )


def test_remap_reduce_sparse(remap_setup):
    X, Y, Y2XDim, Y2X = remap_setup

    def remap_fo(bar: Field[[Y, Y2XDim], int64]) -> Field[[Y], int64]:
        return 5 * neighbor_sum(bar, axis=Y2XDim)

    parsed = FieldOperatorParser.apply_to_function(remap_fo)

    assert parsed.body.stmts[0].value.type == ct.FieldType(
        dims=[Y], dtype=ct.ScalarType(kind=ct.ScalarKind.INT64)
    )


def test_mismatched_literals():
    def mismatched_lit() -> Field[..., "float32"]:
        return float32("1.0") + float64("1.0")

    with pytest.raises(
        FieldOperatorTypeDeductionError,
        match=(r"Could not promote `float32` and `float64` to common type in call to +."),
    ):
        _ = FieldOperatorParser.apply_to_function(mismatched_lit)


def test_broadcast_multi_dim():
    ADim = Dimension("ADim")
    BDim = Dimension("BDim")
    CDim = Dimension("CDim")

    def simple_broadcast(a: Field[[ADim], float64]):
        return broadcast(a, (ADim, BDim, CDim))

    parsed = FieldOperatorParser.apply_to_function(simple_broadcast)

    assert parsed.body.stmts[0].value.type == ct.FieldType(
        dims=[ADim, BDim, CDim], dtype=ct.ScalarType(kind=ct.ScalarKind.FLOAT64)
    )


def test_broadcast_disjoint():
    ADim = Dimension("ADim")
    BDim = Dimension("BDim")
    CDim = Dimension("CDim")

    def disjoint_broadcast(a: Field[[ADim], float64]):
        return broadcast(a, (BDim, CDim))

    with pytest.raises(
        FieldOperatorTypeDeductionError,
        match=r"Expected broadcast dimension is missing",
    ):
        _ = FieldOperatorParser.apply_to_function(disjoint_broadcast)


def test_broadcast_badtype():
    ADim = Dimension("ADim")
    BDim = "BDim"
    CDim = Dimension("CDim")

    def badtype_broadcast(a: Field[[ADim], float64]):
        return broadcast(a, (BDim, CDim))

    with pytest.raises(
        FieldOperatorTypeDeductionError,
        match=r"Expected all broadcast dimensions to be of type Dimension.",
    ):
        _ = FieldOperatorParser.apply_to_function(badtype_broadcast)


def test_where_dim():
    ADim = Dimension("ADim")
    BDim = Dimension("BDim")

    def simple_where(a: Field[[ADim], bool], b: Field[[ADim, BDim], float64]):
        return where(a, b, 9.0)

    parsed = FieldOperatorParser.apply_to_function(simple_where)

    assert parsed.body.stmts[0].value.type == ct.FieldType(
        dims=[ADim, BDim], dtype=ct.ScalarType(kind=ct.ScalarKind.FLOAT64)
    )


def test_where_broadcast_dim():
    ADim = Dimension("ADim")

    def simple_where(a: Field[[ADim], bool]):
        return where(a, 5.0, 9.0)

    parsed = FieldOperatorParser.apply_to_function(simple_where)

    assert parsed.body.stmts[0].value.type == ct.FieldType(
        dims=[ADim], dtype=ct.ScalarType(kind=ct.ScalarKind.FLOAT64)
    )


def test_where_tuple_dim():
    ADim = Dimension("ADim")

    def tuple_where(a: Field[[ADim], bool], b: Field[[ADim], float64]):
        return where(a, ((5.0, 9.0), (b, 6.0)), ((8.0, b), (5.0, 9.0)))

    parsed = FieldOperatorParser.apply_to_function(tuple_where)

    assert parsed.body.stmts[0].value.type == ct.TupleType(
        types=[
            ct.TupleType(
                types=[
                    ct.FieldType(dims=[ADim], dtype=ct.ScalarType(kind=ct.ScalarKind.FLOAT64)),
                    ct.FieldType(dims=[ADim], dtype=ct.ScalarType(kind=ct.ScalarKind.FLOAT64)),
                ]
            ),
            ct.TupleType(
                types=[
                    ct.FieldType(dims=[ADim], dtype=ct.ScalarType(kind=ct.ScalarKind.FLOAT64)),
                    ct.FieldType(dims=[ADim], dtype=ct.ScalarType(kind=ct.ScalarKind.FLOAT64)),
                ]
            ),
        ]
    )


def test_where_bad_dim():
    ADim = Dimension("ADim")

    def bad_dim_where(a: Field[[ADim], bool], b: Field[[ADim], float64]):
        return where(a, ((5.0, 9.0), (b, 6.0)), b)

    with pytest.raises(
        FieldOperatorTypeDeductionError,
        match=r"Return arguments need to be of same type",
    ):
        _ = FieldOperatorParser.apply_to_function(bad_dim_where)


def test_where_mixed_dims():
    ADim = Dimension("ADim")
    BDim = Dimension("BDim")

    def tuple_where_mix_dims(
        a: Field[[ADim], bool], b: Field[[ADim], float64], c: Field[[ADim, BDim], float64]
    ):
        return where(a, ((c, 9.0), (b, 6.0)), ((8.0, b), (5.0, 9.0)))

    parsed = FieldOperatorParser.apply_to_function(tuple_where_mix_dims)

    assert parsed.body.stmts[0].value.type == ct.TupleType(
        types=[
            ct.TupleType(
                types=[
                    ct.FieldType(
                        dims=[ADim, BDim], dtype=ct.ScalarType(kind=ct.ScalarKind.FLOAT64)
                    ),
                    ct.FieldType(dims=[ADim], dtype=ct.ScalarType(kind=ct.ScalarKind.FLOAT64)),
                ]
            ),
            ct.TupleType(
                types=[
                    ct.FieldType(dims=[ADim], dtype=ct.ScalarType(kind=ct.ScalarKind.FLOAT64)),
                    ct.FieldType(dims=[ADim], dtype=ct.ScalarType(kind=ct.ScalarKind.FLOAT64)),
                ]
            ),
        ]
    )
