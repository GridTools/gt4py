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

import gt4py.next.ffront.type_specifications
from gt4py.next import (
    Dimension,
    DimensionKind,
    Field,
    FieldOffset,
    astype,
    broadcast,
    common,
    errors,
    float32,
    float64,
    int32,
    int64,
    neighbor_sum,
    where,
)
from gt4py.next.ffront.ast_passes import single_static_assign as ssa
from gt4py.next.ffront.experimental import as_offset
from gt4py.next.ffront.func_to_foast import FieldOperatorParser
from gt4py.next.type_system import type_info, type_specifications as ts


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
    fieldop_type = gt4py.next.ffront.type_specifications.FieldOperatorType(
        definition=ts.FunctionType(
            pos_only_args=[field_type, float_type],
            pos_or_kw_args={},
            kw_only_args={},
            returns=field_type,
        )
    )
    scanop_type = gt4py.next.ffront.type_specifications.ScanOperatorType(
        axis=KDim,
        definition=ts.FunctionType(
            pos_only_args=[],
            pos_or_kw_args={"carry": float_type, "a": int_type, "b": int_type},
            kw_only_args={},
            returns=float_type,
        ),
    )
    tuple_scanop_type = gt4py.next.ffront.type_specifications.ScanOperatorType(
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
            [
                r"Dimensions can not be promoted. Could not determine order of the "
                r"following dimensions: J, K."
            ],
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


def test_unpack_assign():
    def unpack_explicit_tuple(
        a: Field[[TDim], float64], b: Field[[TDim], float64]
    ) -> tuple[Field[[TDim], float64], Field[[TDim], float64]]:
        tmp_a, tmp_b = (a, b)
        return tmp_a, tmp_b

    parsed = FieldOperatorParser.apply_to_function(unpack_explicit_tuple)

    assert parsed.body.annex.symtable[ssa.unique_name("tmp_a", 0)].type == ts.FieldType(
        dims=[TDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64, shape=None)
    )
    assert parsed.body.annex.symtable[ssa.unique_name("tmp_b", 0)].type == ts.FieldType(
        dims=[TDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64, shape=None)
    )


def test_assign_tuple():
    def temp_tuple(a: Field[[TDim], float64], b: Field[[TDim], int64]):
        tmp = a, b
        return tmp

    parsed = FieldOperatorParser.apply_to_function(temp_tuple)

    assert parsed.body.annex.symtable[ssa.unique_name("tmp", 0)].type == ts.TupleType(
        types=[
            ts.FieldType(dims=[TDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64, shape=None)),
            ts.FieldType(dims=[TDim], dtype=ts.ScalarType(kind=ts.ScalarKind.INT64, shape=None)),
        ]
    )


def test_adding_bool():
    """Expect an error when using arithmetic on bools."""

    def add_bools(a: Field[[TDim], bool], b: Field[[TDim], bool]):
        return a + b

    with pytest.raises(
        errors.DSLError, match=(r"Type 'Field\[\[TDim\], bool\]' can not be used in operator '\+'.")
    ):
        _ = FieldOperatorParser.apply_to_function(add_bools)


def test_binop_nonmatching_dims():
    """Binary operations can only work when both fields have the same dimensions."""
    X = Dimension("X")
    Y = Dimension("Y")

    def nonmatching(a: Field[[X], float64], b: Field[[Y], float64]):
        return a + b

    with pytest.raises(
        errors.DSLError,
        match=(
            r"Could not promote 'Field\[\[X], float64\]' and 'Field\[\[Y\], float64\]' to common type in call to +."
        ),
    ):
        _ = FieldOperatorParser.apply_to_function(nonmatching)


def test_bitopping_float():
    def float_bitop(a: Field[[TDim], float], b: Field[[TDim], float]):
        return a & b

    with pytest.raises(
        errors.DSLError,
        match=(r"Type 'Field\[\[TDim\], float64\]' can not be used in operator '\&'."),
    ):
        _ = FieldOperatorParser.apply_to_function(float_bitop)


def test_signing_bool():
    def sign_bool(a: Field[[TDim], bool]):
        return -a

    with pytest.raises(
        errors.DSLError,
        match=r"Incompatible type for unary operator '\-': 'Field\[\[TDim\], bool\]'.",
    ):
        _ = FieldOperatorParser.apply_to_function(sign_bool)


def test_notting_int():
    def not_int(a: Field[[TDim], int64]):
        return not a

    with pytest.raises(
        errors.DSLError,
        match=r"Incompatible type for unary operator 'not': 'Field\[\[TDim\], int64\]'.",
    ):
        _ = FieldOperatorParser.apply_to_function(not_int)


@pytest.fixture
def premap_setup():
    X = Dimension("X")
    Y = Dimension("Y")
    Y2XDim = Dimension("Y2X", kind=DimensionKind.LOCAL)
    Y2X = FieldOffset("Y2X", source=X, target=(Y, Y2XDim))
    return X, Y, Y2XDim, Y2X


def test_premap(premap_setup):
    X, Y, Y2XDim, Y2X = premap_setup

    def premap_fo(bar: Field[[X], int64]) -> Field[[Y], int64]:
        return bar(Y2X[0])

    parsed = FieldOperatorParser.apply_to_function(premap_fo)

    assert parsed.body.stmts[0].value.type == ts.FieldType(
        dims=[Y], dtype=ts.ScalarType(kind=ts.ScalarKind.INT64)
    )


def test_premap_nbfield(premap_setup):
    X, Y, Y2XDim, Y2X = premap_setup

    def premap_fo(bar: Field[[X], int64]) -> Field[[Y, Y2XDim], int64]:
        return bar(Y2X)

    parsed = FieldOperatorParser.apply_to_function(premap_fo)

    assert parsed.body.stmts[0].value.type == ts.FieldType(
        dims=[Y, Y2XDim], dtype=ts.ScalarType(kind=ts.ScalarKind.INT64)
    )


def test_premap_reduce(premap_setup):
    X, Y, Y2XDim, Y2X = premap_setup

    def premap_fo(bar: Field[[X], int32]) -> Field[[Y], int32]:
        return 2 * neighbor_sum(bar(Y2X), axis=Y2XDim)

    parsed = FieldOperatorParser.apply_to_function(premap_fo)

    assert parsed.body.stmts[0].value.type == ts.FieldType(
        dims=[Y], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32)
    )


def test_premap_reduce_sparse(premap_setup):
    X, Y, Y2XDim, Y2X = premap_setup

    def premap_fo(bar: Field[[Y, Y2XDim], int32]) -> Field[[Y], int32]:
        return 5 * neighbor_sum(bar, axis=Y2XDim)

    parsed = FieldOperatorParser.apply_to_function(premap_fo)

    assert parsed.body.stmts[0].value.type == ts.FieldType(
        dims=[Y], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32)
    )


def test_mismatched_literals():
    def mismatched_lit() -> Field[[TDim], "float32"]:
        return float32("1.0") + float64("1.0")

    with pytest.raises(
        errors.DSLError,
        match=(r"Could not promote 'float32' and 'float64' to common type in call to +."),
    ):
        _ = FieldOperatorParser.apply_to_function(mismatched_lit)


def test_broadcast_multi_dim():
    ADim = Dimension("ADim")
    BDim = Dimension("BDim")
    CDim = Dimension("CDim")

    def simple_broadcast(a: Field[[ADim], float64]):
        return broadcast(a, (ADim, BDim, CDim))

    parsed = FieldOperatorParser.apply_to_function(simple_broadcast)

    assert parsed.body.stmts[0].value.type == ts.FieldType(
        dims=[ADim, BDim, CDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    )


def test_broadcast_disjoint():
    ADim = Dimension("ADim")
    BDim = Dimension("BDim")
    CDim = Dimension("CDim")

    def disjoint_broadcast(a: Field[[ADim], float64]):
        return broadcast(a, (BDim, CDim))

    with pytest.raises(errors.DSLError, match=r"expected broadcast dimension\(s\) \'.*\' missing"):
        _ = FieldOperatorParser.apply_to_function(disjoint_broadcast)


def test_broadcast_badtype():
    ADim = Dimension("ADim")
    BDim = "BDim"
    CDim = Dimension("CDim")

    def badtype_broadcast(a: Field[[ADim], float64]):
        return broadcast(a, (BDim, CDim))

    with pytest.raises(
        errors.DSLError, match=r"expected all broadcast dimensions to be of type 'Dimension'."
    ):
        _ = FieldOperatorParser.apply_to_function(badtype_broadcast)


def test_where_dim():
    ADim = Dimension("ADim")
    BDim = Dimension("BDim")

    def simple_where(a: Field[[ADim], bool], b: Field[[ADim, BDim], float64]):
        return where(a, b, 9.0)

    parsed = FieldOperatorParser.apply_to_function(simple_where)

    assert parsed.body.stmts[0].value.type == ts.FieldType(
        dims=[ADim, BDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    )


def test_where_broadcast_dim():
    ADim = Dimension("ADim")

    def simple_where(a: Field[[ADim], bool]):
        return where(a, 5.0, 9.0)

    parsed = FieldOperatorParser.apply_to_function(simple_where)

    assert parsed.body.stmts[0].value.type == ts.FieldType(
        dims=[ADim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    )


def test_where_tuple_dim():
    ADim = Dimension("ADim")

    def tuple_where(a: Field[[ADim], bool], b: Field[[ADim], float64]):
        return where(a, ((5.0, 9.0), (b, 6.0)), ((8.0, b), (5.0, 9.0)))

    parsed = FieldOperatorParser.apply_to_function(tuple_where)

    assert parsed.body.stmts[0].value.type == ts.TupleType(
        types=[
            ts.TupleType(
                types=[
                    ts.FieldType(dims=[ADim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)),
                    ts.FieldType(dims=[ADim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)),
                ]
            ),
            ts.TupleType(
                types=[
                    ts.FieldType(dims=[ADim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)),
                    ts.FieldType(dims=[ADim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)),
                ]
            ),
        ]
    )


def test_where_bad_dim():
    ADim = Dimension("ADim")

    def bad_dim_where(a: Field[[ADim], bool], b: Field[[ADim], float64]):
        return where(a, ((5.0, 9.0), (b, 6.0)), b)

    with pytest.raises(errors.DSLError, match=r"Return arguments need to be of same type"):
        _ = FieldOperatorParser.apply_to_function(bad_dim_where)


def test_where_mixed_dims():
    ADim = Dimension("ADim")
    BDim = Dimension("BDim")

    def tuple_where_mix_dims(
        a: Field[[ADim], bool], b: Field[[ADim], float64], c: Field[[ADim, BDim], float64]
    ):
        return where(a, ((c, 9.0), (b, 6.0)), ((8.0, b), (5.0, 9.0)))

    parsed = FieldOperatorParser.apply_to_function(tuple_where_mix_dims)

    assert parsed.body.stmts[0].value.type == ts.TupleType(
        types=[
            ts.TupleType(
                types=[
                    ts.FieldType(
                        dims=[ADim, BDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
                    ),
                    ts.FieldType(dims=[ADim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)),
                ]
            ),
            ts.TupleType(
                types=[
                    ts.FieldType(dims=[ADim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)),
                    ts.FieldType(dims=[ADim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)),
                ]
            ),
        ]
    )


def test_astype_dtype():
    def simple_astype(a: Field[[TDim], float64]):
        return astype(a, bool)

    parsed = FieldOperatorParser.apply_to_function(simple_astype)

    assert parsed.body.stmts[0].value.type == ts.FieldType(
        dims=[TDim], dtype=ts.ScalarType(kind=ts.ScalarKind.BOOL)
    )


def test_astype_wrong_dtype():
    def simple_astype(a: Field[[TDim], float64]):
        # we just use broadcast here, but anything with type function is fine
        return astype(a, broadcast)

    with pytest.raises(
        errors.DSLError,
        match=r"Invalid call to 'astype': second argument must be a scalar type, got.",
    ):
        _ = FieldOperatorParser.apply_to_function(simple_astype)


def test_astype_wrong_value_type():
    def simple_astype(a: Field[[TDim], float64]):
        # we just use broadcast here but anything that is not a field, scalar or tuple thereof works
        return astype(broadcast, bool)

    with pytest.raises(errors.DSLError) as exc_info:
        _ = FieldOperatorParser.apply_to_function(simple_astype)

    assert (
        re.search("Expected 1st argument to be of type", exc_info.value.__cause__.args[0])
        is not None
    )


def test_mod_floats():
    def modulo_floats(inp: Field[[TDim], float]):
        return inp % 3.0

    with pytest.raises(errors.DSLError, match=r"Type 'float64' can not be used in operator '%'"):
        _ = FieldOperatorParser.apply_to_function(modulo_floats)


def test_undefined_symbols():
    def return_undefined():
        return undefined_symbol

    with pytest.raises(errors.DSLError, match="Undeclared symbol"):
        _ = FieldOperatorParser.apply_to_function(return_undefined)


def test_as_offset_dim():
    ADim = Dimension("ADim")
    BDim = Dimension("BDim")
    Boff = FieldOffset("Boff", source=BDim, target=(BDim,))

    def as_offset_dim(a: Field[[ADim, BDim], float], b: Field[[ADim], int]):
        return a(as_offset(Boff, b))

    with pytest.raises(errors.DSLError, match=f"not in list of offset field dimensions"):
        _ = FieldOperatorParser.apply_to_function(as_offset_dim)


def test_as_offset_dtype():
    ADim = Dimension("ADim")
    BDim = Dimension("BDim")
    Boff = FieldOffset("Boff", source=BDim, target=(BDim,))

    def as_offset_dtype(a: Field[[ADim, BDim], float], b: Field[[BDim], float]):
        return a(as_offset(Boff, b))

    with pytest.raises(errors.DSLError, match=f"expected integer for offset field dtype"):
        _ = FieldOperatorParser.apply_to_function(as_offset_dtype)
