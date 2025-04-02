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
    """Dimension promotion is applied before Binary operations, i.e., they can also work on two fields that don't have the same dimensions."""
    X = Dimension("X")
    Y = Dimension("Y")

    def nonmatching(a: Field[[X], float64], b: Field[[Y], float64]):
        return a + b

    parsed = FieldOperatorParser.apply_to_function(nonmatching)

    assert parsed.body.stmts[0].value.type == ts.FieldType(
        dims=[X, Y], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    )


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


def test_premap_nbfield_with_vertical(premap_setup):
    X, Y, Y2XDim, Y2X = premap_setup
    K = Dimension("K", kind=DimensionKind.VERTICAL)

    def premap_fo(bar: Field[[X, K], int64]) -> Field[[Y, Y2XDim, K], int64]:
        return bar(Y2X)

    parsed = FieldOperatorParser.apply_to_function(premap_fo)

    assert parsed.body.stmts[0].value.type == ts.FieldType(
        dims=[Y, Y2XDim, K], dtype=ts.ScalarType(kind=ts.ScalarKind.INT64)
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
