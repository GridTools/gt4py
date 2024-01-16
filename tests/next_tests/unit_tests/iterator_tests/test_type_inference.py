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

import numpy as np

import gt4py.next as gtx
from gt4py.next.iterator import ir, type_inference as ti
from gt4py.next.iterator.ir_utils import ir_makers as im


def test_unsatisfiable_constraints():
    a = ir.Sym(id="a", dtype=("float32", False))
    b = ir.Sym(id="b", dtype=("int32", False))

    testee = im.lambda_(a, b)(im.plus("a", "b"))

    # The type inference uses a set to store the constraints. Since the TypeVar indices use a
    # global counter the constraint resolution order depends on previous runs of the inference.
    # To avoid false positives we just ignore which way the constraints have been resolved.
    # (The previous description has never been verified.)
    expected_error = [
        (
            "Type inference failed: Can not satisfy constraints:\n"
            "  Primitive(name='int32') ≡ Primitive(name='float32')"
        ),
        (
            "Type inference failed: Can not satisfy constraints:\n"
            "  Primitive(name='float32') ≡ Primitive(name='int32')"
        ),
    ]

    try:
        inferred = ti.infer(testee)
    except ti.UnsatisfiableConstraintsError as e:
        assert str(e) in expected_error


def test_unsatisfiable_constraints():
    a = ir.Sym(id="a", dtype=("float32", False))
    b = ir.Sym(id="b", dtype=("int32", False))

    testee = im.lambda_(a, b)(im.plus("a", "b"))

    # TODO(tehrengruber): For whatever reason the ordering in the error message is not
    #  deterministic. Ignoring for now, as we want to refactor the type inference anyway.
    expected_error = [
        (
            "Type inference failed: Can not satisfy constraints:\n"
            "  Primitive(name='int32') ≡ Primitive(name='float32')"
        ),
        (
            "Type inference failed: Can not satisfy constraints:\n"
            "  Primitive(name='float32') ≡ Primitive(name='int32')"
        ),
    ]

    try:
        inferred = ti.infer(testee)
    except ti.UnsatisfiableConstraintsError as e:
        assert str(e) in expected_error


def test_sym_ref():
    testee = ir.SymRef(id="x")
    expected = ti.TypeVar(idx=0)
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "T₀"


def test_bool_literal():
    testee = ir.Literal(value="False", type="bool")
    expected = ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="bool"), size=ti.TypeVar(idx=0))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "bool⁰"


def test_int_literal():
    testee = ir.Literal(value="3", type="int32")
    expected = ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int32"), size=ti.TypeVar(idx=0))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "int32⁰"


def test_float_literal():
    testee = ir.Literal(value="3.0", type="float64")
    expected = ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="float64"), size=ti.TypeVar(idx=0))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "float64⁰"


def test_deref():
    testee = ir.SymRef(id="deref")
    expected = ti.FunctionType(
        args=ti.Tuple.from_elems(
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=0),
                size=ti.TypeVar(idx=1),
                current_loc=ti.TypeVar(idx=2),
                defined_loc=ti.TypeVar(idx=2),
            ),
        ),
        ret=ti.Val(kind=ti.Value(), dtype=ti.TypeVar(idx=0), size=ti.TypeVar(idx=1)),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(It[T₂, T₂, T₀¹]) → T₀¹"


def test_deref_call():
    testee = ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="x")])
    expected = ti.Val(kind=ti.Value(), dtype=ti.TypeVar(idx=0), size=ti.TypeVar(idx=1))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "T₀¹"


def test_lambda():
    testee = ir.Lambda(params=[ir.Sym(id="x")], expr=ir.SymRef(id="x"))
    expected = ti.FunctionType(
        args=ti.Tuple.from_elems(
            ti.TypeVar(idx=0),
        ),
        ret=ti.TypeVar(idx=0),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(T₀) → T₀"


def test_typed_lambda():
    testee = ir.Lambda(
        params=[ir.Sym(id="x", kind="Iterator", dtype=("float64", False))], expr=ir.SymRef(id="x")
    )
    expected_val = ti.Val(
        kind=ti.Iterator(),
        dtype=ti.Primitive(name="float64"),
        size=ti.TypeVar(idx=0),
        current_loc=ti.TypeVar(idx=1),
        defined_loc=ti.TypeVar(idx=2),
    )
    expected = ti.FunctionType(
        args=ti.Tuple.from_elems(expected_val),
        ret=expected_val,
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(It[T₁, T₂, float64⁰]) → It[T₁, T₂, float64⁰]"


def test_plus():
    testee = ir.SymRef(id="plus")
    t = ti.Val(kind=ti.Value(), dtype=ti.TypeVar(idx=0), size=ti.TypeVar(idx=1))
    expected = ti.FunctionType(args=ti.Tuple.from_elems(t, t), ret=t)
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(T₀¹, T₀¹) → T₀¹"


def test_power():
    testee = im.call("power")(im.literal_from_value(1.0), im.literal_from_value(2))
    expected = ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="float64"), size=ti.TypeVar(idx=0))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "float64⁰"


def test_eq():
    testee = ir.SymRef(id="eq")
    t = ti.Val(kind=ti.Value(), dtype=ti.TypeVar(idx=0), size=ti.TypeVar(idx=1))
    expected = ti.FunctionType(
        args=ti.Tuple.from_elems(t, t),
        ret=ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="bool"), size=ti.TypeVar(idx=1)),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(T₀¹, T₀¹) → bool¹"


def test_if():
    testee = ir.SymRef(id="if_")
    c = ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="bool"), size=ti.TypeVar(idx=0))
    t = ti.TypeVar(idx=1)
    expected = ti.FunctionType(args=ti.Tuple.from_elems(c, t, t), ret=t)
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(bool⁰, T₁, T₁) → T₁"


def test_if_call():
    testee = im.if_("cond", im.literal("1", "int32"), im.literal("1", "int32"))
    expected = ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int32"), size=ti.TypeVar(idx=0))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "int32⁰"


def test_not():
    testee = ir.SymRef(id="not_")
    t = ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="bool"), size=ti.TypeVar(idx=0))
    expected = ti.FunctionType(
        args=ti.Tuple.from_elems(
            t,
        ),
        ret=t,
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(bool⁰) → bool⁰"


def test_and():
    testee = ir.SymRef(id="and_")
    t = ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="bool"), size=ti.TypeVar(idx=0))
    expected = ti.FunctionType(args=ti.Tuple.from_elems(t, t), ret=t)
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(bool⁰, bool⁰) → bool⁰"


def test_cast():
    testee = ir.FunCall(
        fun=ir.SymRef(id="cast_"),
        args=[ir.Literal(value="1.", type="float64"), ir.SymRef(id="int64")],
    )
    expected = ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int64"), size=ti.TypeVar(idx=0))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "int64⁰"


def test_lift():
    testee = ir.SymRef(id="lift")
    expected = ti.FunctionType(
        args=ti.Tuple.from_elems(
            ti.FunctionType(
                args=ti.ValTuple(
                    kind=ti.Iterator(),
                    dtypes=ti.TypeVar(idx=0),
                    size=ti.TypeVar(idx=1),
                    current_loc=ti.TypeVar(idx=2),
                    defined_locs=ti.TypeVar(idx=3),
                ),
                ret=ti.Val(kind=ti.Value(), dtype=ti.TypeVar(idx=4), size=ti.TypeVar(idx=1)),
            ),
        ),
        ret=ti.FunctionType(
            args=ti.ValTuple(
                kind=ti.Iterator(),
                dtypes=ti.TypeVar(idx=0),
                size=ti.TypeVar(idx=1),
                current_loc=ti.TypeVar(idx=5),
                defined_locs=ti.TypeVar(idx=3),
            ),
            ret=ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=4),
                size=ti.TypeVar(idx=1),
                current_loc=ti.TypeVar(idx=5),
                defined_loc=ti.TypeVar(idx=2),
            ),
        ),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert (
        ti.pformat(inferred)
        == "((It[T₂, …₃, T¹], …)₀ → T₄¹) → (It[T₅, …₃, T¹], …)₀ → It[T₅, T₂, T₄¹]"
    )


def test_lift_lambda_without_args():
    testee = ir.FunCall(
        fun=ir.SymRef(id="lift"), args=[ir.Lambda(params=[], expr=ir.SymRef(id="x"))]
    )
    expected = ti.FunctionType(
        args=ti.ValTuple(
            kind=ti.Iterator(),
            dtypes=ti.EmptyTuple(),
            size=ti.TypeVar(idx=0),
            current_loc=ti.TypeVar(idx=1),
            defined_locs=ti.EmptyTuple(),
        ),
        ret=ti.Val(
            kind=ti.Iterator(),
            dtype=ti.TypeVar(idx=2),
            size=ti.TypeVar(idx=0),
            current_loc=ti.TypeVar(idx=1),
            defined_loc=ti.TypeVar(idx=3),
        ),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "() → It[T₁, T₃, T₂⁰]"


def test_lift_application():
    testee = ir.FunCall(fun=ir.SymRef(id="lift"), args=[ir.SymRef(id="deref")])
    expected = ti.FunctionType(
        args=ti.Tuple.from_elems(
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=0),
                size=ti.TypeVar(idx=1),
                current_loc=ti.TypeVar(idx=2),
                defined_loc=ti.TypeVar(idx=3),
            ),
        ),
        ret=ti.Val(
            kind=ti.Iterator(),
            dtype=ti.TypeVar(idx=0),
            size=ti.TypeVar(idx=1),
            current_loc=ti.TypeVar(idx=2),
            defined_loc=ti.TypeVar(idx=3),
        ),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(It[T₂, T₃, T₀¹]) → It[T₂, T₃, T₀¹]"


def test_lifted_call():
    testee = ir.FunCall(
        fun=ir.FunCall(fun=ir.SymRef(id="lift"), args=[ir.SymRef(id="deref")]),
        args=[ir.SymRef(id="x")],
    )
    expected = ti.Val(
        kind=ti.Iterator(),
        dtype=ti.TypeVar(idx=0),
        size=ti.TypeVar(idx=1),
        current_loc=ti.TypeVar(idx=2),
        defined_loc=ti.TypeVar(idx=3),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "It[T₂, T₃, T₀¹]"


def test_make_tuple():
    testee = ir.FunCall(
        fun=ir.SymRef(id="make_tuple"),
        args=[
            ir.Literal(value="True", type="bool"),
            ir.Literal(value="42.0", type="float64"),
            ir.SymRef(id="x"),
        ],
    )
    expected = ti.Val(
        kind=ti.Value(),
        dtype=ti.Tuple.from_elems(
            ti.Primitive(name="bool"), ti.Primitive(name="float64"), ti.TypeVar(idx=0)
        ),
        size=ti.TypeVar(idx=1),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(bool, float64, T₀)¹"


def test_tuple_get():
    testee = ir.FunCall(
        fun=ir.SymRef(id="tuple_get"),
        args=[
            ir.Literal(value="1", type=ir.INTEGER_INDEX_BUILTIN),
            ir.FunCall(
                fun=ir.SymRef(id="make_tuple"),
                args=[
                    ir.Literal(value="True", type="bool"),
                    ir.Literal(value="42.0", type="float64"),
                ],
            ),
        ],
    )
    expected = ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="float64"), size=ti.TypeVar(idx=0))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "float64⁰"


def test_tuple_get_in_lambda():
    testee = ir.Lambda(
        params=[ir.Sym(id="x")],
        expr=ir.FunCall(
            fun=ir.SymRef(id="tuple_get"),
            args=[ir.Literal(value="1", type=ir.INTEGER_INDEX_BUILTIN), ir.SymRef(id="x")],
        ),
    )
    expected = ti.FunctionType(
        args=ti.Tuple.from_elems(
            ti.Val(
                kind=ti.TypeVar(idx=0),
                dtype=ti.Tuple(
                    front=ti.TypeVar(idx=1),
                    others=ti.Tuple(front=ti.TypeVar(idx=2), others=ti.TypeVar(idx=3)),
                ),
                size=ti.TypeVar(idx=4),
            ),
        ),
        ret=ti.Val(kind=ti.TypeVar(idx=0), dtype=ti.TypeVar(idx=2), size=ti.TypeVar(idx=4)),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(ItOrVal₀[(T₁, T₂):T₃⁴]) → ItOrVal₀[T₂⁴]"


def test_neighbors():
    testee = ir.FunCall(
        fun=ir.SymRef(id="neighbors"), args=[ir.OffsetLiteral(value="V2E"), ir.SymRef(id="it")]
    )
    expected = ti.Val(
        kind=ti.Value(),
        dtype=ti.List(
            dtype=ti.TypeVar(idx=0), max_length=ti.TypeVar(idx=1), has_skip_values=ti.TypeVar(idx=2)
        ),
        size=ti.TypeVar(idx=3),
    )
    inferred = ti.infer(testee)
    assert expected == inferred
    assert ti.pformat(inferred) == "L[T₀, T₁, T₂]³"


def test_reduce():
    reduction_f = ir.Lambda(
        params=[ir.Sym(id="acc"), ir.Sym(id="x"), ir.Sym(id="y")],
        expr=ir.FunCall(
            fun=ir.SymRef(id="plus"),
            args=[
                ir.SymRef(id="acc"),
                ir.FunCall(
                    fun=ir.SymRef(id="cast_"),  # cast to the type of `init`
                    args=[
                        ir.FunCall(
                            fun=ir.SymRef(id="multiplies"),
                            args=[
                                ir.SymRef(id="x"),
                                ir.FunCall(
                                    fun=ir.SymRef(
                                        id="cast_"
                                    ),  # force `x` to be of type `float64` -> `y` is unconstrained
                                    args=[ir.SymRef(id="y"), ir.SymRef(id="float64")],
                                ),
                            ],
                        ),
                        ir.SymRef(id="int64"),
                    ],
                ),
            ],
        ),
    )
    testee = ir.FunCall(
        fun=ir.SymRef(id="reduce"), args=[reduction_f, ir.Literal(value="0", type="int64")]
    )
    expected = ti.FunctionType(
        args=ti.ValListTuple(
            kind=ti.Value(),
            list_dtypes=ti.Tuple.from_elems(ti.Primitive(name="float64"), ti.TypeVar(idx=0)),
            max_length=ti.TypeVar(idx=1),
            has_skip_values=ti.TypeVar(idx=2),
            size=ti.TypeVar(idx=3),
        ),
        ret=ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int64"), size=ti.TypeVar(idx=3)),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(L[float64, T₁, T₂]³, L[T₀, T₁, T₂]³) → int64³"


def test_scan():
    scan_f = ir.Lambda(
        params=[ir.Sym(id="acc"), ir.Sym(id="x"), ir.Sym(id="y")],
        expr=ir.FunCall(
            fun=ir.SymRef(id="plus"),
            args=[
                ir.SymRef(id="acc"),
                ir.FunCall(
                    fun=ir.SymRef(id="multiplies"),
                    args=[
                        ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="x")]),
                        ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="y")]),
                    ],
                ),
            ],
        ),
    )
    testee = ir.FunCall(
        fun=ir.SymRef(id="scan"),
        args=[scan_f, ir.Literal(value="True", type="bool"), ir.Literal(value="0", type="int64")],
    )
    expected = ti.FunctionType(
        args=ti.Tuple.from_elems(
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.Primitive(name="int64"),
                size=ti.Column(),
                current_loc=ti.TypeVar(idx=0),
                defined_loc=ti.TypeVar(idx=0),
            ),
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.Primitive(name="int64"),
                size=ti.Column(),
                current_loc=ti.TypeVar(idx=0),
                defined_loc=ti.TypeVar(idx=0),
            ),
        ),
        ret=ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int64"), size=ti.Column()),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(It[T₀, T₀, int64ᶜ], It[T₀, T₀, int64ᶜ]) → int64ᶜ"


def test_shift():
    testee = ir.FunCall(
        fun=ir.SymRef(id="shift"), args=[ir.OffsetLiteral(value="i"), ir.OffsetLiteral(value=1)]
    )
    expected = ti.FunctionType(
        args=ti.Tuple.from_elems(
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=0),
                size=ti.TypeVar(idx=1),
                current_loc=ti.TypeVar(idx=2),
                defined_loc=ti.TypeVar(idx=3),
            ),
        ),
        ret=ti.Val(
            kind=ti.Iterator(),
            dtype=ti.TypeVar(idx=0),
            size=ti.TypeVar(idx=1),
            current_loc=ti.TypeVar(idx=4),
            defined_loc=ti.TypeVar(idx=3),
        ),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(It[T₂, T₃, T₀¹]) → It[T₄, T₃, T₀¹]"


def test_shift_with_cartesian_offset_provider():
    testee = ir.FunCall(
        fun=ir.SymRef(id="shift"), args=[ir.OffsetLiteral(value="i"), ir.OffsetLiteral(value=1)]
    )
    expected = ti.FunctionType(
        args=ti.Tuple.from_elems(
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=0),
                size=ti.TypeVar(idx=1),
                current_loc=ti.TypeVar(idx=2),
                defined_loc=ti.TypeVar(idx=3),
            ),
        ),
        ret=ti.Val(
            kind=ti.Iterator(),
            dtype=ti.TypeVar(idx=0),
            size=ti.TypeVar(idx=1),
            current_loc=ti.TypeVar(idx=2),
            defined_loc=ti.TypeVar(idx=3),
        ),
    )
    offset_provider = {"i": gtx.Dimension("IDim")}
    inferred = ti.infer(testee, offset_provider=offset_provider)
    assert inferred == expected
    assert ti.pformat(inferred) == "(It[T₂, T₃, T₀¹]) → It[T₂, T₃, T₀¹]"


def test_partial_shift_with_cartesian_offset_provider():
    testee = ir.FunCall(fun=ir.SymRef(id="shift"), args=[ir.OffsetLiteral(value="i")])
    expected = ti.FunctionType(
        args=ti.Tuple.from_elems(
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=0),
                size=ti.TypeVar(idx=1),
                current_loc=ti.TypeVar(idx=2),
                defined_loc=ti.TypeVar(idx=3),
            ),
        ),
        ret=ti.Val(
            kind=ti.Iterator(),
            dtype=ti.TypeVar(idx=0),
            size=ti.TypeVar(idx=1),
            current_loc=ti.TypeVar(idx=2),
            defined_loc=ti.TypeVar(idx=3),
        ),
    )
    offset_provider = {"i": gtx.Dimension("IDim")}
    inferred = ti.infer(testee, offset_provider=offset_provider)
    assert inferred == expected
    assert ti.pformat(inferred) == "(It[T₂, T₃, T₀¹]) → It[T₂, T₃, T₀¹]"


def test_shift_with_unstructured_offset_provider():
    testee = ir.FunCall(
        fun=ir.SymRef(id="shift"), args=[ir.OffsetLiteral(value="V2E"), ir.OffsetLiteral(value=0)]
    )
    expected = ti.FunctionType(
        args=ti.Tuple.from_elems(
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=0),
                size=ti.TypeVar(idx=1),
                current_loc=ti.Location(name="Vertex"),
                defined_loc=ti.TypeVar(idx=2),
            ),
        ),
        ret=ti.Val(
            kind=ti.Iterator(),
            dtype=ti.TypeVar(idx=0),
            size=ti.TypeVar(idx=1),
            current_loc=ti.Location(name="Edge"),
            defined_loc=ti.TypeVar(idx=2),
        ),
    )
    offset_provider = {
        "V2E": gtx.NeighborTableOffsetProvider(
            np.empty((0, 1), dtype=np.int64), gtx.Dimension("Vertex"), gtx.Dimension("Edge"), 1
        )
    }
    inferred = ti.infer(testee, offset_provider=offset_provider)
    assert inferred == expected
    assert ti.pformat(inferred) == "(It[Vertex, T₂, T₀¹]) → It[Edge, T₂, T₀¹]"


def test_partial_shift_with_unstructured_offset_provider():
    testee = ir.FunCall(
        fun=ir.SymRef(id="shift"),
        args=[
            ir.OffsetLiteral(value="V2E"),
            ir.OffsetLiteral(value=0),
            ir.OffsetLiteral(value="E2C"),
        ],
    )
    expected = ti.FunctionType(
        args=ti.Tuple.from_elems(
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=0),
                size=ti.TypeVar(idx=1),
                current_loc=ti.Location(name="Vertex"),
                defined_loc=ti.TypeVar(idx=2),
            ),
        ),
        ret=ti.Val(
            kind=ti.Iterator(),
            dtype=ti.TypeVar(idx=0),
            size=ti.TypeVar(idx=1),
            current_loc=ti.Location(name="Cell"),
            defined_loc=ti.TypeVar(idx=2),
        ),
    )
    offset_provider = {
        "V2E": gtx.NeighborTableOffsetProvider(
            np.empty((0, 1), dtype=np.int64), gtx.Dimension("Vertex"), gtx.Dimension("Edge"), 1
        ),
        "E2C": gtx.NeighborTableOffsetProvider(
            np.empty((0, 1), dtype=np.int64), gtx.Dimension("Edge"), gtx.Dimension("Cell"), 1
        ),
    }
    inferred = ti.infer(testee, offset_provider=offset_provider)
    assert inferred == expected
    assert ti.pformat(inferred) == "(It[Vertex, T₂, T₀¹]) → It[Cell, T₂, T₀¹]"


def test_function_definition():
    testee = ir.FunctionDefinition(id="f", params=[ir.Sym(id="x")], expr=ir.SymRef(id="x"))
    expected = ti.LetPolymorphic(
        dtype=ti.FunctionType(
            args=ti.Tuple.from_elems(
                ti.TypeVar(idx=0),
            ),
            ret=ti.TypeVar(idx=0),
        ),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred.dtype) == "(T₀) → T₀"


def test_dynamic_offset():
    """Test that the type of a dynamic offset is correctly inferred."""
    offset_it = ir.SymRef(id="offset_it")
    testee = ir.FunCall(
        fun=ir.SymRef(id="shift"),
        args=[
            ir.OffsetLiteral(value="V2E"),
            ir.FunCall(fun=ir.SymRef(id="deref"), args=[offset_it]),
        ],
    )
    inferred_all: dict[int, ti.Type] = ti.infer_all(testee)
    offset_it_type = inferred_all[id(offset_it)]
    assert isinstance(offset_it_type, ti.Val) and offset_it_type.kind == ti.Iterator()


CARTESIAN_DOMAIN = ir.FunCall(
    fun=ir.SymRef(id="cartesian_domain"),
    args=[
        ir.FunCall(
            fun=ir.SymRef(id="named_range"),
            args=[
                ir.AxisLiteral(value="IDim"),
                ir.Literal(value="0", type=ir.INTEGER_INDEX_BUILTIN),
                ir.SymRef(id="i"),
            ],
        ),
        ir.FunCall(
            fun=ir.SymRef(id="named_range"),
            args=[
                ir.AxisLiteral(value="JDim"),
                ir.Literal(value="0", type=ir.INTEGER_INDEX_BUILTIN),
                ir.SymRef(id="j"),
            ],
        ),
        ir.FunCall(
            fun=ir.SymRef(id="named_range"),
            args=[
                ir.AxisLiteral(value="KDim"),
                ir.Literal(value="0", type=ir.INTEGER_INDEX_BUILTIN),
                ir.SymRef(id="k"),
            ],
        ),
    ],
)


def test_stencil_closure():
    testee = ir.StencilClosure(
        domain=CARTESIAN_DOMAIN,
        stencil=ir.SymRef(id="deref"),
        output=ir.SymRef(id="out"),
        inputs=[ir.SymRef(id="inp")],
    )
    expected = ti.Closure(
        output=ti.Val(
            kind=ti.Iterator(),
            dtype=ti.TypeVar(idx=0),
            size=ti.Column(),
            current_loc=ti.ANYWHERE,
            defined_loc=ti.TypeVar(idx=1),
        ),
        inputs=ti.Tuple.from_elems(
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=0),
                size=ti.Column(),
                current_loc=ti.ANYWHERE,
                defined_loc=ti.TypeVar(idx=1),
            ),
        ),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(It[ANYWHERE, T₁, T₀ᶜ]) ⇒ It[ANYWHERE, T₁, T₀ᶜ]"


def test_fencil_definition():
    testee = ir.FencilDefinition(
        id="f",
        function_definitions=[],
        params=[
            ir.Sym(id="i"),
            ir.Sym(id="j"),
            ir.Sym(id="k"),
            ir.Sym(id="a"),
            ir.Sym(id="b"),
            ir.Sym(id="c"),
            ir.Sym(id="d"),
        ],
        closures=[
            ir.StencilClosure(
                domain=CARTESIAN_DOMAIN,
                stencil=ir.SymRef(id="deref"),
                output=ir.SymRef(id="b"),
                inputs=[ir.SymRef(id="a")],
            ),
            ir.StencilClosure(
                domain=CARTESIAN_DOMAIN,
                stencil=ir.SymRef(id="deref"),
                output=ir.SymRef(id="d"),
                inputs=[ir.SymRef(id="c")],
            ),
        ],
    )
    expected = ti.FencilDefinitionType(
        name="f",
        fundefs=ti.EmptyTuple(),
        params=ti.Tuple.from_elems(
            ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int32"), size=ti.Scalar()),
            ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int32"), size=ti.Scalar()),
            ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int32"), size=ti.Scalar()),
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=0),
                size=ti.Column(),
                current_loc=ti.ANYWHERE,
                defined_loc=ti.TypeVar(idx=1),
            ),
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=0),
                size=ti.Column(),
                current_loc=ti.ANYWHERE,
                defined_loc=ti.TypeVar(idx=1),
            ),
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=2),
                size=ti.Column(),
                current_loc=ti.ANYWHERE,
                defined_loc=ti.TypeVar(idx=3),
            ),
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=2),
                size=ti.Column(),
                current_loc=ti.ANYWHERE,
                defined_loc=ti.TypeVar(idx=3),
            ),
        ),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert (
        ti.pformat(inferred)
        == "{f(int32ˢ, int32ˢ, int32ˢ, It[ANYWHERE, T₁, T₀ᶜ], It[ANYWHERE, T₁, T₀ᶜ], It[ANYWHERE, T₃, T₂ᶜ], It[ANYWHERE, T₃, T₂ᶜ])}"
    )


def test_fencil_definition_same_closure_input():
    f1 = ir.FunctionDefinition(
        id="f1", params=[im.sym("vertex_it")], expr=im.deref(im.shift("E2V")("vertex_it"))
    )
    f2 = ir.FunctionDefinition(id="f2", params=[im.sym("vertex_it")], expr=im.deref("vertex_it"))

    testee = ir.FencilDefinition(
        id="fencil",
        function_definitions=[f1, f2],
        params=[im.sym("vertex_it"), im.sym("output_edge_it"), im.sym("output_vertex_it")],
        closures=[
            ir.StencilClosure(
                domain=im.call("unstructured_domain")(
                    im.call("named_range")(
                        ir.AxisLiteral(value="Edge"),
                        ir.Literal(value="0", type="int32"),
                        ir.Literal(value="10", type="int32"),
                    )
                ),
                stencil=im.ref("f1"),
                output=im.ref("output_edge_it"),
                inputs=[im.ref("vertex_it")],
            ),
            ir.StencilClosure(
                domain=im.call("unstructured_domain")(
                    im.call("named_range")(
                        ir.AxisLiteral(value="Vertex"),
                        ir.Literal(value="0", type="int32"),
                        ir.Literal(value="10", type="int32"),
                    )
                ),
                stencil=im.ref("f2"),
                output=im.ref("output_vertex_it"),
                inputs=[im.ref("vertex_it")],
            ),
        ],
    )

    offset_provider = {
        "E2V": gtx.NeighborTableOffsetProvider(
            np.empty((0, 2), dtype=np.int64),
            gtx.Dimension("Edge"),
            gtx.Dimension("Vertex"),
            2,
            False,
        )
    }
    inferred_all: dict[int, ti.Type] = ti.infer_all(testee, offset_provider=offset_provider)

    # validate locations of fencil params
    fencil_param_types = [inferred_all[id(testee.params[i])] for i in range(3)]
    assert fencil_param_types[0].defined_loc == ti.Location(name="Vertex")
    assert fencil_param_types[1].defined_loc == ti.Location(name="Edge")
    assert fencil_param_types[2].defined_loc == ti.Location(name="Vertex")

    # validate locations of stencil params
    f1_param_type: ti.Val = inferred_all[id(f1.params[0])]
    assert f1_param_type.current_loc == ti.Location(name="Edge")
    assert f1_param_type.defined_loc == ti.Location(name="Vertex")
    #  f2 is polymorphic and there is no shift inside so we only get a TypeVar here
    f2_param_type: ti.Val = inferred_all[id(f2.params[0])]
    assert isinstance(f2_param_type.current_loc, ti.TypeVar)
    assert isinstance(f2_param_type.defined_loc, ti.TypeVar)


def test_fencil_definition_with_function_definitions():
    fundefs = [
        ir.FunctionDefinition(id="f", params=[ir.Sym(id="x")], expr=ir.SymRef(id="x")),
        ir.FunctionDefinition(
            id="g",
            params=[ir.Sym(id="x")],
            expr=ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="x")]),
        ),
    ]
    testee = ir.FencilDefinition(
        id="foo",
        function_definitions=fundefs,
        params=[
            ir.Sym(id="i"),
            ir.Sym(id="j"),
            ir.Sym(id="k"),
            ir.Sym(id="a"),
            ir.Sym(id="b"),
            ir.Sym(id="c"),
            ir.Sym(id="d"),
            ir.Sym(id="x"),
            ir.Sym(id="y"),
        ],
        closures=[
            ir.StencilClosure(
                domain=CARTESIAN_DOMAIN,
                stencil=ir.SymRef(id="g"),
                output=ir.SymRef(id="b"),
                inputs=[ir.SymRef(id="a")],
            ),
            ir.StencilClosure(
                domain=CARTESIAN_DOMAIN,
                stencil=ir.SymRef(id="deref"),
                output=ir.SymRef(id="d"),
                inputs=[ir.SymRef(id="c")],
            ),
            ir.StencilClosure(
                domain=CARTESIAN_DOMAIN,
                stencil=ir.Lambda(
                    params=[ir.Sym(id="y")],
                    expr=ir.FunCall(
                        fun=ir.SymRef(id="g"),
                        args=[ir.FunCall(fun=ir.SymRef(id="f"), args=[ir.SymRef(id="y")])],
                    ),
                ),
                output=ir.SymRef(id="y"),
                inputs=[ir.SymRef(id="x")],
            ),
        ],
    )
    expected = ti.FencilDefinitionType(
        name="foo",
        fundefs=ti.Tuple.from_elems(
            ti.FunctionDefinitionType(
                name="f",
                fun=ti.FunctionType(
                    args=ti.Tuple.from_elems(
                        ti.TypeVar(idx=0),
                    ),
                    ret=ti.TypeVar(idx=0),
                ),
            ),
            ti.FunctionDefinitionType(
                name="g",
                fun=ti.FunctionType(
                    args=ti.Tuple.from_elems(
                        ti.Val(
                            kind=ti.Iterator(),
                            dtype=ti.TypeVar(idx=1),
                            size=ti.TypeVar(idx=2),
                            current_loc=ti.TypeVar(idx=3),
                            defined_loc=ti.TypeVar(idx=3),
                        ),
                    ),
                    ret=ti.Val(kind=ti.Value(), dtype=ti.TypeVar(idx=1), size=ti.TypeVar(idx=2)),
                ),
            ),
        ),
        params=ti.Tuple.from_elems(
            ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int32"), size=ti.Scalar()),
            ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int32"), size=ti.Scalar()),
            ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int32"), size=ti.Scalar()),
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=4),
                size=ti.Column(),
                current_loc=ti.ANYWHERE,
                defined_loc=ti.TypeVar(idx=5),
            ),
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=4),
                size=ti.Column(),
                current_loc=ti.ANYWHERE,
                defined_loc=ti.TypeVar(idx=5),
            ),
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=6),
                size=ti.Column(),
                current_loc=ti.ANYWHERE,
                defined_loc=ti.TypeVar(idx=7),
            ),
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=6),
                size=ti.Column(),
                current_loc=ti.ANYWHERE,
                defined_loc=ti.TypeVar(idx=7),
            ),
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=8),
                size=ti.Column(),
                current_loc=ti.ANYWHERE,
                defined_loc=ti.TypeVar(idx=9),
            ),
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=8),
                size=ti.Column(),
                current_loc=ti.ANYWHERE,
                defined_loc=ti.TypeVar(idx=9),
            ),
        ),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert (
        ti.pformat(inferred)
        == "{f :: (T₀) → T₀, g :: (It[T₃, T₃, T₁²]) → T₁², foo(int32ˢ, int32ˢ, int32ˢ, It[ANYWHERE, T₅, T₄ᶜ], It[ANYWHERE, T₅, T₄ᶜ], It[ANYWHERE, T₇, T₆ᶜ], It[ANYWHERE, T₇, T₆ᶜ], It[ANYWHERE, T₉, T₈ᶜ], It[ANYWHERE, T₉, T₈ᶜ])}"
    )


def test_save_types_to_annex():
    testee = im.lambda_("a")(im.plus("a", im.literal("1", "float32")))
    ti.infer(testee, save_to_annex=True)
    param_type = testee.params[0].annex.type
    assert isinstance(param_type, ti.Val) and param_type.dtype.name == "float32"


def test_pformat():
    vs = [ti.TypeVar(idx=i) for i in range(5)]
    assert ti.pformat(vs[0]) == "T₀"
    assert ti.pformat(ti.Tuple.from_elems(*vs[:2])) == "(T₀, T₁)"
    assert (
        ti.pformat(ti.Tuple(front=vs[0], others=ti.Tuple(front=vs[1], others=vs[2])))
        == "(T₀, T₁):T₂"
    )
    assert ti.pformat(ti.FunctionType(args=vs[0], ret=vs[1])) == "T₀ → T₁"
    assert ti.pformat(ti.Val(kind=vs[0], dtype=vs[1], size=vs[2])) == "ItOrVal₀[T₁²]"
    assert ti.pformat(ti.Val(kind=ti.Value(), dtype=vs[0], size=vs[1])) == "T₀¹"
    assert ti.pformat(ti.Val(kind=ti.Iterator(), dtype=vs[0], size=vs[1])) == "It[T₀¹]"
    assert (
        ti.pformat(
            ti.Val(
                kind=ti.Iterator(), dtype=vs[0], size=vs[1], current_loc=vs[2], defined_loc=vs[3]
            )
        )
        == "It[T₂, T₃, T₀¹]"
    )
    assert ti.pformat(ti.Val(kind=ti.Value(), dtype=vs[0], size=ti.Scalar())) == "T₀ˢ"
    assert ti.pformat(ti.Val(kind=ti.Value(), dtype=vs[0], size=ti.Column())) == "T₀ᶜ"
    assert ti.pformat(ti.ValTuple(kind=vs[0], dtypes=vs[1], size=vs[2])) == "(ItOrVal₀[T²], …)₁"
    assert (
        ti.pformat(
            ti.ValListTuple(
                list_dtypes=ti.Tuple.from_elems(vs[0], vs[1]),
                max_length=vs[2],
                has_skip_values=vs[3],
                size=vs[4],
            )
        )
        == "(L[T₀, T₂, T₃]⁴, L[T₁, T₂, T₃]⁴)"
    )
    assert (
        ti.pformat(
            ti.ValListTuple(list_dtypes=vs[0], max_length=vs[1], has_skip_values=vs[2], size=vs[3])
        )
        == "(L[…₀, T₁, T₂]³, …)"
    )
    assert ti.pformat(ti.Primitive(name="foo")) == "foo"
    assert ti.pformat(ti.Closure(output=vs[0], inputs=vs[1])) == "T₁ ⇒ T₀"
    assert (
        ti.pformat(ti.FunctionDefinitionType(name="f", fun=ti.FunctionType(args=vs[0], ret=vs[1])))
        == "f :: T₀ → T₁"
    )
    assert (
        ti.pformat(
            ti.FencilDefinitionType(name="f", fundefs=ti.EmptyTuple(), params=ti.EmptyTuple())
        )
        == "{f()}"
    )
