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

import numpy as np

from gt4py.next.common import Dimension
from gt4py.next.iterator import ir, type_inference as ti
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider
from gt4py.next.iterator.runtime import CartesianAxis


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
    testee = ir.Literal(value="3", type="int")
    expected = ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int"), size=ti.TypeVar(idx=0))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "int⁰"


def test_float_literal():
    testee = ir.Literal(value="3.0", type="float")
    expected = ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="float"), size=ti.TypeVar(idx=0))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "float⁰"


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


def test_plus():
    testee = ir.SymRef(id="plus")
    t = ti.Val(kind=ti.Value(), dtype=ti.TypeVar(idx=0), size=ti.TypeVar(idx=1))
    expected = ti.FunctionType(args=ti.Tuple.from_elems(t, t), ret=t)
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(T₀¹, T₀¹) → T₀¹"


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
    t = ti.Val(kind=ti.Value(), dtype=ti.TypeVar(idx=1), size=ti.TypeVar(idx=0))
    expected = ti.FunctionType(args=ti.Tuple.from_elems(c, t, t), ret=t)
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(bool⁰, T₁⁰, T₁⁰) → T₁⁰"


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
            ir.Literal(value="42.0", type="float"),
            ir.SymRef(id="x"),
        ],
    )
    expected = ti.Val(
        kind=ti.Value(),
        dtype=ti.Tuple.from_elems(
            ti.Primitive(name="bool"), ti.Primitive(name="float"), ti.TypeVar(idx=0)
        ),
        size=ti.TypeVar(idx=1),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(bool, float, T₀)¹"


def test_tuple_get():
    testee = ir.FunCall(
        fun=ir.SymRef(id="tuple_get"),
        args=[
            ir.Literal(value="1", type="int"),
            ir.FunCall(
                fun=ir.SymRef(id="make_tuple"),
                args=[
                    ir.Literal(value="True", type="bool"),
                    ir.Literal(value="42.0", type="float"),
                ],
            ),
        ],
    )
    expected = ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="float"), size=ti.TypeVar(idx=0))
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "float⁰"


def test_tuple_get_in_lambda():
    testee = ir.Lambda(
        params=[ir.Sym(id="x")],
        expr=ir.FunCall(
            fun=ir.SymRef(id="tuple_get"),
            args=[ir.Literal(value="1", type="int"), ir.SymRef(id="x")],
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


def test_reduce():
    reduction_f = ir.Lambda(
        params=[ir.Sym(id="acc"), ir.Sym(id="x"), ir.Sym(id="y")],
        expr=ir.FunCall(
            fun=ir.SymRef(id="plus"),
            args=[
                ir.SymRef(id="acc"),
                ir.FunCall(
                    fun=ir.SymRef(id="multiplies"), args=[ir.SymRef(id="x"), ir.SymRef(id="y")]
                ),
            ],
        ),
    )
    testee = ir.FunCall(
        fun=ir.SymRef(id="reduce"), args=[reduction_f, ir.Literal(value="0", type="int")]
    )
    expected = ti.FunctionType(
        args=ti.ValTuple(
            kind=ti.Iterator(),
            dtypes=ti.Tuple.from_elems(ti.Primitive(name="int"), ti.Primitive(name="int")),
            size=ti.TypeVar(idx=0),
            current_loc=ti.TypeVar(idx=1),
            defined_locs=ti.TypeVar(idx=2),
        ),
        ret=ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int"), size=ti.TypeVar(idx=0)),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(It[T₁, _, int⁰], It[T₁, _, int⁰]) → int⁰"


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
        args=[scan_f, ir.Literal(value="True", type="bool"), ir.Literal(value="0", type="int")],
    )
    expected = ti.FunctionType(
        args=ti.Tuple.from_elems(
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.Primitive(name="int"),
                size=ti.Column(),
                current_loc=ti.TypeVar(idx=0),
                defined_loc=ti.TypeVar(idx=0),
            ),
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.Primitive(name="int"),
                size=ti.Column(),
                current_loc=ti.TypeVar(idx=0),
                defined_loc=ti.TypeVar(idx=0),
            ),
        ),
        ret=ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int"), size=ti.Column()),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(It[T₀, T₀, intᶜ], It[T₀, T₀, intᶜ]) → intᶜ"


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
    offset_provider = {"i": CartesianAxis("IDim")}
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
    offset_provider = {"i": CartesianAxis("IDim")}
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
        "V2E": NeighborTableOffsetProvider(
            np.empty((0, 1), dtype=np.int64), Dimension("Vertex"), Dimension("Edge"), 1
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
        "V2E": NeighborTableOffsetProvider(
            np.empty((0, 1), dtype=np.int64), Dimension("Vertex"), Dimension("Edge"), 1
        ),
        "E2C": NeighborTableOffsetProvider(
            np.empty((0, 1), dtype=np.int64), Dimension("Edge"), Dimension("Cell"), 1
        ),
    }
    inferred = ti.infer(testee, offset_provider=offset_provider)
    assert inferred == expected
    assert ti.pformat(inferred) == "(It[Vertex, T₂, T₀¹]) → It[Cell, T₂, T₀¹]"


def test_function_definition():
    testee = ir.FunctionDefinition(id="f", params=[ir.Sym(id="x")], expr=ir.SymRef(id="x"))
    expected = ti.FunctionDefinitionType(
        name="f",
        fun=ti.FunctionType(
            args=ti.Tuple.from_elems(
                ti.TypeVar(idx=0),
            ),
            ret=ti.TypeVar(idx=0),
        ),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "f :: (T₀) → T₀"


CARTESIAN_DOMAIN = ir.FunCall(
    fun=ir.SymRef(id="cartesian_domain"),
    args=[
        ir.FunCall(
            fun=ir.SymRef(id="named_range"),
            args=[
                ir.AxisLiteral(value="IDim"),
                ir.Literal(value="0", type="int"),
                ir.SymRef(id="i"),
            ],
        ),
        ir.FunCall(
            fun=ir.SymRef(id="named_range"),
            args=[
                ir.AxisLiteral(value="JDim"),
                ir.Literal(value="0", type="int"),
                ir.SymRef(id="j"),
            ],
        ),
        ir.FunCall(
            fun=ir.SymRef(id="named_range"),
            args=[
                ir.AxisLiteral(value="KDim"),
                ir.Literal(value="0", type="int"),
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
            current_loc=ti.TypeVar(idx=1),
            defined_loc=ti.TypeVar(idx=1),
        ),
        inputs=ti.Tuple.from_elems(
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=0),
                size=ti.Column(),
                current_loc=ti.TypeVar(idx=2),
                defined_loc=ti.TypeVar(idx=2),
            ),
        ),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert ti.pformat(inferred) == "(It[T₂, T₂, T₀ᶜ]) ⇒ It[T₁, T₁, T₀ᶜ]"


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
            ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int"), size=ti.Scalar()),
            ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int"), size=ti.Scalar()),
            ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int"), size=ti.Scalar()),
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=0),
                size=ti.Column(),
                current_loc=ti.TypeVar(idx=1),
                defined_loc=ti.TypeVar(idx=1),
            ),
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=0),
                size=ti.Column(),
                current_loc=ti.TypeVar(idx=2),
                defined_loc=ti.TypeVar(idx=2),
            ),
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=3),
                size=ti.Column(),
                current_loc=ti.TypeVar(idx=4),
                defined_loc=ti.TypeVar(idx=4),
            ),
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=3),
                size=ti.Column(),
                current_loc=ti.TypeVar(idx=5),
                defined_loc=ti.TypeVar(idx=5),
            ),
        ),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert (
        ti.pformat(inferred)
        == "{f(intˢ, intˢ, intˢ, It[T₁, T₁, T₀ᶜ], It[T₂, T₂, T₀ᶜ], It[T₄, T₄, T₃ᶜ], It[T₅, T₅, T₃ᶜ])}"
    )


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
            ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int"), size=ti.Scalar()),
            ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int"), size=ti.Scalar()),
            ti.Val(kind=ti.Value(), dtype=ti.Primitive(name="int"), size=ti.Scalar()),
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=4),
                size=ti.Column(),
                current_loc=ti.TypeVar(idx=5),
                defined_loc=ti.TypeVar(idx=5),
            ),
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=4),
                size=ti.Column(),
                current_loc=ti.TypeVar(idx=6),
                defined_loc=ti.TypeVar(idx=6),
            ),
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=7),
                size=ti.Column(),
                current_loc=ti.TypeVar(idx=8),
                defined_loc=ti.TypeVar(idx=8),
            ),
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=7),
                size=ti.Column(),
                current_loc=ti.TypeVar(idx=9),
                defined_loc=ti.TypeVar(idx=9),
            ),
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=10),
                size=ti.Column(),
                current_loc=ti.TypeVar(idx=11),
                defined_loc=ti.TypeVar(idx=11),
            ),
            ti.Val(
                kind=ti.Iterator(),
                dtype=ti.TypeVar(idx=10),
                size=ti.Column(),
                current_loc=ti.TypeVar(idx=12),
                defined_loc=ti.TypeVar(idx=12),
            ),
        ),
    )
    inferred = ti.infer(testee)
    assert inferred == expected
    assert (
        ti.pformat(inferred)
        == "{f :: (T₀) → T₀, g :: (It[T₃, T₃, T₁²]) → T₁², foo(intˢ, intˢ, intˢ, It[T₅, T₅, T₄ᶜ], It[T₆, T₆, T₄ᶜ], It[T₈, T₈, T₇ᶜ], It[T₉, T₉, T₇ᶜ], It[T₁₁, T₁₁, T₁₀ᶜ], It[T₁₂, T₁₂, T₁₀ᶜ])}"
    )


def test_pformat():
    vs = [ti.TypeVar(idx=i) for i in range(4)]
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
