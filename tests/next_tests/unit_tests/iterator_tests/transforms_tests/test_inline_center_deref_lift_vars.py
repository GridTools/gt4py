# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.type_system import type_specifications as ts
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import cse
from gt4py.next.iterator.transforms.inline_center_deref_lift_vars import InlineCenterDerefLiftVars

field_type = ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))


def wrap_in_program(expr: itir.Expr, *, arg_dtypes=None) -> itir.Program:
    if arg_dtypes is None:
        arg_dtypes = [ts.ScalarKind.FLOAT64]
    arg_types = [ts.FieldType(dims=[], dtype=ts.ScalarType(kind=dtype)) for dtype in arg_dtypes]
    indices = [i for i in range(1, len(arg_dtypes) + 1)] if len(arg_dtypes) > 1 else [""]
    return itir.Program(
        id="f",
        function_definitions=[],
        params=[
            *(im.sym(f"inp{i}", type_) for i, type_ in zip(indices, arg_types)),
            im.sym("out", field_type),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(im.lambda_(*(f"it{i}" for i in indices))(expr))(
                    *(im.ref(f"inp{i}") for i in indices)
                ),
                domain=im.call("cartesian_domain")(),
                target=im.ref("out"),
            )
        ],
    )


def unwrap_from_program(program: itir.Program) -> itir.Expr:
    stencil = program.body[0].expr.fun.args[0]
    return stencil.expr


def test_simple():
    testee = im.let("var", im.lift("deref")("it"))(im.deref("var"))
    expected = "(λ(_icdlv_1) → ·(↑(λ() → _icdlv_1()))())(λ() → ·it)"

    actual = unwrap_from_program(InlineCenterDerefLiftVars.apply(wrap_in_program(testee)))
    assert str(actual) == expected


def test_double_deref():
    testee = im.let("var", im.lift("deref")("it"))(im.plus(im.deref("var"), im.deref("var")))
    expected = "(λ(_icdlv_1) → ·(↑(λ() → _icdlv_1()))() + ·(↑(λ() → _icdlv_1()))())(λ() → ·it)"

    actual = unwrap_from_program(InlineCenterDerefLiftVars.apply(wrap_in_program(testee)))
    assert str(actual) == expected


def test_deref_at_non_center_different_pos():
    testee = im.let("var", im.lift("deref")("it"))(im.deref(im.shift("I", 1)("var")))

    actual = unwrap_from_program(InlineCenterDerefLiftVars.apply(wrap_in_program(testee)))
    assert testee == actual


def test_deref_at_multiple_pos():
    testee = im.let("var", im.lift("deref")("it"))(
        im.plus(im.deref("var"), im.deref(im.shift("I", 1)("var")))
    )

    actual = unwrap_from_program(InlineCenterDerefLiftVars.apply(wrap_in_program(testee)))
    assert testee == actual


def test_bc():
    # we also check that the common subexpression is able to extract the inlined value, such
    # that it is only evaluated once
    testee = im.let("var", im.lift("deref")("it2"))(
        im.if_(
            im.deref("it1"), im.literal_from_value(0.0), im.plus(im.deref("var"), im.deref("var"))
        )
    )
    expected = """(λ(_icdlv_1) → if ·it1 then 0.0 else (λ(_cs_1) → _cs_1 + _cs_1)(·(↑(λ() → _icdlv_1()))()))(
  λ() → ·it2
)"""

    actual = InlineCenterDerefLiftVars.apply(
        wrap_in_program(testee, arg_dtypes=[ts.ScalarKind.BOOL, ts.ScalarKind.FLOAT64])
    )
    simplified = unwrap_from_program(cse.CommonSubexpressionElimination.apply(actual))
    assert str(simplified) == expected
