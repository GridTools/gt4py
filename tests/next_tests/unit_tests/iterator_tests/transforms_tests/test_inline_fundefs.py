# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import inline_fundefs
from gt4py.next.type_system import type_specifications as ts


TDim = common.Dimension(value="TDim")
int_type = ts.ScalarType(kind=ts.ScalarKind.INT32)
domain = im.domain(common.GridType.CARTESIAN, {TDim: (0, 1)})


def program_factory(
    body: list[itir.Stmt], function_definitions: list[itir.FunctionDefinition]
) -> itir.Program:
    return itir.Program(
        id="testee",
        function_definitions=function_definitions,
        params=[
            im.sym("inp", ts.FieldType(dims=[TDim], dtype=int_type)),
            im.sym("out", ts.FieldType(dims=[TDim], dtype=int_type)),
        ],
        declarations=[],
        body=body,
    )


def test_simple():
    fun = itir.FunctionDefinition(id="fun", params=[im.sym("a")], expr=im.deref("a"))
    testee = program_factory(
        body=[itir.SetAt(expr=im.call("fun")("inp"), target=im.ref("out"), domain=domain)],
        function_definitions=[fun],
    )
    expected = program_factory(
        body=[
            itir.SetAt(
                expr=im.let("fun", im.lambda_("a")(im.deref("a")))(im.call("fun")("inp")),
                target=im.ref("out"),
                domain=domain,
            )
        ],
        function_definitions=[],
    )

    actual = inline_fundefs.inline_fundefs(testee)
    assert actual == expected


def test_unreferenced_fundef():
    # unreferenced function definitions are still bound, dead code elimination removes them later
    fun = itir.FunctionDefinition(id="fun", params=[im.sym("a")], expr=im.deref("a"))
    testee = program_factory(
        body=[itir.SetAt(expr=im.deref("inp"), target=im.ref("out"), domain=domain)],
        function_definitions=[fun],
    )
    expected = program_factory(
        body=[
            itir.SetAt(
                expr=im.let("fun", im.lambda_("a")(im.deref("a")))(im.deref("inp")),
                target=im.ref("out"),
                domain=domain,
            )
        ],
        function_definitions=[],
    )

    actual = inline_fundefs.inline_fundefs(testee)
    assert actual == expected


def test_shadowed_by_binder():
    # a binder of the same name shadows the function definition and must not be replaced by it
    fun = itir.FunctionDefinition(id="fun", params=[im.sym("a")], expr=im.deref("a"))
    stencil = im.lambda_("fun")(im.deref("fun"))
    testee = program_factory(
        body=[
            itir.SetAt(
                expr=im.as_fieldop(stencil, domain)("inp"), target=im.ref("out"), domain=domain
            )
        ],
        function_definitions=[fun],
    )

    expected = program_factory(
        body=[
            itir.SetAt(
                expr=im.let("fun", im.lambda_("a")(im.deref("a")))(
                    # the `fun` binder of the stencil is untouched
                    im.as_fieldop(stencil, domain)("inp")
                ),
                target=im.ref("out"),
                domain=domain,
            )
        ],
        function_definitions=[],
    )

    actual = inline_fundefs.inline_fundefs(testee)
    assert actual == expected


def test_dependent_fundefs():
    # function definitions may reference each other, independent of their order in the program
    fun1 = itir.FunctionDefinition(id="fun1", params=[im.sym("a")], expr=im.deref("a"))
    fun2 = itir.FunctionDefinition(id="fun2", params=[im.sym("a")], expr=im.call("fun1")("a"))
    testee = program_factory(
        body=[itir.SetAt(expr=im.call("fun2")("inp"), target=im.ref("out"), domain=domain)],
        function_definitions=[fun2, fun1],
    )
    expected = program_factory(
        body=[
            itir.SetAt(
                expr=im.let("fun1", im.lambda_("a")(im.deref("a")))(
                    im.let("fun2", im.lambda_("a")(im.call("fun1")("a")))(im.call("fun2")("inp"))
                ),
                target=im.ref("out"),
                domain=domain,
            )
        ],
        function_definitions=[],
    )

    actual = inline_fundefs.inline_fundefs(testee)
    assert actual == expected


def test_if_stmt():
    fun = itir.FunctionDefinition(id="fun", params=[im.sym("a")], expr=im.deref("a"))
    testee = program_factory(
        body=[
            itir.IfStmt(
                cond=im.call("fun")(True),
                true_branch=[
                    itir.SetAt(expr=im.call("fun")("inp"), target=im.ref("out"), domain=domain)
                ],
                false_branch=[],
            )
        ],
        function_definitions=[fun],
    )

    actual = inline_fundefs.inline_fundefs(testee)
    binding = im.let("fun", im.lambda_("a")(im.deref("a")))
    assert actual.body[0].cond == binding(im.call("fun")(True))
    assert actual.body[0].true_branch[0].expr == binding(im.call("fun")("inp"))
