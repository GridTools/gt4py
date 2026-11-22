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
    body: list[itir.Stmt],
    function_definitions: list[itir.FunctionDefinition],
    declarations: list[itir.Temporary] | None = None,
) -> itir.Program:
    return itir.Program(
        id="testee",
        function_definitions=function_definitions,
        params=[
            im.sym("inp", ts.FieldType(dims=[TDim], dtype=int_type)),
            im.sym("out", ts.FieldType(dims=[TDim], dtype=int_type)),
        ],
        declarations=declarations or [],
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
    # an unreferenced function definition is not bound at all
    fun = itir.FunctionDefinition(id="fun", params=[im.sym("a")], expr=im.deref("a"))
    testee = program_factory(
        body=[itir.SetAt(expr=im.deref("inp"), target=im.ref("out"), domain=domain)],
        function_definitions=[fun],
    )
    expected = program_factory(
        body=[itir.SetAt(expr=im.deref("inp"), target=im.ref("out"), domain=domain)],
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
                expr=im.as_fieldop(stencil, domain)(im.call("fun")("inp")),
                target=im.ref("out"),
                domain=domain,
            )
        ],
        function_definitions=[fun],
    )

    expected = program_factory(
        body=[
            itir.SetAt(
                expr=im.let("fun", im.lambda_("a")(im.deref("a")))(
                    # the `fun` binder of the stencil is untouched, the call is bound
                    im.as_fieldop(stencil, domain)(im.call("fun")("inp"))
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


def test_param_shadowing_own_fundef():
    # the parameter of `foo` shadows the function definition `foo` itself, which is not a
    # self-reference and hence not a dependency cycle
    foo = itir.FunctionDefinition(id="foo", params=[im.sym("foo")], expr=im.deref("foo"))
    testee = program_factory(
        body=[itir.SetAt(expr=im.call("foo")("inp"), target=im.ref("out"), domain=domain)],
        function_definitions=[foo],
    )
    expected = program_factory(
        body=[
            itir.SetAt(
                expr=im.let("foo", im.lambda_("foo")(im.deref("foo")))(im.call("foo")("inp")),
                target=im.ref("out"),
                domain=domain,
            )
        ],
        function_definitions=[],
    )

    actual = inline_fundefs.inline_fundefs(testee)
    assert actual == expected


def test_params_shadowing_other_fundefs():
    # `f` and `g` do not reference each other, their parameters only happen to be named like the
    # respective other function definition
    f = itir.FunctionDefinition(id="f", params=[im.sym("g")], expr=im.deref("g"))
    g = itir.FunctionDefinition(id="g", params=[im.sym("f")], expr=im.deref("f"))
    testee = program_factory(
        body=[
            itir.SetAt(expr=im.call("f")(im.call("g")("inp")), target=im.ref("out"), domain=domain)
        ],
        function_definitions=[f, g],
    )
    expected = program_factory(
        body=[
            itir.SetAt(
                expr=im.let("f", im.lambda_("g")(im.deref("g")))(
                    im.let("g", im.lambda_("f")(im.deref("f")))(im.call("f")(im.call("g")("inp")))
                ),
                target=im.ref("out"),
                domain=domain,
            )
        ],
        function_definitions=[],
    )

    actual = inline_fundefs.inline_fundefs(testee)
    assert actual == expected


def test_param_shadowing_fundef_with_dependency():
    # `mixed` shadows `helper` with its parameter, but genuinely depends on `other`, which
    # therefore has to be bound further outside
    helper = itir.FunctionDefinition(id="helper", params=[im.sym("x")], expr=im.deref("x"))
    other = itir.FunctionDefinition(id="other", params=[im.sym("x")], expr=im.deref("x"))
    mixed = itir.FunctionDefinition(
        id="mixed", params=[im.sym("helper")], expr=im.call("other")("helper")
    )
    testee = program_factory(
        body=[itir.SetAt(expr=im.call("mixed")("inp"), target=im.ref("out"), domain=domain)],
        function_definitions=[mixed, helper, other],
    )
    expected = program_factory(
        body=[
            itir.SetAt(
                # `helper` is only shadowed, never referenced, hence it is not bound
                expr=im.let("other", im.lambda_("x")(im.deref("x")))(
                    im.let("mixed", im.lambda_("helper")(im.call("other")("helper")))(
                        im.call("mixed")("inp")
                    )
                ),
                target=im.ref("out"),
                domain=domain,
            )
        ],
        function_definitions=[],
    )

    actual = inline_fundefs.inline_fundefs(testee)
    assert actual == expected


def test_fundef_referenced_in_domain():
    # the domain of a statement is wrapped just like its expression
    upper = itir.FunctionDefinition(id="upper", params=[], expr=im.literal_from_value(10))
    symbolic_domain = im.domain(common.GridType.CARTESIAN, {TDim: (0, im.call("upper")())})
    testee = program_factory(
        body=[itir.SetAt(expr=im.deref("inp"), target=im.ref("out"), domain=symbolic_domain)],
        function_definitions=[upper],
    )
    expected = program_factory(
        body=[
            itir.SetAt(
                expr=im.deref("inp"),
                target=im.ref("out"),
                domain=im.let("upper", im.lambda_()(im.literal_from_value(10)))(symbolic_domain),
            )
        ],
        function_definitions=[],
    )

    actual = inline_fundefs.inline_fundefs(testee)
    assert actual == expected


def test_fundef_referenced_in_temporary_domain():
    # the domain of a temporary is wrapped as well
    upper = itir.FunctionDefinition(id="upper", params=[], expr=im.literal_from_value(10))
    symbolic_domain = im.domain(common.GridType.CARTESIAN, {TDim: (0, im.call("upper")())})
    testee = program_factory(
        body=[itir.SetAt(expr=im.deref("inp"), target=im.ref("out"), domain=domain)],
        function_definitions=[upper],
        declarations=[itir.Temporary(id="tmp", domain=symbolic_domain, dtype=int_type)],
    )
    expected = program_factory(
        body=[itir.SetAt(expr=im.deref("inp"), target=im.ref("out"), domain=domain)],
        function_definitions=[],
        declarations=[
            itir.Temporary(
                id="tmp",
                domain=im.let("upper", im.lambda_()(im.literal_from_value(10)))(symbolic_domain),
                dtype=int_type,
            )
        ],
    )

    actual = inline_fundefs.inline_fundefs(testee)
    assert actual == expected
