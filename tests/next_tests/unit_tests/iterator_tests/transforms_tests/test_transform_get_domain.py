# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Optional

import pytest
from next_tests.integration_tests.cases import (
    IField,
    IDim,
    KDim,
    Vertex,
    unstructured_case,
    exec_alloc_descriptor,
)

from gt4py import next as gtx
from gt4py.next import Domain
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import inline_fundefs, global_tmps, inline_lambdas
from gt4py.next.iterator.transforms.transform_get_domain import TransformGetDomain
from gt4py.next.iterator.type_system import inference as type_inference
from gt4py.next.type_system import type_specifications as ts
from tests.next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    simple_cartesian_grid,
    Edge,
    simple_mesh,
)

IOff = gtx.FieldOffset("IOff", source=IDim, target=(IDim,))

float_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
int_type = ts.ScalarType(kind=ts.ScalarKind.INT32)
v_field_type = ts.FieldType(dims=[Vertex], dtype=float_type)
ve_field_type = ts.FieldType(dims=[Edge, Vertex], dtype=float_type)
e_field_type = ts.FieldType(dims=[Edge], dtype=float_type)
i_field_type = ts.FieldType(dims=[IDim], dtype=float_type)


# TODO: maybe check if domains are consistent in global_tmps


# override mesh descriptor to contain only the simple mesh
@pytest.fixture
def mesh_descriptor(exec_alloc_descriptor):
    return simple_mesh(exec_alloc_descriptor.allocator)


def program_factory(
    params: list[itir.Sym],
    body: list[itir.SetAt],
    declarations: Optional[list[itir.Temporary]] = None,
) -> itir.Program:
    return itir.Program(
        id="testee",
        function_definitions=[],
        params=params,
        declarations=declarations or [],
        body=body,
    )


def construct_domains(domain_resolved: Domain, symbol_name: str, type: str):
    named_ranges_get, named_ranges_resolved = [], []

    for dim, range_ in zip(domain_resolved.dims, domain_resolved.ranges):
        get_domain_call = im.call("get_domain")(symbol_name, im.axis_literal(dim))
        named_ranges_get.append(
            im.named_range(
                im.axis_literal(dim),
                im.tuple_get(0, get_domain_call),
                im.tuple_get(1, get_domain_call),
            )
        )
        bounds_tuple = im.make_tuple(range_.start, range_.stop)
        named_ranges_resolved.append(
            im.named_range(
                im.axis_literal(dim), im.tuple_get(0, bounds_tuple), im.tuple_get(1, bounds_tuple)
            )
        )

    return im.call(type)(*named_ranges_resolved), im.call(type)(*named_ranges_get)


def test_get_domain():
    sizes = {"out": gtx.domain({Vertex: (0, 10), KDim: (0, 20)})}
    unstructured_domain, unstructured_domain_get = construct_domains(
        sizes["out"], "out", "unstructured_domain"
    )

    testee = program_factory(
        params=[im.sym("inp"), im.sym("out")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(im.ref("deref"))(im.ref("inp")),
                domain=unstructured_domain_get,
                target=im.ref("out"),
            ),
        ],
    )

    expected = program_factory(
        params=[im.sym("inp"), im.sym("out")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(im.ref("deref"))(im.ref("inp")),
                domain=unstructured_domain,
                target=im.ref("out"),
            ),
        ],
    )

    actual = TransformGetDomain.apply(testee, sizes=sizes)
    assert actual == expected


def test_get_domain_inside_as_fieldop():
    sizes = {"out": gtx.domain({Vertex: (0, 10), KDim: (0, 20)})}
    unstructured_domain, unstructured_domain_get = construct_domains(
        sizes["out"], "out", "unstructured_domain"
    )

    testee = program_factory(
        params=[im.sym("inp"), im.sym("out")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(im.ref("deref"), unstructured_domain_get)(im.ref("inp")),
                domain=unstructured_domain_get,
                target=im.ref("out"),
            ),
        ],
    )

    expected = program_factory(
        params=[im.sym("inp"), im.sym("out")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(im.ref("deref"), unstructured_domain)(im.ref("inp")),
                domain=unstructured_domain,
                target=im.ref("out"),
            ),
        ],
    )

    actual = TransformGetDomain.apply(testee, sizes=sizes)
    assert actual == expected


def test_get_domain_tuples():
    sizes = {"out": (gtx.domain({Vertex: (0, 5)}), gtx.domain({Vertex: (0, 7)}))}

    unstructured_domain_get = im.call("unstructured_domain")(
        im.named_range(
            im.axis_literal(Vertex),
            im.tuple_get(0, im.call("get_domain")(im.tuple_get(0, "out"), im.axis_literal(Vertex))),
            im.tuple_get(1, im.call("get_domain")(im.tuple_get(0, "out"), im.axis_literal(Vertex))),
        )
    )
    unstructured_domain = im.call("unstructured_domain")(
        im.named_range(
            im.axis_literal(Vertex),
            im.tuple_get(0, im.make_tuple(0, 5)),
            im.tuple_get(1, im.make_tuple(0, 5)),
        ),
    )

    testee = program_factory(
        params=[im.sym("inp"), im.sym("out")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(im.ref("deref"))(im.ref("inp")),
                domain=unstructured_domain_get,
                target=im.tuple_get(0, "out"),
            ),
        ],
    )

    expected = program_factory(
        params=[im.sym("inp"), im.sym("out")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(im.ref("deref"))(im.ref("inp")),
                domain=unstructured_domain,
                target=im.tuple_get(0, "out"),
            ),
        ],
    )

    actual = TransformGetDomain.apply(testee, sizes=sizes)
    assert actual == expected


def test_get_domain_nested_tuples():
    sizes = {"a": gtx.domain({KDim: (0, 3)})}

    t0 = im.make_tuple("a", "b")
    t1 = im.make_tuple("c", "d")
    tup = im.make_tuple(im.tuple_get(0, t0), im.tuple_get(1, t1))
    unstructured_domain_get = im.call("unstructured_domain")(
        im.named_range(
            im.axis_literal(KDim),
            im.tuple_get(0, im.call("get_domain")(im.tuple_get(0, tup), im.axis_literal(KDim))),
            im.tuple_get(1, im.call("get_domain")(im.tuple_get(0, tup), im.axis_literal(KDim))),
        )
    )
    unstructured_domain = im.call("unstructured_domain")(
        im.named_range(
            im.axis_literal(KDim),
            im.tuple_get(0, im.make_tuple(0, 3)),
            im.tuple_get(1, im.make_tuple(0, 3)),
        ),
    )

    testee = program_factory(
        params=[im.sym("inp"), im.sym("a"), im.sym("b"), im.sym("c"), im.sym("d")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(im.ref("deref"))(im.ref("inp")),
                domain=unstructured_domain_get,
                target=im.ref("a"),
            ),
        ],
    )

    expected = program_factory(
        params=[im.sym("inp"), im.sym("a"), im.sym("b"), im.sym("c"), im.sym("d")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(im.ref("deref"))(im.ref("inp")),
                domain=unstructured_domain,
                target=im.ref("a"),
            ),
        ],
    )

    actual = TransformGetDomain.apply(testee, sizes=sizes)
    assert actual == expected


@pytest.fixture
def testee():
    @gtx.field_operator
    def testee_tmp(x: IField) -> IField:
        y = x(IOff[2])
        return y(IOff[3])

    # @gtx.field_operator
    # def testee_tmp(x: IField) -> IField:
    #     return x(IOff[1])

    # @gtx.field_operator
    # def testee_op(x: IField) -> IField:
    #     return testee_tmp(x)

    @gtx.field_operator
    def testee_op(x: IField) -> IField:
        return testee_tmp(x) + testee_tmp(x)

    @gtx.program(static_domain_sizes=True, grid_type=gtx.GridType.UNSTRUCTURED)
    def prog(
        inp: IField,
        out: IField,
    ):
        testee_op(inp, out=out)

    return prog


def test_get_domain_inference_temporary_symbols(testee, unstructured_case):
    sizes = {"out": gtx.domain({IDim: (0, 20)})}
    ir = inline_fundefs.InlineFundefs().visit(testee.gtir)
    ir = inline_fundefs.prune_unreferenced_fundefs(ir)
    ir = type_inference.infer(
        ir, offset_provider_type={**unstructured_case.offset_provider, "IOff": IDim}
    )
    ir = inline_lambdas.InlineLambdas.apply(ir)
    ir = global_tmps.create_global_tmps(
        ir, offset_provider={**unstructured_case.offset_provider, "IOff": IDim}
    )
    ir = TransformGetDomain.apply(ir, sizes=sizes)
    # ir = infer_domain.infer_program(
    #     ir, offset_provider={**unstructured_case.offset_provider, "IOff": IDim}
    # ) # TODO: domain inference does not seem to be necessary anymore since it is already done in create_global_tmps
    unstructured_domain = im.call("unstructured_domain")(
        im.call("named_range")(
            im.axis_literal(IDim),
            im.tuple_get(0, im.make_tuple(0, 20)),
            im.tuple_get(1, im.make_tuple(0, 20)),
        )
    )

    unstructured_domain_p3 = im.call("unstructured_domain")(
        im.call("named_range")(
            im.axis_literal(IDim),
            im.plus(im.tuple_get(0, im.make_tuple(0, 20)), 3),
            im.plus(im.tuple_get(1, im.make_tuple(0, 20)), 3),
        )
    )

    expected = itir.Program(
        id="prog",
        function_definitions=[],
        params=[im.sym("inp"), im.sym("out"), im.sym("__inp_0_range"), im.sym("__out_0_range")],
        declarations=[
            itir.Temporary(id="__tmp_1", domain=unstructured_domain, dtype=int_type),
            itir.Temporary(id="__tmp_2", domain=unstructured_domain_p3, dtype=int_type),
        ],
        body=[
            itir.SetAt(
                target=im.ref("__tmp_2"),
                expr=im.as_fieldop(
                    im.lambda_("__it")(im.deref(im.shift("IOff", 2)("__it"))),
                    unstructured_domain_p3,
                )("inp"),
                domain=unstructured_domain_p3,
            ),
            itir.SetAt(
                target=im.ref("__tmp_1"),
                expr=im.as_fieldop(
                    im.lambda_("__it_")(im.deref(im.shift("IOff", 3)("__it_"))), unstructured_domain
                )("__tmp_2"),
                domain=unstructured_domain,
            ),
            itir.SetAt(
                target=im.ref("out"),
                expr=im.as_fieldop(
                    im.lambda_("__arg0", "__arg1")(im.plus(im.deref("__arg0"), im.deref("__arg1"))),
                    unstructured_domain,
                )("__tmp_1", "__tmp_1"),
                domain=unstructured_domain,
            ),
        ],
    )
    assert ir == expected


def test_trivial_shift(unstructured_case):
    sizes = {"out": gtx.domain({Edge: (0, 18), Vertex: (0, 9)})}
    unstructured_domain_get_E = im.call("unstructured_domain")(
        im.call("named_range")(
            im.axis_literal(Edge),
            im.tuple_get(0, im.call("get_domain")("out", im.axis_literal(Edge))),
            im.tuple_get(1, im.call("get_domain")("out", im.axis_literal(Edge))),
        )
    )

    unstructured_domain_E = im.call("unstructured_domain")(
        im.call("named_range")(
            im.axis_literal(Edge),
            im.tuple_get(0, im.make_tuple(0, 18)),
            im.tuple_get(1, im.make_tuple(0, 18)),
        )
    )

    unstructured_domain_V_p1_expected = im.call("unstructured_domain")(
        im.call("named_range")(im.axis_literal(Vertex), 1, 9),
    )

    offset_provider = unstructured_case.offset_provider
    testee = program_factory(
        params=[im.sym("vertex_values", v_field_type), im.sym("out", e_field_type)],
        body=[
            itir.SetAt(
                target=im.ref("out"),
                expr=im.as_fieldop(im.lambda_("x")(im.deref(im.shift("E2V", 1)("x"))))(
                    im.as_fieldop("deref")("vertex_values")
                ),
                domain=unstructured_domain_get_E,
            )
        ],
    )

    ir = inline_fundefs.InlineFundefs().visit(testee)
    ir = inline_fundefs.prune_unreferenced_fundefs(ir)
    ir = global_tmps.create_global_tmps(ir, offset_provider=offset_provider)
    ir = TransformGetDomain.apply(ir, sizes=sizes)
    # ir = infer_domain.infer_program(ir, offset_provider=offset_provider) # TODO: domain inference does not seem to be necessary anymore since it is already done in create_global_tmps

    expected = program_factory(
        params=[im.sym("vertex_values", v_field_type), im.sym("out", e_field_type)],
        declarations=[
            itir.Temporary(id="__tmp_1", domain=unstructured_domain_V_p1_expected, dtype=float_type)
        ],
        body=[
            itir.SetAt(
                target=im.ref("__tmp_1"),
                expr=im.as_fieldop("deref", unstructured_domain_V_p1_expected)("vertex_values"),
                domain=unstructured_domain_V_p1_expected,
            ),
            itir.SetAt(
                target=im.ref("out"),
                expr=im.as_fieldop(
                    im.lambda_("x")(im.deref(im.shift("E2V", 1)("x"))), unstructured_domain_E
                )("__tmp_1"),
                domain=unstructured_domain_E,
            ),
        ],
    )

    assert ir == expected


def test_trivial_shift_switched(unstructured_case):
    sizes = {"out": gtx.domain({Edge: (0, 18), Vertex: (0, 9)})}
    unstructured_domain_get_E = im.call("unstructured_domain")(
        im.call("named_range")(
            im.axis_literal(Edge),
            im.tuple_get(0, im.call("get_domain")("out", im.axis_literal(Edge))),
            im.tuple_get(1, im.call("get_domain")("out", im.axis_literal(Edge))),
        )
    )

    unstructured_domain_E = im.call("unstructured_domain")(
        im.call("named_range")(
            im.axis_literal(Edge),
            im.tuple_get(0, im.make_tuple(0, 18)),
            im.tuple_get(1, im.make_tuple(0, 18)),
        )
    )

    offset_provider = unstructured_case.offset_provider
    testee = program_factory(
        params=[im.sym("vertex_values", v_field_type), im.sym("out", e_field_type)],
        body=[
            itir.SetAt(
                target=im.ref("out"),
                expr=im.as_fieldop("deref")(
                    im.as_fieldop(im.lambda_("x")(im.deref(im.shift("E2V", 1)("x"))))(
                        "vertex_values"
                    )
                ),
                domain=unstructured_domain_get_E,
            )
        ],
    )

    ir = inline_fundefs.InlineFundefs().visit(testee)
    ir = inline_fundefs.prune_unreferenced_fundefs(ir)
    ir = global_tmps.create_global_tmps(ir, offset_provider=offset_provider)
    ir = TransformGetDomain.apply(ir, sizes=sizes)
    # ir = infer_domain.infer_program(ir, offset_provider=offset_provider) # TODO: domain inference does not seem to be necessary anymore since it is already done in create_global_tmps

    expected = program_factory(
        params=[im.sym("vertex_values", v_field_type), im.sym("out", e_field_type)],
        declarations=[itir.Temporary(id="__tmp_1", domain=unstructured_domain_E, dtype=float_type)],
        body=[
            itir.SetAt(
                target=im.ref("__tmp_1"),
                expr=im.as_fieldop(
                    im.lambda_("x")(im.deref(im.shift("E2V", 1)("x"))), unstructured_domain_E
                )("vertex_values"),
                domain=unstructured_domain_E,
            ),
            itir.SetAt(
                target=im.ref("out"),
                expr=im.as_fieldop("deref", unstructured_domain_E)("__tmp_1"),
                domain=unstructured_domain_E,
            ),
        ],
    )

    assert ir == expected


def test_two_shifts(unstructured_case):
    sizes = {"out": gtx.domain({Edge: (0, 18), Vertex: (0, 9)})}
    unstructured_domain_get_E = im.call("unstructured_domain")(
        im.call("named_range")(
            im.axis_literal(Edge),
            im.tuple_get(0, im.call("get_domain")("out", im.axis_literal(Edge))),
            im.tuple_get(1, im.call("get_domain")("out", im.axis_literal(Edge))),
        )
    )

    unstructured_domain_E = im.call("unstructured_domain")(
        im.call("named_range")(
            im.axis_literal(Edge),
            im.tuple_get(0, im.make_tuple(0, 18)),
            im.tuple_get(1, im.make_tuple(0, 18)),
        )
    )

    unstructured_domain_V_expected = im.call("unstructured_domain")(
        im.call("named_range")(im.axis_literal(Vertex), 0, 9),
    )
    offset_provider = unstructured_case.offset_provider
    testee = program_factory(
        params=[im.sym("vertex_values", v_field_type), im.sym("out", e_field_type)],
        body=[
            itir.SetAt(
                target=im.ref("out"),
                expr=im.as_fieldop(
                    im.lambda_("x")(
                        im.plus(
                            im.deref(im.shift("E2V", 0)("x")), im.deref(im.shift("E2V", 1)("x"))
                        )
                    )
                )(im.as_fieldop("deref")("vertex_values")),
                domain=unstructured_domain_get_E,
            )
        ],
    )

    ir = inline_fundefs.InlineFundefs().visit(testee)
    ir = inline_fundefs.prune_unreferenced_fundefs(ir)
    ir = global_tmps.create_global_tmps(ir, offset_provider=offset_provider)
    ir = TransformGetDomain.apply(ir, sizes=sizes)
    # ir = infer_domain.infer_program(ir, offset_provider=offset_provider) # TODO: domain inference does not seem to be necessary anymore since it is already done in create_global_tmps

    expected = program_factory(
        params=[im.sym("vertex_values", v_field_type), im.sym("out", e_field_type)],
        declarations=[
            itir.Temporary(id="__tmp_1", domain=unstructured_domain_V_expected, dtype=float_type)
        ],
        body=[
            itir.SetAt(
                target=im.ref("__tmp_1"),
                expr=im.as_fieldop("deref", unstructured_domain_V_expected)("vertex_values"),
                domain=unstructured_domain_V_expected,
            ),
            itir.SetAt(
                target=im.ref("out"),
                expr=im.as_fieldop(
                    im.lambda_("x")(
                        im.plus(
                            im.deref(im.shift("E2V", 0)("x")), im.deref(im.shift("E2V", 1)("x"))
                        )
                    ),
                    unstructured_domain_E,
                )("__tmp_1"),
                domain=unstructured_domain_E,
            ),
        ],
    )

    assert ir == expected


def test_nested_shift(unstructured_case):
    sizes = {"out": gtx.domain({Edge: (0, 18), Vertex: (0, 9)})}
    unstructured_domain_V = im.call("unstructured_domain")(
        im.call("named_range")(
            im.axis_literal(Vertex),
            im.tuple_get(0, im.make_tuple(0, 9)),
            im.tuple_get(1, im.make_tuple(0, 9)),
        )
    )
    unstructured_domain_get_V = im.call("unstructured_domain")(
        im.call("named_range")(
            im.axis_literal(Vertex),
            im.tuple_get(0, im.call("get_domain")("out", im.axis_literal(Vertex))),
            im.tuple_get(1, im.call("get_domain")("out", im.axis_literal(Vertex))),
        )
    )

    unstructured_domain_V_p1_expected = im.call("unstructured_domain")(
        im.call("named_range")(im.axis_literal(Vertex), 1, 9),
    )

    unstructured_domain_E_918_expected = im.call("unstructured_domain")(
        im.call("named_range")(im.axis_literal(Edge), 9, 18),
    )

    offset_provider = unstructured_case.offset_provider

    testee = program_factory(
        params=[im.sym("vertex_values", v_field_type), im.sym("out", v_field_type)],
        body=[
            itir.SetAt(
                target=im.ref("out"),
                expr=im.as_fieldop(im.lambda_("x")(im.deref(im.shift("V2E", 1)("x"))))(
                    im.as_fieldop(im.lambda_("x")(im.deref(im.shift("E2V", 1)("x"))))(
                        im.as_fieldop("deref")("vertex_values")
                    )
                ),
                domain=unstructured_domain_get_V,
            )
        ],
    )

    # testee = program_factory(
    #     params=[im.sym("vertex_values", v_field_type), im.sym("out", v_field_type)],
    #     body=[
    #         itir.SetAt(
    #             target=im.ref("out"),
    #             expr=im.as_fieldop(
    #                 im.lambda_("x")(im.deref(im.shift("E2V", 1)(
    #                     im.shift("V2E", 1)("x")))))(im.as_fieldop("deref")("vertex_values")),
    #             domain=unstructured_domain_get_V, # TODO: why is the order switched in here?
    #         )
    #     ],
    # )

    ir = inline_fundefs.InlineFundefs().visit(testee)
    ir = inline_fundefs.prune_unreferenced_fundefs(ir)
    ir = global_tmps.create_global_tmps(ir, offset_provider=offset_provider)
    ir = TransformGetDomain.apply(ir, sizes=sizes)
    # ir = infer_domain.infer_program(ir, offset_provider=offset_provider) # TODO: domain inference does not seem to be necessary anymore since it is already done in create_global_tmps

    expected = program_factory(
        params=[im.sym("vertex_values", v_field_type), im.sym("out", e_field_type)],
        declarations=[
            itir.Temporary(
                id="__tmp_1", domain=unstructured_domain_E_918_expected, dtype=float_type
            ),
            itir.Temporary(
                id="__tmp_2", domain=unstructured_domain_V_p1_expected, dtype=float_type
            ),
        ],
        body=[
            itir.SetAt(
                target=im.ref("__tmp_2"),
                expr=im.as_fieldop("deref", unstructured_domain_V_p1_expected)("vertex_values"),
                domain=unstructured_domain_V_p1_expected,
            ),
            itir.SetAt(
                target=im.ref("__tmp_1"),
                expr=im.as_fieldop(
                    im.lambda_("x")(im.deref(im.shift("E2V", 1)("x"))),
                    unstructured_domain_E_918_expected,
                )("__tmp_2"),
                domain=unstructured_domain_E_918_expected,
            ),
            itir.SetAt(
                target=im.ref("out"),
                expr=im.as_fieldop(
                    im.lambda_("x")(im.deref(im.shift("V2E", 1)("x"))), unstructured_domain_V
                )("__tmp_1"),
                domain=unstructured_domain_V,
            ),
        ],
    )

    assert ir == expected


def test_trivial_cartesian():  # TODO: fix/remove?
    grid = simple_cartesian_grid()
    offset_provider = {"Ioff": grid.offset_provider["Ioff"]}
    sizes = {"out": gtx.domain({IDim: (0, 8)})}
    cartesian_domain, cartesian_domain_get = construct_domains(
        sizes["out"], "out", "cartesian_domain"
    )
    cartesian_domain_p1 = im.call("cartesian_domain")(
        im.call("named_range")(
            im.axis_literal(IDim),
            im.plus(im.tuple_get(0, im.make_tuple(0, 8)), 1),
            im.plus(im.tuple_get(1, im.make_tuple(0, 8)), 1),
        )
    )

    testee = program_factory(
        params=[im.sym("i_values", i_field_type), im.sym("out", i_field_type)],
        body=[
            itir.SetAt(
                target=im.ref("out"),
                expr=im.as_fieldop(im.lambda_("x")(im.deref(im.shift("Ioff", 1)("x"))))(
                    im.as_fieldop("deref")("i_values")
                ),
                domain=cartesian_domain_get,
            )
        ],
    )

    ir = inline_fundefs.InlineFundefs().visit(testee)
    ir = inline_fundefs.prune_unreferenced_fundefs(ir)
    ir = global_tmps.create_global_tmps(ir, offset_provider=offset_provider)
    ir = TransformGetDomain.apply(ir, sizes=sizes)
    # ir = infer_domain.infer_program(ir, offset_provider=offset_provider) # TODO: domain inference does not seem to be necessary anymore since it is already done in create_global_tmps

    expected = program_factory(
        params=[im.sym("i_values", i_field_type), im.sym("out", i_field_type)],
        declarations=[itir.Temporary(id="__tmp_1", domain=cartesian_domain_p1, dtype=float_type)],
        body=[
            itir.SetAt(
                target=im.ref("__tmp_1"),
                expr=im.as_fieldop("deref", cartesian_domain_p1)("i_values"),
                domain=cartesian_domain_p1,
            ),
            itir.SetAt(
                target=im.ref("out"),
                expr=im.as_fieldop(
                    im.lambda_("x")(im.deref(im.shift("Ioff", 1)("x"))), cartesian_domain
                )("__tmp_1"),
                domain=cartesian_domain,
            ),
        ],
    )

    assert ir == expected


def test_trivial_cartesian_forward():  # TODO: fix/remove?
    grid = simple_cartesian_grid()
    offset_provider = {"Ioff": grid.offset_provider["Ioff"]}
    sizes = {"out": gtx.domain({IDim: (0, 8)})}

    cartesian_domain_get = im.call("cartesian_domain")(
        im.call("named_range")(
            im.axis_literal(IDim),
            im.minus(im.tuple_get(0, im.call("get_domain")("out", im.axis_literal(IDim))), 4),
            im.minus(im.tuple_get(1, im.call("get_domain")("out", im.axis_literal(IDim))), 4),
        )
    )
    testee = program_factory(
        params=[im.sym("i_values", i_field_type), im.sym("out", i_field_type)],
        body=[
            itir.SetAt(
                target=im.ref("out"),
                expr=im.as_fieldop(
                    im.lambda_("x")(im.deref(im.shift("Ioff", 2)("x"))),
                )(
                    im.as_fieldop(
                        im.lambda_("x")(im.deref(im.shift("Ioff", 2)("x"))),
                    )("i_values")
                ),
                domain=cartesian_domain_get,
            )
        ],
    )

    ir = inline_fundefs.InlineFundefs().visit(testee)
    ir = inline_fundefs.prune_unreferenced_fundefs(ir)
    ir = global_tmps.create_global_tmps(ir, offset_provider=offset_provider)
    ir = TransformGetDomain.apply(ir, sizes=sizes)
    # ir = infer_domain.infer_program(ir, offset_provider=offset_provider) # TODO: domain inference does not seem to be necessary anymore since it is already done in create_global_tmps

    cartesian_domain_m2 = im.call("cartesian_domain")(
        im.call("named_range")(
            im.axis_literal(IDim),
            im.minus(im.tuple_get(0, im.make_tuple(0, 8)), 2),
            im.minus(im.tuple_get(1, im.make_tuple(0, 8)), 2),
        )
    )

    cartesian_domain_m4 = im.call("cartesian_domain")(
        im.call("named_range")(
            im.axis_literal(IDim),
            im.minus(im.tuple_get(0, im.make_tuple(0, 8)), 4),
            im.minus(im.tuple_get(1, im.make_tuple(0, 8)), 4),
        )
    )

    expected = program_factory(
        params=[im.sym("i_values", i_field_type), im.sym("out", i_field_type)],
        declarations=[itir.Temporary(id="__tmp_1", domain=cartesian_domain_m2, dtype=float_type)],
        body=[
            itir.SetAt(
                target=im.ref("__tmp_1"),
                expr=im.as_fieldop(
                    im.lambda_("x")(im.deref(im.shift("Ioff", 2)("x"))), cartesian_domain_m2
                )("i_values"),
                domain=cartesian_domain_m2,
            ),
            itir.SetAt(
                target=im.ref("out"),
                expr=im.as_fieldop(
                    im.lambda_("x")(im.deref(im.shift("Ioff", 2)("x"))), cartesian_domain_m4
                )("__tmp_1"),
                domain=cartesian_domain_m4,
            ),
        ],
    )

    assert ir == expected
