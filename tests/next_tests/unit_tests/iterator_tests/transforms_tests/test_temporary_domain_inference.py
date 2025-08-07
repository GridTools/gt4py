# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Dict

import pytest
from next_tests.integration_tests.cases import (
    IDim,
    Vertex,
    unstructured_case,
    exec_alloc_descriptor,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    simple_cartesian_grid,
    Edge,
    simple_mesh,
)

from gt4py import next as gtx
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import (
    ir_makers as im,
)
from gt4py.next.iterator.transforms import inline_fundefs, global_tmps
from gt4py.next.iterator.transforms.transform_get_domain import TransformGetDomain
from gt4py.next.type_system import type_specifications as ts

IOff = gtx.FieldOffset("IOff", source=IDim, target=(IDim,))

float_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
v_field_type = ts.FieldType(dims=[Vertex], dtype=float_type)
e_field_type = ts.FieldType(dims=[Edge], dtype=float_type)
i_field_type = ts.FieldType(dims=[IDim], dtype=float_type)


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


def run_test_program(
    testee: itir.Program,
    expected: itir.Program,
    sizes: Dict[str, common.Domain],
    offset_provider: common.OffsetProvider,
) -> None:
    ir = inline_fundefs.InlineFundefs().visit(testee)
    ir = inline_fundefs.prune_unreferenced_fundefs(ir)
    ir = TransformGetDomain.apply(ir, sizes=sizes)
    actual_program = global_tmps.create_global_tmps(ir, offset_provider=offset_provider)

    assert actual_program == expected


def test_trivial_shift(unstructured_case):
    sizes = {"out": gtx.domain({Edge: (9, 13), Vertex: (0, 9)})}
    unstructured_domain_get_E = im.domain(
        common.GridType.UNSTRUCTURED,
        {
            Edge: (
                im.tuple_get(0, im.call("get_domain")("out", im.axis_literal(Edge))),
                im.tuple_get(1, im.call("get_domain")("out", im.axis_literal(Edge))),
            )
        },
    )

    unstructured_domain_E = im.domain(
        common.GridType.UNSTRUCTURED,
        {Edge: (im.tuple_get(0, im.make_tuple(9, 13)), im.tuple_get(1, im.make_tuple(9, 13)))},
    )

    unstructured_domain_V_37 = im.domain(common.GridType.UNSTRUCTURED, {Vertex: (3, 7)})

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

    expected = program_factory(
        params=[im.sym("vertex_values", v_field_type), im.sym("out", e_field_type)],
        declarations=[
            itir.Temporary(id="__tmp_1", domain=unstructured_domain_V_37, dtype=float_type)
        ],
        body=[
            itir.SetAt(
                target=im.ref("__tmp_1"),
                expr=im.as_fieldop("deref", unstructured_domain_V_37)("vertex_values"),
                domain=unstructured_domain_V_37,
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

    run_test_program(testee, expected, sizes, offset_provider)


def test_trivial_shift_warning(unstructured_case):
    with pytest.warns(
        UserWarning,
        match=r"For Vertex\[horizontal\] the accessed range \[3, 9\[ covers 6 values, "
        r"but only 2 are actually present and 4 were added in between \[8 3\]\. "
        r"Please consider reordering the mesh\.",
    ):
        sizes = {"out": gtx.domain({Edge: (8, 10), Vertex: (0, 9)})}
        unstructured_domain_get_E = im.domain(
            common.GridType.UNSTRUCTURED,
            {
                Edge: (
                    im.tuple_get(0, im.call("get_domain")("out", im.axis_literal(Edge))),
                    im.tuple_get(1, im.call("get_domain")("out", im.axis_literal(Edge))),
                )
            },
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
        ir = TransformGetDomain.apply(ir, sizes=sizes)

        global_tmps.create_global_tmps(ir, offset_provider=offset_provider)


def test_trivial_shift_switched(unstructured_case):
    sizes = {"out": gtx.domain({Edge: (2, 16), Vertex: (0, 9)})}
    unstructured_domain_get_E = im.domain(
        common.GridType.UNSTRUCTURED,
        {
            Edge: (
                im.tuple_get(0, im.call("get_domain")("out", im.axis_literal(Edge))),
                im.tuple_get(1, im.call("get_domain")("out", im.axis_literal(Edge))),
            )
        },
    )

    unstructured_domain_E = im.domain(
        common.GridType.UNSTRUCTURED,
        {Edge: (im.tuple_get(0, im.make_tuple(2, 16)), im.tuple_get(1, im.make_tuple(2, 16)))},
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

    run_test_program(testee, expected, sizes, offset_provider)


def test_two_shifts(unstructured_case):
    sizes = {"out": gtx.domain({Edge: (3, 8), Vertex: (0, 9)})}
    unstructured_domain_get_E = im.domain(
        common.GridType.UNSTRUCTURED,
        {
            Edge: (
                im.tuple_get(0, im.call("get_domain")("out", im.axis_literal(Edge))),
                im.tuple_get(1, im.call("get_domain")("out", im.axis_literal(Edge))),
            )
        },
    )

    unstructured_domain_E = im.domain(
        common.GridType.UNSTRUCTURED,
        {Edge: (im.tuple_get(0, im.make_tuple(3, 8)), im.tuple_get(1, im.make_tuple(3, 8)))},
    )

    unstructured_domain_V_39 = im.domain(common.GridType.UNSTRUCTURED, {Vertex: (3, 9)})

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

    expected = program_factory(
        params=[im.sym("vertex_values", v_field_type), im.sym("out", e_field_type)],
        declarations=[
            itir.Temporary(id="__tmp_1", domain=unstructured_domain_V_39, dtype=float_type)
        ],
        body=[
            itir.SetAt(
                target=im.ref("__tmp_1"),
                expr=im.as_fieldop("deref", unstructured_domain_V_39)("vertex_values"),
                domain=unstructured_domain_V_39,
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

    run_test_program(testee, expected, sizes, offset_provider)


def test_nested_shift(unstructured_case):
    sizes = {"out": gtx.domain({Edge: (0, 18), Vertex: (3, 7)})}
    unstructured_domain_V = im.domain(
        common.GridType.UNSTRUCTURED,
        {Vertex: (im.tuple_get(0, im.make_tuple(3, 7)), im.tuple_get(1, im.make_tuple(3, 7)))},
    )
    unstructured_domain_get_V = im.domain(
        common.GridType.UNSTRUCTURED,
        {
            Vertex: (
                im.tuple_get(0, im.call("get_domain")("out", im.axis_literal(Vertex))),
                im.tuple_get(1, im.call("get_domain")("out", im.axis_literal(Vertex))),
            )
        },
    )

    unstructured_domain_V_69 = im.domain(common.GridType.UNSTRUCTURED, {Vertex: (6, 9)})

    unstructured_domain_E_1216 = im.domain(common.GridType.UNSTRUCTURED, {Edge: (12, 16)})

    offset_provider = unstructured_case.offset_provider

    testee = program_factory(
        params=[im.sym("vertex_values", v_field_type), im.sym("out", v_field_type)],
        body=[
            itir.SetAt(
                target=im.ref("out"),
                expr=im.as_fieldop(im.lambda_("x")(im.deref(im.shift("V2E", 3)("x"))))(
                    im.as_fieldop(im.lambda_("x")(im.deref(im.shift("E2V", 1)("x"))))(
                        im.as_fieldop("deref")("vertex_values")
                    )
                ),
                domain=unstructured_domain_get_V,
            )
        ],
    )

    expected = program_factory(
        params=[im.sym("vertex_values", v_field_type), im.sym("out", e_field_type)],
        declarations=[
            itir.Temporary(id="__tmp_1", domain=unstructured_domain_E_1216, dtype=float_type),
            itir.Temporary(id="__tmp_2", domain=unstructured_domain_V_69, dtype=float_type),
        ],
        body=[
            itir.SetAt(
                target=im.ref("__tmp_2"),
                expr=im.as_fieldop("deref", unstructured_domain_V_69)("vertex_values"),
                domain=unstructured_domain_V_69,
            ),
            itir.SetAt(
                target=im.ref("__tmp_1"),
                expr=im.as_fieldop(
                    im.lambda_("x")(im.deref(im.shift("E2V", 1)("x"))),
                    unstructured_domain_E_1216,
                )("__tmp_2"),
                domain=unstructured_domain_E_1216,
            ),
            itir.SetAt(
                target=im.ref("out"),
                expr=im.as_fieldop(
                    im.lambda_("x")(im.deref(im.shift("V2E", 3)("x"))), unstructured_domain_V
                )("__tmp_1"),
                domain=unstructured_domain_V,
            ),
        ],
    )

    run_test_program(testee, expected, sizes, offset_provider)


def test_trivial_cartesian():
    grid = simple_cartesian_grid()
    offset_provider = {"Ioff": grid.offset_provider["Ioff"]}
    sizes = {"out": gtx.domain({IDim: (2, 7)})}

    cartesian_domain = im.domain(
        common.GridType.CARTESIAN,
        {IDim: (im.tuple_get(0, im.make_tuple(2, 7)), im.tuple_get(1, im.make_tuple(2, 7)))},
    )
    cartesian_domain_get = im.domain(
        common.GridType.CARTESIAN,
        {
            IDim: (
                im.tuple_get(0, im.call("get_domain")("out", im.axis_literal(IDim))),
                im.tuple_get(1, im.call("get_domain")("out", im.axis_literal(IDim))),
            )
        },
    )

    cartesian_domain_27_p1 = im.domain(
        common.GridType.CARTESIAN,
        {
            IDim: (
                im.plus(im.tuple_get(0, im.make_tuple(2, 7)), 1),
                im.plus(im.tuple_get(1, im.make_tuple(2, 7)), 1),
            )
        },
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

    expected = program_factory(
        params=[im.sym("i_values", i_field_type), im.sym("out", i_field_type)],
        declarations=[
            itir.Temporary(id="__tmp_1", domain=cartesian_domain_27_p1, dtype=float_type)
        ],
        body=[
            itir.SetAt(
                target=im.ref("__tmp_1"),
                expr=im.as_fieldop("deref", cartesian_domain_27_p1)("i_values"),
                domain=cartesian_domain_27_p1,
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

    run_test_program(testee, expected, sizes, offset_provider)


def test_trivial_cartesian_forward():
    grid = simple_cartesian_grid()
    offset_provider = {"Ioff": grid.offset_provider["Ioff"]}
    sizes = {"out": gtx.domain({IDim: (2, 7)})}

    cartesian_domain_get = im.domain(
        common.GridType.CARTESIAN,
        {
            IDim: (
                im.minus(im.tuple_get(0, im.call("get_domain")("out", im.axis_literal(IDim))), 4),
                im.minus(im.tuple_get(1, im.call("get_domain")("out", im.axis_literal(IDim))), 4),
            )
        },
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

    cartesian_domain_m2 = im.domain(
        common.GridType.CARTESIAN,
        {
            IDim: (
                im.minus(im.tuple_get(0, im.make_tuple(2, 7)), 2),
                im.minus(im.tuple_get(1, im.make_tuple(2, 7)), 2),
            )
        },
    )

    cartesian_domain_m4 = im.domain(
        common.GridType.CARTESIAN,
        {
            IDim: (
                im.minus(im.tuple_get(0, im.make_tuple(2, 7)), 4),
                im.minus(im.tuple_get(1, im.make_tuple(2, 7)), 4),
            )
        },
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

    run_test_program(testee, expected, sizes, offset_provider)
