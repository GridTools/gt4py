# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

from gt4py.next import common
from gt4py.next.iterator import builtins, ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import global_tmps, infer_domain
from gt4py.next.iterator.type_system import inference as type_inference
from gt4py.next.type_system import type_specifications as ts


IDim = common.Dimension(value="IDim")
JDim = common.Dimension(value="JDim")
KDim = common.Dimension(value="KDim", kind=common.DimensionKind.VERTICAL)
index_type = ts.ScalarType(kind=getattr(ts.ScalarKind, builtins.INTEGER_INDEX_BUILTIN.upper()))
float_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
i_field_type = ts.FieldType(dims=[IDim], dtype=float_type)
index_field_type_factory = lambda dim: ts.FieldType(dims=[dim], dtype=index_type)


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


def test_trivial():
    domain = im.domain("cartesian_domain", {IDim: (0, 1)})
    offset_provider = {}
    testee = program_factory(
        params=[im.sym("inp", i_field_type), im.sym("out", i_field_type)],
        body=[
            itir.SetAt(
                target=im.ref("out"),
                expr=im.as_fieldop("deref", domain)(im.as_fieldop("deref", domain)("inp")),
                domain=domain,
            )
        ],
    )
    testee = type_inference.infer(testee, offset_provider_type=offset_provider)
    testee = infer_domain.infer_program(testee, offset_provider=offset_provider)

    expected = program_factory(
        params=[im.sym("inp", i_field_type), im.sym("out", i_field_type)],
        declarations=[itir.Temporary(id="__tmp_1", domain=domain, dtype=float_type)],
        body=[
            itir.SetAt(
                target=im.ref("__tmp_1"), expr=im.as_fieldop("deref", domain)("inp"), domain=domain
            ),
            itir.SetAt(
                target=im.ref("out"), expr=im.as_fieldop("deref", domain)("__tmp_1"), domain=domain
            ),
        ],
    )

    actual = global_tmps.create_global_tmps(testee, offset_provider)
    assert actual == expected


def test_trivial_let():
    domain = im.domain("cartesian_domain", {IDim: (0, 1)})
    offset_provider = {}
    testee = program_factory(
        params=[im.sym("inp", i_field_type), im.sym("out", i_field_type)],
        body=[
            itir.SetAt(
                target=im.ref("out"),
                expr=im.let("tmp", im.as_fieldop("deref", domain)("inp"))(
                    im.as_fieldop("deref", domain)("tmp")
                ),
                domain=domain,
            )
        ],
    )
    testee = type_inference.infer(testee, offset_provider_type=offset_provider)
    testee = infer_domain.infer_program(testee, offset_provider=offset_provider)

    expected = program_factory(
        params=[im.sym("inp", i_field_type), im.sym("out", i_field_type)],
        declarations=[itir.Temporary(id="__tmp_1", domain=domain, dtype=float_type)],
        body=[
            itir.SetAt(
                target=im.ref("__tmp_1"), expr=im.as_fieldop("deref", domain)("inp"), domain=domain
            ),
            itir.SetAt(
                target=im.ref("out"), expr=im.as_fieldop("deref", domain)("__tmp_1"), domain=domain
            ),
        ],
    )

    actual = global_tmps.create_global_tmps(testee, offset_provider)
    assert actual == expected


def test_top_level_if():
    domain = im.domain("cartesian_domain", {IDim: (0, 1)})
    offset_provider = {}
    testee = program_factory(
        params=[
            im.sym("inp1", i_field_type),
            im.sym("inp2", i_field_type),
            im.sym("out", i_field_type),
        ],
        body=[
            itir.SetAt(
                target=im.ref("out"),
                expr=im.if_(
                    True,
                    im.as_fieldop("deref", domain)("inp1"),
                    im.as_fieldop("deref", domain)("inp2"),
                ),
                domain=domain,
            )
        ],
    )
    testee = type_inference.infer(testee, offset_provider_type=offset_provider)
    testee = infer_domain.infer_program(testee, offset_provider=offset_provider)

    expected = program_factory(
        params=[
            im.sym("inp1", i_field_type),
            im.sym("inp2", i_field_type),
            im.sym("out", i_field_type),
        ],
        declarations=[],
        body=[
            itir.IfStmt(
                cond=im.literal_from_value(True),
                true_branch=[
                    itir.SetAt(
                        target=im.ref("out"),
                        expr=im.as_fieldop("deref", domain)("inp1"),
                        domain=domain,
                    )
                ],
                false_branch=[
                    itir.SetAt(
                        target=im.ref("out"),
                        expr=im.as_fieldop("deref", domain)("inp2"),
                        domain=domain,
                    )
                ],
            )
        ],
    )

    actual = global_tmps.create_global_tmps(testee, offset_provider)
    assert actual == expected


def test_nested_if():
    domain = im.domain("cartesian_domain", {IDim: (0, 1)})
    offset_provider = {}
    testee = program_factory(
        params=[
            im.sym("inp1", i_field_type),
            im.sym("inp2", i_field_type),
            im.sym("out", i_field_type),
        ],
        body=[
            itir.SetAt(
                target=im.ref("out"),
                expr=im.as_fieldop("deref", domain)(
                    im.if_(
                        True,
                        im.as_fieldop("deref", domain)("inp1"),
                        im.as_fieldop("deref", domain)("inp2"),
                    )
                ),
                domain=domain,
            )
        ],
    )
    testee = type_inference.infer(testee, offset_provider_type=offset_provider)
    testee = infer_domain.infer_program(testee, offset_provider=offset_provider)

    expected = program_factory(
        params=[
            im.sym("inp1", i_field_type),
            im.sym("inp2", i_field_type),
            im.sym("out", i_field_type),
        ],
        declarations=[itir.Temporary(id="__tmp_1", domain=domain, dtype=float_type)],
        body=[
            itir.IfStmt(
                cond=im.literal_from_value(True),
                true_branch=[
                    itir.SetAt(
                        target=im.ref("__tmp_1"),
                        expr=im.as_fieldop("deref", domain)("inp1"),
                        domain=domain,
                    )
                ],
                false_branch=[
                    itir.SetAt(
                        target=im.ref("__tmp_1"),
                        expr=im.as_fieldop("deref", domain)("inp2"),
                        domain=domain,
                    )
                ],
            ),
            itir.SetAt(
                target=im.ref("out"), expr=im.as_fieldop("deref", domain)("__tmp_1"), domain=domain
            ),
        ],
    )

    actual = global_tmps.create_global_tmps(testee, offset_provider)
    assert actual == expected


def test_tuple_different_domain():
    domain1 = im.domain("cartesian_domain", {IDim: (0, 1)})
    domain2 = im.domain("cartesian_domain", {IDim: (0, 2)})
    offset_provider = {"I": IDim}
    testee = program_factory(
        params=[
            im.sym("cond", ts.ScalarType(kind=ts.ScalarKind.BOOL)),
            im.sym("inp1", i_field_type),
            im.sym("inp2", i_field_type),
            im.sym("out", i_field_type),
        ],
        body=[
            itir.SetAt(
                # val = if_(cond, make_tuple(as_fieldop(...), as_fieldop(...)), make_tuple())
                # val1 = if_(cond, as_fieldop(...), as_fieldop(...))
                # val2 = if_(cond, as_fieldop(...), as_fieldop(...))
                # materialize_into(out, make_tuple(as_fieldop(...), ...))
                target=im.ref("out"),
                expr=im.let(
                    "val",
                    im.if_(
                        "cond", im.make_tuple("inp1", "inp2"), im.make_tuple("inp2", "inp1")
                    ),  # todo: fix, the domain is strange
                )(
                    im.as_fieldop(
                        im.lambda_("it1", "it2")(
                            im.plus(im.deref("it1"), im.deref(im.shift("I", 1)("it2")))
                        )
                    )(im.tuple_get(0, "val"), im.tuple_get(1, "val"))
                ),
                domain=domain1,
            )
        ],
    )
    testee = type_inference.infer(testee, offset_provider_type=offset_provider)
    testee = infer_domain.infer_program(testee, offset_provider=offset_provider)

    # TODO:
    expected = program_factory(
        params=[im.sym("inp", i_field_type), im.sym("out", i_field_type)],
        declarations=[itir.Temporary(id="__tmp_1", domain=domain1, dtype=float_type)],
        body=[
            itir.SetAt(
                target=im.ref("__tmp_1"),
                expr=im.as_fieldop("deref", domain1)("inp"),
                domain=domain1,
            ),
            itir.SetAt(
                target=im.ref("out"),
                expr=im.as_fieldop("deref", domain1)("__tmp_1"),
                domain=domain1,
            ),
        ],
    )

    actual = global_tmps.create_global_tmps(testee, offset_provider)
    assert actual == expected
