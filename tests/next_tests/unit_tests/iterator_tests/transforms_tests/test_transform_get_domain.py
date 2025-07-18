# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from gt4py import next as gtx
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.transform_get_domain import TransformGetDomain

KDim = common.Dimension(value="KDim", kind=common.DimensionKind.VERTICAL)
Vertex = common.Dimension(value="Vertex", kind=common.DimensionKind.HORIZONTAL)


def test_get_domain():
    sizes = {"out": gtx.domain({Vertex: (0, 10), KDim: (0, 20)})}

    unstructured_domain_get = im.call("unstructured_domain")(
        im.call("get_domain")("out", im.axis_literal(Vertex)),
        im.call("get_domain")("out", im.axis_literal(KDim)),
    )
    testee = itir.Program(
        id="test",
        function_definitions=[],
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

    unstructured_domain = im.call("unstructured_domain")(
        im.call("named_range")(im.axis_literal(Vertex), 0, 10),
        im.call("named_range")(im.axis_literal(KDim), 0, 20),
    )
    expected = itir.Program(
        id="test",
        function_definitions=[],
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

    unstructured_domain_get = im.call("unstructured_domain")(
        im.call("get_domain")("out", im.axis_literal(Vertex)),
        im.call("get_domain")("out", im.axis_literal(KDim)),
    )
    testee = itir.Program(
        id="test",
        function_definitions=[],
        params=[im.sym("inp"), im.sym("out")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(im.ref("deref"), unstructured_domain_get)(
                    im.ref("inp")
                ),  # TODO: unstructured_domain_get raises AssertionError in domain_utils.py line 77: assert cpm.is_call_to(named_range, "named_range")
                domain=unstructured_domain_get,
                target=im.ref("out"),
            ),
        ],
    )

    unstructured_domain = im.call("unstructured_domain")(
        im.call("named_range")(im.axis_literal(Vertex), 0, 10),
        im.call("named_range")(im.axis_literal(KDim), 0, 20),
    )
    expected = itir.Program(
        id="test",
        function_definitions=[],
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
    assert actual == expected  # TODO: this test still fails because of the AssertionError


def test_get_domain_tuples():
    sizes = {"out": (gtx.domain({Vertex: (0, 5)}), gtx.domain({Vertex: (0, 7)}))}

    unstructured_domain_get = im.call("unstructured_domain")(
        im.call("get_domain")(im.tuple_get(0, "out"), im.axis_literal(Vertex))
    )

    testee = itir.Program(
        id="test",
        function_definitions=[],
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

    unstructured_domain = im.call("unstructured_domain")(
        im.call("named_range")(im.axis_literal(Vertex), 0, 5),
    )
    expected = itir.Program(
        id="test",
        function_definitions=[],
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
        im.call("get_domain")(im.tuple_get(0, tup), im.axis_literal(KDim))
    )

    testee = itir.Program(
        id="test",
        function_definitions=[],
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

    unstructured_domain = im.call("unstructured_domain")(
        im.call("named_range")(im.axis_literal(KDim), 0, 3),
    )
    expected = itir.Program(
        id="test",
        function_definitions=[],
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
