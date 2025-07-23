# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from next_tests.integration_tests.cases import (
    IDim,
    KDim,
    Vertex,
)

from gt4py import next as gtx
from gt4py.next import Domain
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.transform_get_domain import TransformGetDomain

IOff = gtx.FieldOffset("IOff", source=IDim, target=(IDim,))


def construct_domains(domain_resolved: Domain, symbol_name: str, type: str):
    named_ranges_get, named_ranges_resolved = [], []

    for dim, range_ in zip(domain_resolved.dims, domain_resolved.ranges):
        get_domain_call = im.call("get_domain")(symbol_name, im.axis_literal(dim))
        named_ranges_get.append(
            im.named_range(im.axis_literal(dim), im.tuple_get(0, get_domain_call), im.tuple_get(1, get_domain_call))
        )
        bounds_tuple = im.make_tuple(range_.start, range_.stop)
        named_ranges_resolved.append(
            im.named_range(im.axis_literal(dim), im.tuple_get(0, bounds_tuple), im.tuple_get(1, bounds_tuple))
        )

    return im.call(type)(*named_ranges_resolved), im.call(type)(*named_ranges_get)


def test_get_domain():
    sizes = {"out": gtx.domain({Vertex: (0, 10), KDim: (0, 20)})}
    unstructured_domain, unstructured_domain_get = construct_domains(sizes["out"], "out", "unstructured_domain")

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
    unstructured_domain, unstructured_domain_get = construct_domains(sizes["out"], "out", "unstructured_domain")

    testee = itir.Program(
        id="test",
        function_definitions=[],
        params=[im.sym("inp"), im.sym("out")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(im.ref("deref"), unstructured_domain_get)(
                    im.ref("inp")
                ),
                domain=unstructured_domain_get,
                target=im.ref("out"),
            ),
        ],
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
    assert actual == expected


def test_get_domain_tuples():
    sizes = {"out": (gtx.domain({Vertex: (0, 5)}), gtx.domain({Vertex: (0, 7)}))}

    unstructured_domain_get = im.call("unstructured_domain")(
        im.named_range(im.axis_literal(Vertex),
                       im.tuple_get(0, im.call("get_domain")(im.tuple_get(0, "out"), im.axis_literal(Vertex))),
                       im.tuple_get(1, im.call("get_domain")(im.tuple_get(0, "out"), im.axis_literal(Vertex)))
                       )
    )
    unstructured_domain = im.call("unstructured_domain")(
        im.named_range(im.axis_literal(Vertex), im.tuple_get(0, im.make_tuple(0, 5)),
                       im.tuple_get(1, im.make_tuple(0, 5))),
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
        im.named_range(im.axis_literal(KDim),
                       im.tuple_get(0, im.call("get_domain")(im.tuple_get(0, tup), im.axis_literal(KDim))),
                       im.tuple_get(1, im.call("get_domain")(im.tuple_get(0, tup), im.axis_literal(KDim)))
                       )
    )
    unstructured_domain = im.call("unstructured_domain")(
        im.named_range(im.axis_literal(KDim), im.tuple_get(0, im.make_tuple(0, 3)),
                       im.tuple_get(1, im.make_tuple(0, 3))),
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
