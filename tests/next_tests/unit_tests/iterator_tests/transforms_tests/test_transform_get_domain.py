# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Dict, Union

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
from gt4py.next import Domain, common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.transform_get_domain import TransformGetDomain


def program_factory(
    params: list[str],
    body: list[itir.SetAt],
) -> itir.Program:
    return itir.Program(
        id="testee",
        function_definitions=[],
        params=[im.sym(par) for par in params],
        declarations=[],
        body=body,
    )


def setat_factory(
    domain: common.Domain,
    target: str,
) -> itir.SetAt:
    return itir.SetAt(
        expr=im.as_fieldop(im.ref("deref"))(im.ref("inp")),
        domain=domain,
        target=im.ref(target),
    )


def run_test_program(
    params: list[str],
    sizes: Dict[str, common.Domain],
    target: str,
    domain: itir.Expr,
    domain_get: itir.Expr,
) -> None:
    testee = program_factory(
        params=params,
        body=[setat_factory(domain=domain_get, target=im.ref(target))],
    )
    expected = program_factory(
        params=params,
        body=[setat_factory(domain=domain, target=im.ref(target))],
    )
    actual = TransformGetDomain.apply(testee, sizes=sizes)
    assert actual == expected


def construct_domains(
    domain_resolved: Domain, symbol_name: str, type: Union[common.GridType, str]
) -> tuple[itir.FunCall, itir.FunCall]:
    ranges_get = {}
    ragnes_resolved = {}

    for dim, r in zip(domain_resolved.dims, domain_resolved.ranges):
        get_call = im.call("get_domain")(symbol_name, im.axis_literal(dim))
        ranges_get[dim] = (im.tuple_get(0, get_call), im.tuple_get(1, get_call))
        bounds = im.make_tuple(r.start, r.stop)
        ragnes_resolved[dim] = (im.tuple_get(0, bounds), im.tuple_get(1, bounds))

    return im.domain(type, ragnes_resolved), im.domain(type, ranges_get)


def test_get_domain():
    sizes = {"out": gtx.domain({Vertex: (0, 10), KDim: (0, 20)})}
    unstructured_domain, unstructured_domain_get = construct_domains(
        sizes["out"], "out", "unstructured_domain"
    )

    run_test_program(["inp", "out"], sizes, "out", unstructured_domain, unstructured_domain_get)


def test_get_domain_inside_as_fieldop():
    sizes = {"out": gtx.domain({Vertex: (0, 10), KDim: (0, 20)})}
    unstructured_domain, unstructured_domain_get = construct_domains(
        sizes["out"], "out", "unstructured_domain"
    )

    testee = program_factory(
        params=["inp", "out"],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(im.ref("deref"), unstructured_domain_get)(im.ref("inp")),
                domain=unstructured_domain_get,
                target=im.ref("out"),
            ),
        ],
    )

    expected = program_factory(
        params=["inp", "out"],
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

    unstructured_domain_get = im.domain(
        common.GridType.UNSTRUCTURED,
        {
            Vertex: (
                im.tuple_get(
                    0, im.call("get_domain")(im.tuple_get(0, "out"), im.axis_literal(Vertex))
                ),
                im.tuple_get(
                    1, im.call("get_domain")(im.tuple_get(0, "out"), im.axis_literal(Vertex))
                ),
            )
        },
    )
    unstructured_domain = im.domain(
        common.GridType.UNSTRUCTURED,
        {Vertex: (im.tuple_get(0, im.make_tuple(0, 5)), im.tuple_get(1, im.make_tuple(0, 5)))},
    )

    run_test_program(["inp", "out"], sizes, "out", unstructured_domain, unstructured_domain_get)


def test_get_domain_nested_tuples():
    sizes = {"a": gtx.domain({KDim: (0, 3)})}

    t0 = im.make_tuple("a", "b")
    t1 = im.make_tuple("c", "d")
    tup = im.make_tuple(im.tuple_get(0, t0), im.tuple_get(1, t1))

    unstructured_domain_get = im.domain(
        common.GridType.UNSTRUCTURED,
        {
            KDim: (
                im.tuple_get(0, im.call("get_domain")(im.tuple_get(0, tup), im.axis_literal(KDim))),
                im.tuple_get(1, im.call("get_domain")(im.tuple_get(0, tup), im.axis_literal(KDim))),
            )
        },
    )

    unstructured_domain = im.domain(
        common.GridType.UNSTRUCTURED,
        {KDim: (im.tuple_get(0, im.make_tuple(0, 3)), im.tuple_get(1, im.make_tuple(0, 3)))},
    )

    run_test_program(
        ["inp", "a", "b", "c", "d"], sizes, "a", unstructured_domain, unstructured_domain_get
    )
