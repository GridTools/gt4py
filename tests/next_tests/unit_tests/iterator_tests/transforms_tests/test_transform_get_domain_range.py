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
from gt4py.next.iterator.transforms.transform_get_domain_range import TransformGetDomainRange
from gt4py.next.iterator.transforms.collapse_tuple import CollapseTuple


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
    actual = TransformGetDomainRange.apply(testee, sizes=sizes)
    actual = CollapseTuple.apply(
        actual, enabled_transformations=CollapseTuple.Transformation.COLLAPSE_TUPLE_GET_MAKE_TUPLE
    )
    assert actual == expected


def domain_as_expr(domain: gtx.Domain) -> itir.Expr:
    return im.domain(
        common.GridType.UNSTRUCTURED,
        {d: (r.start, r.stop) for d, r in zip(domain.dims, domain.ranges)},
    )


def test_get_domain():
    sizes = {"out": gtx.domain({Vertex: (0, 10), KDim: (0, 20)})}
    get_domain_expr = im.get_field_domain(common.GridType.UNSTRUCTURED, "out", sizes["out"].dims)

    run_test_program(["inp", "out"], sizes, "out", domain_as_expr(sizes["out"]), get_domain_expr)


def test_get_domain_tuples():
    sizes = {"out": (gtx.domain({Vertex: (0, 5)}), gtx.domain({Vertex: (0, 7)}))}

    get_domain_expr = im.get_field_domain(
        common.GridType.UNSTRUCTURED, im.tuple_get(1, "out"), sizes["out"][1].dims
    )

    run_test_program(["inp", "out"], sizes, "out", domain_as_expr(sizes["out"][1]), get_domain_expr)


def test_get_domain_nested_tuples():
    sizes = {"a": gtx.domain({KDim: (0, 3)})}

    get_domain_expr = im.get_field_domain(
        common.GridType.UNSTRUCTURED,
        im.tuple_get(
            0,
            im.make_tuple(
                im.tuple_get(0, im.make_tuple("a", "b")), im.tuple_get(1, im.make_tuple("c", "d"))
            ),
        ),
        sizes["a"].dims,
    )

    run_test_program(
        ["inp", "a", "b", "c", "d"], sizes, "a", domain_as_expr(sizes["a"]), get_domain_expr
    )
