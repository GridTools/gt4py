# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next import common, utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.unroll_tree_map import UnrollTreeMap
from gt4py.next.type_system import type_specifications as ts

IDim = common.Dimension("IDim")
T = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
i_field = ts.FieldType(dims=[IDim], dtype=T)
i_tuple_field = ts.TupleType(types=[i_field, i_field])
i_nested_tuple_field = ts.TupleType(types=[i_tuple_field, i_field])

i_domain = im.call("cartesian_domain")(im.named_range(itir.AxisLiteral(value="IDim"), 0, 1))


def _make_program(
    params: list[itir.Sym], expr: itir.Expr, out_type: ts.TypeSpec = i_field
) -> itir.Program:
    return itir.Program(
        id="testee",
        function_definitions=[],
        params=[*params, im.sym("out", out_type)],
        declarations=[],
        body=[
            itir.SetAt(
                expr=expr,
                domain=i_domain,
                target=im.ref("out", out_type),
            )
        ],
    )


def _neg():
    return im.lambda_("__a")(im.op_as_fieldop("neg")("__a"))


def _plus():
    return im.lambda_("__a", "__b")(im.op_as_fieldop("plus")("__a", "__b"))


def test_multi_arg():
    uids = utils.IDGeneratorPool()
    program = _make_program(
        [im.sym("a", i_tuple_field), im.sym("b", i_tuple_field)],
        im.call(im.call("tree_map")(_plus()))(
            im.ref("a", i_tuple_field), im.ref("b", i_tuple_field)
        ),
        out_type=i_tuple_field,
    )
    result = UnrollTreeMap.apply(program, uids=uids)

    expected = _make_program(
        [im.sym("a", i_tuple_field), im.sym("b", i_tuple_field)],
        im.let(("_utm_0", "a"), ("_utm_1", "b"))(
            im.make_tuple(
                im.call(_plus())(im.tuple_get(0, "_utm_0"), im.tuple_get(0, "_utm_1")),
                im.call(_plus())(im.tuple_get(1, "_utm_0"), im.tuple_get(1, "_utm_1")),
            )
        ),
        out_type=i_tuple_field,
    )
    assert result == expected


def test_nested():
    uids = utils.IDGeneratorPool()
    program = _make_program(
        [im.sym("t", i_nested_tuple_field)],
        im.call(im.call("tree_map")(_neg()))(im.ref("t", i_nested_tuple_field)),
        out_type=i_nested_tuple_field,
    )
    result = UnrollTreeMap.apply(program, uids=uids)

    expected = _make_program(
        [im.sym("t", i_nested_tuple_field)],
        im.let("_utm_0", "t")(
            im.make_tuple(
                im.make_tuple(
                    im.call(_neg())(im.tuple_get(0, im.tuple_get(0, "_utm_0"))),
                    im.call(_neg())(im.tuple_get(1, im.tuple_get(0, "_utm_0"))),
                ),
                im.call(_neg())(im.tuple_get(1, "_utm_0")),
            )
        ),
        out_type=i_nested_tuple_field,
    )
    assert result == expected
