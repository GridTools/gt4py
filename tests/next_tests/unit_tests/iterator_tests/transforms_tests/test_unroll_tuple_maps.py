# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.next import common, utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.unroll_tuple_maps import UnrollTupleMaps
from gt4py.next.type_system import type_specifications as ts


IDim = common.Dimension("IDim")
T = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
i_field = ts.FieldType(dims=[IDim], dtype=T)
i_tuple_field = ts.TupleType(types=[i_field, i_field])


def _apply(expr: itir.Expr) -> itir.Expr:
    return UnrollTupleMaps.apply(expr, uids=utils.IDGeneratorPool(), offset_provider_type={})


def _neg():
    return im.lambda_("__a")(im.op_as_fieldop("neg")("__a"))


def _plus():
    return im.lambda_("__a", "__b")(im.op_as_fieldop("plus")("__a", "__b"))


def test_tree_map_tuple_multi_arg():
    result = _apply(
        im.call(im.call("tree_map_tuple")(_plus()))(
            im.ref("a", i_tuple_field), im.ref("b", i_tuple_field)
        )
    )

    expected = im.make_tuple(
        im.call(_plus())(im.tuple_get(0, "a"), im.tuple_get(0, "b")),
        im.call(_plus())(im.tuple_get(1, "a"), im.tuple_get(1, "b")),
    )
    assert result == expected


def test_tree_map_tuple_nested():
    nested = ts.TupleType(types=[i_tuple_field, i_tuple_field])
    result = _apply(im.call(im.call("tree_map_tuple")(_neg()))(im.ref("t", nested)))

    expected = im.make_tuple(
        im.make_tuple(
            im.call(_neg())(im.tuple_get(0, im.tuple_get(0, "t"))),
            im.call(_neg())(im.tuple_get(1, im.tuple_get(0, "t"))),
        ),
        im.make_tuple(
            im.call(_neg())(im.tuple_get(0, im.tuple_get(1, "t"))),
            im.call(_neg())(im.tuple_get(1, im.tuple_get(1, "t"))),
        ),
    )
    assert result == expected


def test_map_tuple_single_arg():
    result = _apply(im.call(im.call("map_tuple")(_neg()))(im.ref("t", i_tuple_field)))

    expected = im.make_tuple(
        im.call(_neg())(im.tuple_get(0, "t")),
        im.call(_neg())(im.tuple_get(1, "t")),
    )
    assert result == expected


def test_apply_infers_uninferred_expr():
    expr = im.call(im.call("tree_map_tuple")(_neg()))(
        im.make_tuple(im.ref("a", i_field), im.ref("b", i_field))
    )

    result = UnrollTupleMaps.apply(expr, offset_provider_type={})

    assert expr.type is None
    assert result == im.make_tuple(im.call(_neg())("a"), im.call(_neg())("b"))


def test_map_tuple_does_not_recurse():
    nested = ts.TupleType(types=[i_tuple_field, i_tuple_field])
    g = im.lambda_("__p")(im.op_as_fieldop("plus")(im.tuple_get(0, "__p"), im.tuple_get(1, "__p")))
    result = _apply(im.call(im.call("map_tuple")(g))(im.ref("t", nested)))

    expected = im.make_tuple(
        im.call(g)(im.tuple_get(0, "t")),
        im.call(g)(im.tuple_get(1, "t")),
    )
    assert result == expected


def test_make_tuple_arg_is_collapsed():
    """When the input tuple is a `make_tuple` literal, projection should collapse
    directly to the element (no residual `tuple_get(make_tuple(...))`)."""
    result = _apply(
        im.call(im.call("tree_map_tuple")(_neg()))(
            im.make_tuple(im.ref("a", i_field), im.ref("b", i_field))
        )
    )

    expected = im.make_tuple(im.call(_neg())("a"), im.call(_neg())("b"))
    assert result == expected


def test_nested_make_tuple_arg_is_collapsed():
    """A nested `make_tuple` arg should be fully collapsed at every depth: each
    `tuple_get(i, make_tuple(...))` along the recursion is folded directly."""
    result = _apply(
        im.call(im.call("tree_map_tuple")(_neg()))(
            im.make_tuple(
                im.make_tuple(im.ref("a", i_field), im.ref("b", i_field)),
                im.make_tuple(im.ref("c", i_field), im.ref("d", i_field)),
            )
        )
    )

    expected = im.make_tuple(
        im.make_tuple(im.call(_neg())("a"), im.call(_neg())("b")),
        im.make_tuple(im.call(_neg())("c"), im.call(_neg())("d")),
    )
    assert result == expected


def test_map_tuple_with_make_tuple_arg_is_collapsed():
    """The `make_tuple` short-circuit must also apply for the `map_tuple` builtin."""
    result = _apply(
        im.call(im.call("map_tuple")(_neg()))(
            im.make_tuple(im.ref("a", i_field), im.ref("b", i_field))
        )
    )

    expected = im.make_tuple(im.call(_neg())("a"), im.call(_neg())("b"))
    assert result == expected


def test_non_trivial_arg_is_let_bound():
    """Non-trivial (potentially expensive) tuple expressions must still be
    let-bound to avoid duplicating work across leaf projections."""
    # `f(t)` is a non-trivial expression returning a tuple
    f = im.lambda_("__t")(im.ref("__t", i_tuple_field))
    result = _apply(
        im.call(im.call("tree_map_tuple")(_neg()))(im.call(f)(im.ref("t", i_tuple_field)))
    )

    expected = im.let("_utm_0", im.call(f)("t"))(
        im.make_tuple(
            im.call(_neg())(im.tuple_get(0, "_utm_0")),
            im.call(_neg())(im.tuple_get(1, "_utm_0")),
        )
    )
    assert result == expected


@pytest.mark.parametrize(
    "lhs_type, rhs_type",
    [
        (ts.TupleType(types=[i_field, i_field, i_field]), i_tuple_field),
        (ts.TupleType(types=[i_field, i_tuple_field]), i_tuple_field),
    ],
)
def test_tree_map_tuple_mismatched_structure_raises_type_error(lhs_type, rhs_type):
    expr = im.call(im.call("tree_map_tuple")(_plus()))(im.ref("a", lhs_type), im.ref("b", rhs_type))

    with pytest.raises(TypeError, match=r"same tuple structure"):
        _apply(expr)
