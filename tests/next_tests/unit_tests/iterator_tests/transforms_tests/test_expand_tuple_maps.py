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
from gt4py.next.iterator.transforms.collapse_tuple import CollapseTuple
from gt4py.next.iterator.transforms.expand_tuple_maps import ExpandTupleMaps
from gt4py.next.type_system import type_specifications as ts


IDim = common.Dimension("IDim")
T = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
i_field = ts.FieldType(dims=[IDim], dtype=T)
i_tuple_field = ts.TupleType(types=[i_field, i_field])


def _apply(expr: itir.Expr) -> itir.Expr:
    return ExpandTupleMaps.apply(expr, uids=utils.IDGeneratorPool(), offset_provider_type={})


def _apply_and_collapse(expr: itir.Expr) -> itir.Expr:
    """Expand and then run the regular `CollapseTuple` pass, as happens in the pipeline."""
    uids = utils.IDGeneratorPool()
    result = ExpandTupleMaps.apply(expr, uids=uids, offset_provider_type={})
    return CollapseTuple.apply(
        result, within_stencil=False, allow_undeclared_symbols=True, uids=uids
    )


def _fun_type(arity: int) -> ts.FunctionType:
    return ts.FunctionType(
        pos_only_args=[i_field] * arity, pos_or_kw_args={}, kw_only_args={}, returns=i_field
    )


@pytest.fixture
def _unary_fun() -> itir.SymRef:
    return im.ref("unary_f", _fun_type(1))


def test_tree_map_tuple_nested(_unary_fun):
    nested = ts.TupleType(types=[i_tuple_field, i_tuple_field])
    result = _apply_and_collapse(
        im.call(im.call("tree_map_tuple")(_unary_fun))(im.ref("t", nested))
    )

    expected = im.make_tuple(
        im.make_tuple(
            im.call(_unary_fun)(im.tuple_get(0, im.tuple_get(0, "t"))),
            im.call(_unary_fun)(im.tuple_get(1, im.tuple_get(0, "t"))),
        ),
        im.make_tuple(
            im.call(_unary_fun)(im.tuple_get(0, im.tuple_get(1, "t"))),
            im.call(_unary_fun)(im.tuple_get(1, im.tuple_get(1, "t"))),
        ),
    )
    assert result == expected


def test_map_tuple_single_arg(_unary_fun):
    result = _apply_and_collapse(
        im.call(im.call("map_tuple")(_unary_fun))(im.ref("t", i_tuple_field))
    )

    expected = im.make_tuple(
        im.call(_unary_fun)(im.tuple_get(0, "t")),
        im.call(_unary_fun)(im.tuple_get(1, "t")),
    )
    assert result == expected


def test_apply_creates_default_uids(_unary_fun):
    expr = im.call(im.call("tree_map_tuple")(_unary_fun))(
        im.make_tuple(im.ref("a", i_field), im.ref("b", i_field))
    )

    result = ExpandTupleMaps.apply(expr, uids=None, offset_provider_type={})

    assert expr.type is None
    assert result == im.let("_etm_0", im.make_tuple("a", "b"))(
        im.make_tuple(
            im.call(_unary_fun)(im.tuple_get(0, "_etm_0")),
            im.call(_unary_fun)(im.tuple_get(1, "_etm_0")),
        )
    )


def test_map_tuple_does_not_recurse():
    nested = ts.TupleType(types=[i_tuple_field, i_tuple_field])
    g = im.ref(
        "tuple_f",
        ts.FunctionType(
            pos_only_args=[i_tuple_field],
            pos_or_kw_args={},
            kw_only_args={},
            returns=i_tuple_field,
        ),
    )
    # Raw `_apply` (no `CollapseTuple`): running collapse here would inline `g` and thereby hide
    # the property under test, namely that `map_tuple` applies `g` to whole top-level elements
    # without recursing into them.
    result = _apply(im.call(im.call("map_tuple")(g))(im.ref("t", nested)))

    expected = im.let("_etm_0", im.ref("t", nested))(
        im.make_tuple(
            im.call(g)(im.tuple_get(0, "_etm_0")),
            im.call(g)(im.tuple_get(1, "_etm_0")),
        )
    )
    assert result == expected


def test_non_trivial_arg_is_let_bound(_unary_fun):
    """A non-trivial (potentially expensive) tuple argument is let-bound so it is evaluated
    once and not duplicated across the leaf projections."""
    # `f(t)` is a non-trivial expression returning a tuple. `f` is an opaque function symbol
    # (not an inlinable lambda), so `CollapseTuple` cannot inline it and the `let` survives.
    f = im.ref(
        "f",
        ts.FunctionType(
            pos_only_args=[i_tuple_field],
            pos_or_kw_args={},
            kw_only_args={},
            returns=i_tuple_field,
        ),
    )
    result = _apply(
        im.call(im.call("tree_map_tuple")(_unary_fun))(im.call(f)(im.ref("t", i_tuple_field)))
    )

    expected = im.let("_etm_0", im.call(f)("t"))(
        im.make_tuple(
            im.call(_unary_fun)(im.tuple_get(0, "_etm_0")),
            im.call(_unary_fun)(im.tuple_get(1, "_etm_0")),
        )
    )
    assert result == expected
