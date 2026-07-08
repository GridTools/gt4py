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
from gt4py.next.iterator.transforms.unroll_tuple_maps import UnrollTupleMaps
from gt4py.next.type_system import type_specifications as ts


IDim = common.Dimension("IDim")
T = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
i_field = ts.FieldType(dims=[IDim], dtype=T)
i_tuple_field = ts.TupleType(types=[i_field, i_field])


def _apply(expr: itir.Expr) -> itir.Expr:
    return UnrollTupleMaps.apply(expr, uids=utils.IDGeneratorPool(), offset_provider_type={})


def _apply_and_collapse(expr: itir.Expr) -> itir.Expr:
    """Unroll and then run the regular `CollapseTuple` pass, as happens in the pipeline."""
    uids = utils.IDGeneratorPool()
    result = UnrollTupleMaps.apply(expr, uids=uids, offset_provider_type={})
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


@pytest.fixture
def _binary_fun() -> itir.SymRef:
    return im.ref("binary_f", _fun_type(2))


def test_tree_map_tuple_multi_arg(_binary_fun):
    result = _apply_and_collapse(
        im.call(im.call("tree_map_tuple")(_binary_fun))(
            im.ref("a", i_tuple_field), im.ref("b", i_tuple_field)
        )
    )

    expected = im.make_tuple(
        im.call(_binary_fun)(im.tuple_get(0, "a"), im.tuple_get(0, "b")),
        im.call(_binary_fun)(im.tuple_get(1, "a"), im.tuple_get(1, "b")),
    )
    assert result == expected


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


def test_apply_infers_uninferred_expr(_unary_fun):
    expr = im.call(im.call("tree_map_tuple")(_unary_fun))(
        im.make_tuple(im.ref("a", i_field), im.ref("b", i_field))
    )

    result = UnrollTupleMaps.apply(expr, uids=None, offset_provider_type={})

    assert expr.type is None
    # `UnrollTupleMaps` let-binds the argument and leaves the `tuple_get(i, make_tuple(...))`
    # cleanup to the regular `CollapseTuple` pass (see `test_make_tuple_arg_is_collapsed`).
    assert result == im.let("_utm_0", im.make_tuple("a", "b"))(
        im.make_tuple(
            im.call(_unary_fun)(im.tuple_get(0, "_utm_0")),
            im.call(_unary_fun)(im.tuple_get(1, "_utm_0")),
        )
    )


def test_map_tuple_does_not_recurse():
    nested = ts.TupleType(types=[i_tuple_field, i_tuple_field])
    g = im.lambda_("__p")(im.op_as_fieldop("plus")(im.tuple_get(0, "__p"), im.tuple_get(1, "__p")))
    # Raw `_apply` (no `CollapseTuple`): running collapse here would inline `g` and thereby hide
    # the property under test, namely that `map_tuple` applies `g` to whole top-level elements
    # without recursing into them.
    result = _apply(im.call(im.call("map_tuple")(g))(im.ref("t", nested)))

    expected = im.let("_utm_0", im.ref("t", nested))(
        im.make_tuple(
            im.call(g)(im.tuple_get(0, "_utm_0")),
            im.call(g)(im.tuple_get(1, "_utm_0")),
        )
    )
    assert result == expected


def test_make_tuple_arg_is_collapsed(_unary_fun):
    """A `make_tuple` literal argument is fully collapsed by the subsequent `CollapseTuple`
    pass, leaving no residual `tuple_get(make_tuple(...))`."""
    result = _apply_and_collapse(
        im.call(im.call("tree_map_tuple")(_unary_fun))(
            im.make_tuple(im.ref("a", i_field), im.ref("b", i_field))
        )
    )

    expected = im.make_tuple(im.call(_unary_fun)("a"), im.call(_unary_fun)("b"))
    assert result == expected


def test_nested_make_tuple_arg_is_collapsed(_unary_fun):
    """A nested `make_tuple` argument is fully collapsed at every depth by `CollapseTuple`."""
    result = _apply_and_collapse(
        im.call(im.call("tree_map_tuple")(_unary_fun))(
            im.make_tuple(
                im.make_tuple(im.ref("a", i_field), im.ref("b", i_field)),
                im.make_tuple(im.ref("c", i_field), im.ref("d", i_field)),
            )
        )
    )

    expected = im.make_tuple(
        im.make_tuple(im.call(_unary_fun)("a"), im.call(_unary_fun)("b")),
        im.make_tuple(im.call(_unary_fun)("c"), im.call(_unary_fun)("d")),
    )
    assert result == expected


def test_map_tuple_with_make_tuple_arg_is_collapsed(_unary_fun):
    """The `make_tuple` collapse also applies for the `map_tuple` builtin."""
    result = _apply_and_collapse(
        im.call(im.call("map_tuple")(_unary_fun))(
            im.make_tuple(im.ref("a", i_field), im.ref("b", i_field))
        )
    )

    expected = im.make_tuple(im.call(_unary_fun)("a"), im.call(_unary_fun)("b"))
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

    expected = im.let("_utm_0", im.call(f)("t"))(
        im.make_tuple(
            im.call(_unary_fun)(im.tuple_get(0, "_utm_0")),
            im.call(_unary_fun)(im.tuple_get(1, "_utm_0")),
        )
    )
    assert result == expected


def test_unroll_then_collapse_inlines_trivial_arg(_unary_fun):
    """End-to-end: `UnrollTupleMaps` let-binds every argument, and the subsequent `CollapseTuple`
    pass inlines the trivial ones back (here a bare `SymRef`), leaving no `let`."""
    testee = im.call(im.call("tree_map_tuple")(_unary_fun))(im.ref("t", i_tuple_field))

    unrolled = _apply(testee)
    # `UnrollTupleMaps` alone let-binds the trivial argument `t`.
    assert unrolled == im.let("_utm_0", im.ref("t", i_tuple_field))(
        im.make_tuple(
            im.call(_unary_fun)(im.tuple_get(0, "_utm_0")),
            im.call(_unary_fun)(im.tuple_get(1, "_utm_0")),
        )
    )

    # After `CollapseTuple` the trivial `let` is inlined away.
    assert _apply_and_collapse(testee) == im.make_tuple(
        im.call(_unary_fun)(im.tuple_get(0, "t")),
        im.call(_unary_fun)(im.tuple_get(1, "t")),
    )


@pytest.mark.parametrize(
    "lhs_type, rhs_type",
    [
        (ts.TupleType(types=[i_field, i_field, i_field]), i_tuple_field),
        (ts.TupleType(types=[i_field, i_tuple_field]), i_tuple_field),
    ],
)
def test_tree_map_tuple_mismatched_structure_raises_type_error(lhs_type, rhs_type, _binary_fun):
    expr = im.call(im.call("tree_map_tuple")(_binary_fun))(
        im.ref("a", lhs_type), im.ref("b", rhs_type)
    )

    with pytest.raises(TypeError, match=r"same tuple structure"):
        _apply(expr)
