# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.next import common, utils
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.unroll_reduce import UnrollReduce, _get_partial_offset_tags
from gt4py.next.type_system import type_specifications as ts


def dummy_connectivity_type(max_neighbors: int, has_skip_values: bool):
    return common.NeighborConnectivityType(
        domain=[common.Dimension("dummy_origin"), common.Dimension("dummy_neighbor")],
        codomain=common.Dimension("dummy_codomain"),
        skip_value=common._DEFAULT_SKIP_VALUE if has_skip_values else None,
        dtype=None,
        max_neighbors=max_neighbors,
    )


def _list_type(dim: str) -> ts.ListType:
    return ts.ListType(
        element_type=ts.DataType(),
        offset_type=common.Dimension(value=dim, kind=common.DimensionKind.LOCAL),
    )


def typed_neighbors(dim: str, arg: str | ir.Expr) -> ir.FunCall:
    neighbors = im.neighbors(dim, arg)
    neighbors.type = _list_type(dim)
    return neighbors


@pytest.fixture(params=[True, False])
def has_skip_values(request):
    return request.param


@pytest.fixture
def basic_reduction():
    return im.reduce("foo", 0.0)(typed_neighbors("Dim", "x"))


@pytest.fixture
def reduction_with_shift_on_second_arg():
    const_list = im.call("make_const_list")(42)
    const_list.type = _list_type("Dim")
    return im.reduce("foo", 0.0)(const_list, typed_neighbors("Dim", "y"))


@pytest.fixture
def reduction_with_incompatible_shifts():
    return im.reduce("foo", 0.0)(typed_neighbors("Dim", "x"), typed_neighbors("Dim2", "y"))


@pytest.fixture
def reduction_with_irrelevant_full_shift():
    return im.reduce("foo", 0.0)(
        typed_neighbors("Dim", im.shift("IrrelevantDim", 0)("x")), typed_neighbors("Dim", "y")
    )


@pytest.fixture
def reduction_if():
    if_expr = im.if_(True, typed_neighbors("Dim", "x"), "y")
    if_expr.type = _list_type("Dim")
    return im.reduce("foo", 0.0)(if_expr)


@pytest.mark.parametrize(
    "reduction",
    [
        "basic_reduction",
        "reduction_with_irrelevant_full_shift",
        "reduction_with_shift_on_second_arg",
        "reduction_if",
    ],
)
def test_get_partial_offsets(reduction, request):
    partial_offsets = _get_partial_offset_tags(request.getfixturevalue(reduction).args)

    assert set(partial_offsets) == {"Dim"}


def _expected(red, max_neighbors, has_skip_values, shifted_arg=0):
    acc, offset, step = "_acc_0", "_i_0", "_step_0"

    red_fun, red_init = red.fun.args

    elements = [im.list_get(offset, arg) for arg in red.args]

    step_expr = im.call(red_fun)(acc, *elements)
    if has_skip_values:
        neighbors_offset = red.args[shifted_arg].args[0]
        neighbors_it = red.args[shifted_arg].args[1]
        can_deref = im.can_deref(im.shift(neighbors_offset, offset)(neighbors_it))

        step_expr = im.if_(can_deref, step_expr, acc)
    step_fun = im.lambda_(acc, offset)(step_expr)

    step_app = red_init
    for i in range(max_neighbors):
        step_app = im.call(step)(step_app, ir.OffsetLiteral(value=i))

    return im.let(step, step_fun)(step_app)


def test_basic(basic_reduction, has_skip_values, uids: utils.IDGeneratorPool):
    expected = _expected(basic_reduction, 3, has_skip_values)

    offset_provider_type = {
        "Dim": dummy_connectivity_type(max_neighbors=3, has_skip_values=has_skip_values)
    }
    actual = UnrollReduce.apply(
        basic_reduction, offset_provider_type=offset_provider_type, uids=uids
    )
    assert actual == expected


def test_reduction_with_shift_on_second_arg(
    reduction_with_shift_on_second_arg, has_skip_values, uids
):
    expected = _expected(reduction_with_shift_on_second_arg, 1, has_skip_values, 1)

    offset_provider_type = {
        "Dim": dummy_connectivity_type(max_neighbors=1, has_skip_values=has_skip_values)
    }
    actual = UnrollReduce.apply(
        reduction_with_shift_on_second_arg, offset_provider_type=offset_provider_type, uids=uids
    )
    assert actual == expected


def test_reduction_with_if(reduction_if, uids: utils.IDGeneratorPool):
    expected = _expected(reduction_if, 2, False)

    offset_provider_type = {"Dim": dummy_connectivity_type(max_neighbors=2, has_skip_values=False)}
    actual = UnrollReduce.apply(reduction_if, offset_provider_type=offset_provider_type, uids=uids)
    assert actual == expected


def test_reduction_with_irrelevant_full_shift(
    reduction_with_irrelevant_full_shift, uids: utils.IDGeneratorPool
):
    expected = _expected(reduction_with_irrelevant_full_shift, 3, False)

    offset_provider_type = {
        "Dim": dummy_connectivity_type(max_neighbors=3, has_skip_values=False),
        "IrrelevantDim": dummy_connectivity_type(
            max_neighbors=1, has_skip_values=True
        ),  # different max_neighbors and skip value to trigger error
    }
    actual = UnrollReduce.apply(
        reduction_with_irrelevant_full_shift, offset_provider_type=offset_provider_type, uids=uids
    )
    assert actual == expected


@pytest.mark.parametrize(
    "offset_provider_type",
    [
        {
            "Dim": dummy_connectivity_type(max_neighbors=3, has_skip_values=False),
            "Dim2": dummy_connectivity_type(max_neighbors=2, has_skip_values=False),
        },
        {
            "Dim": dummy_connectivity_type(max_neighbors=3, has_skip_values=False),
            "Dim2": dummy_connectivity_type(max_neighbors=3, has_skip_values=True),
        },
        {
            "Dim": dummy_connectivity_type(max_neighbors=3, has_skip_values=False),
            "Dim2": dummy_connectivity_type(max_neighbors=2, has_skip_values=True),
        },
    ],
)
def test_reduction_with_incompatible_shifts(
    reduction_with_incompatible_shifts, offset_provider_type, uids
):
    with pytest.raises(RuntimeError, match="incompatible"):
        UnrollReduce.apply(
            reduction_with_incompatible_shifts, offset_provider_type=offset_provider_type, uids=uids
        )
