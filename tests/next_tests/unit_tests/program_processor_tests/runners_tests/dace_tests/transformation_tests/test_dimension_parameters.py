# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Dimension-valued options of the DaCe transformations accept classes, refs and map variables."""

from __future__ import annotations

import pytest

import gt4py.next as gtx
from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace import (
    lowering as gtx_dace_lowering,
    transformations as gtx_transformations,
)


class IDim(gtx.CartesianAxisIndex): ...


class KDim(gtx.CartesianAxisIndex, kind=gtx.DimensionKind.VERTICAL): ...


_I_VAR = gtx_dace_lowering.get_map_variable(IDim)
_K_VAR = gtx_dace_lowering.get_map_variable(KDim)


@pytest.mark.parametrize(
    "param", [IDim, gtx_common.dimension_ref(IDim), _I_VAR], ids=["class", "ref", "map_variable"]
)
def test_loop_blocking_parameter(param):
    # a ref used to fall through as-is, silently disabling blocking
    assert gtx_transformations.LoopBlocking(blocking_parameters=[param]).blocking_parameters == [
        _I_VAR
    ]
    assert gtx_transformations.LoopBlocking(blocking_parameters=param).blocking_parameters == [
        _I_VAR
    ]


@pytest.mark.parametrize(
    "param", [KDim, gtx_common.dimension_ref(KDim), _K_VAR], ids=["class", "ref", "map_variable"]
)
def test_map_iteration_order_parameter(param):
    # a single ref used not to be wrapped in a list
    assert gtx_transformations.MapIterationOrder(unit_strides_dims=param).unit_strides_dims == [
        _K_VAR
    ]
    assert gtx_transformations.MapIterationOrder(
        unit_strides_dims=[IDim, param]
    ).unit_strides_dims == [_I_VAR, _K_VAR]
