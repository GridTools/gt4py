# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import typing

import pytest

import gt4py.next as gtx
from gt4py.next.ffront.transform_utils import _deduce_grid_type


class HDim(gtx.DimensionIndex, kind=gtx.DimensionKind.HORIZONTAL): ...


class VDim(gtx.DimensionIndex, kind=gtx.DimensionKind.VERTICAL): ...


class Dim(gtx.DimensionIndex): ...


class LocalDim(gtx.LocalDimensionIndex): ...


class UnstructuredOffset(gtx.NeighborConnectivity[Dim, Dim]):
    Local: typing.TypeAlias = LocalDim


def test_domain_deduction_cartesian():
    assert _deduce_grid_type(None, {Dim}) == gtx.GridType.CARTESIAN
    assert _deduce_grid_type(None, {HDim, VDim}) == gtx.GridType.CARTESIAN


def test_domain_deduction_unstructured():
    assert _deduce_grid_type(None, {UnstructuredOffset}) == gtx.GridType.UNSTRUCTURED
    assert _deduce_grid_type(None, {LocalDim}) == gtx.GridType.UNSTRUCTURED


def test_domain_complies_with_request_cartesian():
    assert _deduce_grid_type(gtx.GridType.CARTESIAN, {Dim}) == gtx.GridType.CARTESIAN
    with pytest.raises(ValueError, match="NeighborConnectivity.*local dimension was found"):
        _deduce_grid_type(gtx.GridType.CARTESIAN, {UnstructuredOffset})
    with pytest.raises(ValueError, match="NeighborConnectivity.*local dimension was found"):
        _deduce_grid_type(gtx.GridType.CARTESIAN, {LocalDim})


def test_domain_complies_with_request_unstructured():
    assert (
        _deduce_grid_type(gtx.GridType.UNSTRUCTURED, {UnstructuredOffset})
        == gtx.GridType.UNSTRUCTURED
    )
    # unstructured is ok, even if we don't have unstructured offsets
    assert _deduce_grid_type(gtx.GridType.UNSTRUCTURED, {Dim}) == gtx.GridType.UNSTRUCTURED
