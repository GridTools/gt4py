# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import gt4py.next as gtx
from gt4py.next.ffront.transform_utils import _deduce_grid_type


Dim = gtx.Dimension("Dim")
LocalDim = gtx.Dimension("LocalDim", kind=gtx.DimensionKind.LOCAL)

CartesianOffset = gtx.FieldOffset("CartesianOffset", source=Dim, target=(Dim,))
UnstructuredOffset = gtx.FieldOffset("UnstructuredOffset", source=Dim, target=(Dim, LocalDim))


def test_domain_deduction_cartesian():
    assert _deduce_grid_type(None, {CartesianOffset}) == gtx.GridType.CARTESIAN
    assert _deduce_grid_type(None, {Dim}) == gtx.GridType.CARTESIAN


def test_domain_deduction_unstructured():
    assert _deduce_grid_type(None, {UnstructuredOffset}) == gtx.GridType.UNSTRUCTURED
    assert _deduce_grid_type(None, {LocalDim}) == gtx.GridType.UNSTRUCTURED


def test_domain_complies_with_request_cartesian():
    assert _deduce_grid_type(gtx.GridType.CARTESIAN, {CartesianOffset}) == gtx.GridType.CARTESIAN
    with pytest.raises(ValueError, match="unstructured.*FieldOffset.*found"):
        _deduce_grid_type(gtx.GridType.CARTESIAN, {UnstructuredOffset})
        _deduce_grid_type(gtx.GridType.CARTESIAN, {LocalDim})


def test_domain_complies_with_request_unstructured():
    assert (
        _deduce_grid_type(gtx.GridType.UNSTRUCTURED, {UnstructuredOffset})
        == gtx.GridType.UNSTRUCTURED
    )
    # unstructured is ok, even if we don't have unstructured offsets
    assert (
        _deduce_grid_type(gtx.GridType.UNSTRUCTURED, {CartesianOffset}) == gtx.GridType.UNSTRUCTURED
    )
