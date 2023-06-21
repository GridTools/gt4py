# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest

import gt4py.next as gtx
from gt4py.next.ffront.decorator import _deduce_grid_type


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
