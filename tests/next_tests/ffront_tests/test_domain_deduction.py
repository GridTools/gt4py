# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

from gt4py.next.common import Dimension, DimensionKind, GridType, GTTypeError
from gt4py.next.ffront.decorator import _deduce_grid_type
from gt4py.next.ffront.fbuiltins import FieldOffset


Dim = Dimension("Dim")
LocalDim = Dimension("LocalDim", kind=DimensionKind.LOCAL)

CartesianOffset = FieldOffset("CartesianOffset", source=Dim, target=(Dim,))
UnstructuredOffset = FieldOffset("UnstructuredOffset", source=Dim, target=(Dim, LocalDim))


def test_domain_deduction_cartesian():
    assert _deduce_grid_type(None, {CartesianOffset}) == GridType.CARTESIAN
    assert _deduce_grid_type(None, {Dim}) == GridType.CARTESIAN


def test_domain_deduction_unstructured():
    assert _deduce_grid_type(None, {UnstructuredOffset}) == GridType.UNSTRUCTURED
    assert _deduce_grid_type(None, {LocalDim}) == GridType.UNSTRUCTURED


def test_domain_complies_with_request_cartesian():
    assert _deduce_grid_type(GridType.CARTESIAN, {CartesianOffset}) == GridType.CARTESIAN
    with pytest.raises(GTTypeError, match="unstructured.*FieldOffset.*found"):
        _deduce_grid_type(GridType.CARTESIAN, {UnstructuredOffset})
        _deduce_grid_type(GridType.CARTESIAN, {LocalDim})


def test_domain_complies_with_request_unstructured():
    assert _deduce_grid_type(GridType.UNSTRUCTURED, {UnstructuredOffset}) == GridType.UNSTRUCTURED
    # unstructured is ok, even if we don't have unstructured offsets
    assert _deduce_grid_type(GridType.UNSTRUCTURED, {CartesianOffset}) == GridType.UNSTRUCTURED
