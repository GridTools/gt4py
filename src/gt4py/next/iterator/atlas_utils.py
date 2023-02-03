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

try:
    from atlas4py import IrregularConnectivity  # type: ignore[import]
except ImportError:
    IrregularConnectivity = None


# TODO(tehrengruber): make this a proper Connectivity instead of faking a numpy array
class AtlasTable:
    def __init__(self, atlas_connectivity) -> None:
        self.atlas_connectivity = atlas_connectivity

    def __getitem__(self, indices):
        primary_index, neigh_index = indices
        if isinstance(self.atlas_connectivity, IrregularConnectivity):
            if neigh_index < self.atlas_connectivity.cols(primary_index):
                return self.atlas_connectivity[primary_index, neigh_index]
            else:
                return None
        else:
            if neigh_index < 2:
                return self.atlas_connectivity[primary_index, neigh_index]
            else:
                raise AssertionError()

    @property
    def dtype(self):
        assert self.atlas_connectivity.rows > 0
        return type(self[0, 0])

    @property
    def shape(self):
        return (self.atlas_connectivity.rows, self.atlas_connectivity.maxcols)

    def max(self):  # noqa: A003
        maximum = -1
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                v = self[i, j]
                if v is not None:
                    maximum = max(maximum, v)
        return maximum
