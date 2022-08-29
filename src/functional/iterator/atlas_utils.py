# GT4Py New Semantic Model - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.  GT4Py
# New Semantic Model is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or any later version.
# See the LICENSE.txt file at the top-level directory of this distribution for
# a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later


try:
    from atlas4py import IrregularConnectivity
except ImportError:
    IrregularConnectivity = None


class AtlasTable:
    def __init__(self, atlas_connectivity) -> None:
        self.atlas_connectivity = atlas_connectivity

    def __getitem__(self, indices):
        primary_index = indices[0]
        neigh_index = indices[1]
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
    def shape(self):
        return (self.atlas_connectivity.rows, self.atlas_connectivity.maxcols)
