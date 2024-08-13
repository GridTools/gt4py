# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

try:
    from atlas4py import IrregularConnectivity
except ImportError:
    IrregularConnectivity = None

from gt4py.next import common


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
                return common._DEFAULT_SKIP_VALUE
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

    def max(self):
        maximum = -1
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                v = self[i, j]
                if v is not None:
                    maximum = max(maximum, v)
        return maximum

    def asnumpy(self):
        import numpy as np

        res = np.empty(self.shape, dtype=self.dtype)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                res[i, j] = self[i, j]
        return res
