# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Operations of the embedded connectivities that read the values of their table."""

from __future__ import annotations

import dataclasses
from types import ModuleType
from typing import Optional

from gt4py._core import definitions as core_defs
from gt4py.next import common


@dataclasses.dataclass(frozen=True)
class ConnectivityOps:
    """
    Operations on the buffer of one connectivity.

    An instance belongs to one connectivity and is never inherited by a copy of it:
    `ArrayOps.connectivity_ops` creates it from the connectivity's buffer, and `restrict`
    returns the one of the restricted buffer. A library whose connectivities know more
    about their buffer than the buffer itself, e.g. the values of a traced table,
    subclasses `ConnectivityOps` and keeps that in its instances.
    """

    xp: ModuleType

    def hyperslice(
        self,
        ndarray: core_defs.NDArrayObject,
        image_range: common.UnitRange,
        skip_value: Optional[core_defs.IntegralScalar] = None,
    ) -> Optional[tuple[slice, ...]]:
        """
        Return the hypercube slice that contains all indices in `ndarray` that are within `image_range`, or `None` if no such hypercube exists.

        If `skip_value` is given, the selected values are ignored. It returns the smallest hypercube.
        A bigger hypercube could be constructed by adding lines that contain only `skip_value`s.

        Example:
            ndarray =  0  1 -1
                       3  4 -1
                      -1 -1 -1
            skip_value = -1

            would currently select the 2x2 range [0,2], [0,2], but could also select the 3x3 range [0,3], [0,3].

        An implementation which cannot read the values of `ndarray` returns all of it. What an
        entry outside of `image_range` reads is then defined by `ArrayOps.gather`.
        """
        xp = self.xp
        select_mask = (ndarray >= image_range.start) & (ndarray < image_range.stop)

        if not xp.any(select_mask):
            return None

        nnz: tuple[core_defs.NDArrayObject, ...] = xp.nonzero(select_mask)

        slices = tuple(
            slice(int(xp.min(dim_nnz_indices)), int(xp.max(dim_nnz_indices)) + 1)
            for dim_nnz_indices in nnz
        )
        hcube = select_mask[tuple(slices)]
        if skip_value is not None:
            ignore_mask = ndarray == skip_value
            hcube = hcube | ignore_mask[tuple(slices)]
        if not xp.all(hcube):
            return None

        return slices

    def may_cache(self, ndarray: core_defs.NDArrayObject) -> bool:
        """Whether a connectivity with the buffer `ndarray` may be kept beyond the current call, e.g. in a cache."""
        return True

    def restrict(
        self, ndarray: core_defs.NDArrayObject, buffer_slice: common.RelativeIndexSequence
    ) -> tuple[core_defs.NDArrayObject, ConnectivityOps]:
        """Return `ndarray[buffer_slice]` and the `ConnectivityOps` of the restricted connectivity."""
        return self.xp.asarray(ndarray[buffer_slice]), self
