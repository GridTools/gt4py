# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Operations of the embedded connectivities that read the values of their table, per array namespace.

`ConnectivityOps` builds on the `array_ops.ArrayOps` of the buffer's namespace. A library
that needs different behaviour subclasses `ConnectivityOps` and registers the subclass for
its namespace with `register`.
"""

from __future__ import annotations

import dataclasses
from types import ModuleType
from typing import Any, Optional

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.embedded import array_ops
from gt4py.next.embedded.array_ops import ArrayOps


@dataclasses.dataclass(frozen=True, eq=False)
class ConnectivityOps:
    """
    Operations on the buffer of one connectivity.

    An instance belongs to one connectivity and is never inherited by a copy of it: `of`
    creates it from the connectivity's buffer, and `restrict` returns the one of the
    restricted buffer. The `ndarray` its methods take is the buffer of that connectivity.
    A library whose connectivities know more about their buffer than the buffer itself,
    e.g. the values of a traced table, subclasses `ConnectivityOps` and keeps that in its
    instances.
    """

    array_ops: ArrayOps

    @classmethod
    def from_ndarray(cls, ops: ArrayOps, ndarray: core_defs.NDArrayObject) -> ConnectivityOps:
        """The `ConnectivityOps` of a connectivity whose buffer is `ndarray`, of the namespace of `ops`."""
        return cls(ops)

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
        xp = self.array_ops.xp
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
        return self.array_ops.asarray(ndarray[buffer_slice]), self


_types_by_namespace: dict[ModuleType, type[ConnectivityOps]] = {}


def register(xp: ModuleType, ops_type: type[ConnectivityOps]) -> None:
    """
    Use `ops_type` for the connectivities whose buffer has the compat namespace `xp`.

    Connectivities which already have their `ConnectivityOps` keep them.
    """
    _types_by_namespace[xp] = ops_type


def of(ndarray: Any) -> ConnectivityOps:
    """The `ConnectivityOps` of a connectivity whose buffer is `ndarray`; raises `TypeError` if it is not an array."""
    ops = array_ops.of(ndarray)
    return _types_by_namespace.get(ops.xp, ConnectivityOps).from_ndarray(ops, ndarray)
