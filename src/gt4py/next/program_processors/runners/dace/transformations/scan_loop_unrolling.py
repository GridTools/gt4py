# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, ClassVar, Final, Optional, Union

import dace
from dace import properties as dace_properties, transformation as dace_transformation
from dace.sdfg import state as dace_state


@dace_properties.make_properties
class ScanLoopUnrolling(dace_transformation.MultiStateTransformation):
    """
    Unrolls loops that are part of a GPU scan operation.
    """

    _default_unroll_factor: Final[int] = 4
    unroll = dace_properties.Property(
        dtype=bool,
        default=True,
        desc="Whether to unroll GPU scan loops.",
    )
    unroll_factor = dace_properties.Property(
        dtype=int,
        default=_default_unroll_factor,
        desc="The unroll factor to use when unrolling GPU scan loops.",
    )
    scan_loop_region = dace_transformation.PatternNode(dace_state.LoopRegion)
    applied_scans: ClassVar[set[dace_state.LoopRegion]] = set()

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.scan_loop_region)]

    def __init__(
        self,
        unroll: Optional[bool] = None,
        unroll_factor: Optional[int] = None,
    ) -> None:
        super().__init__()
        if unroll is not None:
            self.unroll = unroll
        if unroll_factor is not None and unroll_factor > 0:
            self.unroll = True
            self.unroll_factor = unroll_factor
        elif unroll_factor is not None and unroll_factor == 0:
            self.unroll = True
            self.unroll_factor = 0
        elif unroll_factor is not None and unroll_factor < 0:
            self.unroll = False

    def can_be_applied(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        if self.scan_loop_region in self.applied_scans:
            return False
        if not self.scan_loop_region.label.startswith("scan_"):
            return False
        return self.unroll

    def apply(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
    ) -> None:
        loop_region: dace_state.LoopRegion = self.scan_loop_region
        loop_region.unroll_factor = self.unroll_factor
        loop_region.unroll = True
        self.applied_scans.add(loop_region)
