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

"""Functions for turning an SDFG into a GPU SDFG."""

import copy
from typing import Any, Optional, Sequence, Union

import dace
import numpy as np
from dace import properties, transformation
from dace.sdfg import SDFG, SDFGState, nodes

from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)


__all__ = [
    "SerialMapPromoterGPU",
    "GPUSetBlockSize",
    "gt_gpu_transformation",
    "gt_set_gpu_blocksize",
]


def gt_gpu_transformation(
    sdfg: dace.SDFG,
    promote_serial_maps: bool = True,
    gpu_block_size: Optional[Sequence[int | str] | str] = None,
    validate: bool = True,
    validate_all: bool = False,
) -> dace.SDFG:
    """Transform an SDFG into an GPU SDFG.

    The transformations are done in place.
    The function will roughly do the same:
    - Move all arrays used as input to the GPU.
    - Apply the standard DaCe GPU transformation.
    - Run `gt_simplify()` (recommended by the DaCe documentation).
    - Try to promote trivial maps.
    - If given set the GPU block size.
    """

    # Turn all global arrays (which we identify as input) into GPU memory.
    #  This way the GPU transformation will not create this copying stuff.
    for desc in sdfg.arrays.values():
        if desc.transient:
            continue
        if not isinstance(desc, dace.data.Array):
            continue
        desc.storage = dace.dtypes.StorageType.GPU_Global

    # Now turn it into a GPU SDFG
    sdfg.apply_gpu_transformations(
        validate=validate,
        validate_all=validate_all,
        simplify=False,
    )

    # The documentation recommend to run simplify afterwards
    gtx_transformations.gt_simplify(sdfg)

    # Start to promote the maps.
    if promote_serial_maps:
        sdfg.apply_transformations_repeated(
            [gtx_transformations.SerialMapPromoterGPU()],
            validate=validate,
            validate_all=validate_all,
        )

    # Set the GPU block size if it is known.
    if gpu_block_size is not None:
        gt_set_gpu_blocksize(sdfg, gpu_block_size)

    return sdfg


def gt_set_gpu_blocksize(
    sdfg: dace.SDFG,
    gpu_block_size: Optional[Sequence[int | str] | str],
) -> Any:
    """Set the block sizes of GPU Maps.

    Args:
        sdfg: The SDFG to process.
        gpu_block_size: The block size to use.
    """
    return sdfg.apply_transformations_once_everywhere([GPUSetBlockSize(block_size=gpu_block_size)])


def _gpu_block_parser(
    self: "GPUSetBlockSize",
    val: Any,
) -> None:
    """Used by the setter ob `GPUSetBlockSize.block_size`."""
    org_val = val
    if isinstance(val, tuple):
        pass
    elif isinstance(val, list):
        val = tuple(val)
    elif isinstance(val, str):
        val = tuple(x.replace(" ", "") for x in val.split(","))
    else:
        raise TypeError(f"Does not know how to transform '{type(val).__name__}' into a proper GPU block size.")
    if len(val) == 1:
        val = (*val, 1, 1)
    elif len(val) == 2:
        val = (*val, 1)
    elif len(val) != 3:
        raise ValueError(f"Can not parse block size '{org_val}': wrong length")
    assert all(isinstance(x, (str, int, np.integer)) for x in val)
    self._block_size = [int(x) for x in val]


def _gpu_block_getter(
    self: "GPUSetBlockSize",
) -> tuple[int, int, int]:
    """Used as getter in the `GPUSetBlockSize.block_size` property."""
    assert isinstance(self._block_size, (tuple, list)) and len(self._block_size) == 3
    assert all(isinstance(x, int) for x in self._block_size)
    return tuple(self._block_size)


@properties.make_properties
class GPUSetBlockSize(transformation.SingleStateTransformation):
    """Sets the GPU block size on GPU Maps.

    Todo:
        Depending on the number of dimensions of a map, there should be different sources.
    """

    block_size = properties.Property(
        dtype=None,
        allow_none=False,
        default=(32,1,1),
        setter=_gpu_block_parser,
        getter=_gpu_block_getter,
        desc="Size of the block size a GPU Map should have.",
    )

    map_entry = transformation.transformation.PatternNode(nodes.MapEntry)

    def __int__(
        self,
        block_size: Sequence[int | str] | str | None = None,
    ) -> None:
        super().__init__()
        if block_size is not None:
            self.block_size = block_size

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.map_entry)]

    def can_be_applied(
        self,
        graph: Union[SDFGState, SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        """Test if the block size can be set.

        The function tests:
        - If the block size of the map is already set.
        - If the map is at global scope.
        - If if the schedule of the map is correct.
        """

        scope = graph.scope_dict()
        if scope[self.map_entry] is not None:
            return False
        if self.map_entry.map.schedule not in dace.dtypes.GPU_SCHEDULES:
            return False
        if self.map_entry.map.gpu_block_size is not None:
            return False
        return True

    def apply(
        self,
        graph: Union[SDFGState, SDFG],
        sdfg: SDFG,
    ) -> None:
        """Sets the block size."""
        self.map_entry.map.gpu_block_size = self.block_size


@properties.make_properties
class SerialMapPromoterGPU(transformation.SingleStateTransformation):
    """Serial Map promoter for empty Maps in case of trivial Maps.

    In CPU mode a Tasklet can be outside of a map, however, this is not
    possible in CPU mode. For this reason DaCe wraps every such Tasklet
    in a trivial Map.
    This function will look for such Maps and promote them, such that they
    can be fused with downstream maps.

    Note:
        This transformation must be run after the GPU Transformation.
    """

    # Pattern Matching
    map_exit1 = transformation.transformation.PatternNode(nodes.MapExit)
    access_node = transformation.transformation.PatternNode(nodes.AccessNode)
    map_entry2 = transformation.transformation.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.map_exit1, cls.access_node, cls.map_entry2)]

    def can_be_applied(
        self,
        graph: Union[SDFGState, SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        """Tests if the promotion is possible.

        The function tests:
        - If the top map is a trivial map.
        - If a valid partition exists that can be fused at all.
        """
        from .map_seriall_fusion import SerialMapFusion

        map_exit_1: nodes.MapExit = self.map_exit1
        map_1: nodes.Map = map_exit_1.map
        map_entry_2: nodes.MapEntry = self.map_entry2

        # Check if the first map is trivial.
        if len(map_1.params) != 1:
            return False
        if map_1.range.num_elements() != 1:
            return False

        # Check if it is a GPU map
        if map_1.schedule not in [
            dace.dtypes.ScheduleType.GPU_Device,
            dace.dtypes.ScheduleType.GPU_Default,
        ]:
            return False

        # Check if the partition exists, if not promotion to fusing is pointless.
        #  TODO(phimuell): Find the proper way of doing it.
        serial_fuser = SerialMapFusion(only_toplevel_maps=True)
        output_partition = serial_fuser.partition_first_outputs(
            state=graph,
            sdfg=sdfg,
            map_exit_1=map_exit_1,
            map_entry_2=map_entry_2,
        )
        if output_partition is None:
            return False

        return True

    def apply(self, graph: Union[SDFGState, SDFG], sdfg: SDFG) -> None:
        """Performs the Map Promoting.

        The function essentially copies the parameters and the ranges from the
        bottom map to the top one.
        """
        map_1: nodes.Map = self.map_exit1.map
        map_2: nodes.Map = self.map_entry2.map

        map_1.params = copy.deepcopy(map_2.params)
        map_1.range = copy.deepcopy(map_2.range)

        return
