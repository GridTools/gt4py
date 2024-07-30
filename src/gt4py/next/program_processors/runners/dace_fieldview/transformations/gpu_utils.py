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
    try_removing_trivial_maps: bool = True,
    use_gpu_storage: bool = True,
    gpu_block_size: Optional[Sequence[int | str] | str] = None,
    validate: bool = True,
    validate_all: bool = False,
    **kwargs: Any,
) -> dace.SDFG:
    """Transform an SDFG into a GPU SDFG.

    The transformation expects a rather optimized SDFG and turn it into an SDFG
    capable of running on the GPU.
    The function performs the following steps:
    - If requested, modify the storage location of the non transient arrays such
        that they reside in GPU memory.
    - Call the normal GPU transform function followed by simplify.
    - If requested try to remove trivial kernels.
    - If specified, set the `gpu_block_size` parameters of the Maps to the given value.

    Args:
        sdfg: The SDFG that should be processed.
        try_removing_trivial_maps: Try to get rid of trivial maps by incorporating them.
        use_gpu_storage: Assume that the non global memory is already on the GPU.
        gpu_block_size: Set to true when the SDFG array arguments are already allocated
            on GPU global memory. This will avoid the data copy from host to GPU memory.

    Notes:
        The function might modify the order of the iteration variables of some
        maps and fuse other Maps.

    Todo:
        - Solve the fusing problem.
        - Currently only one block size for all maps is given, add more options.
    """

    # You need guru level or above to use these arguments.
    gpu_launch_factor: Optional[int] = kwargs.get("gpu_launch_factor", None)
    gpu_launch_bounds: Optional[int] = kwargs.get("gpu_launch_bounds", None)

    # Turn all global arrays (which we identify as input) into GPU memory.
    #  This way the GPU transformation will not create this copying stuff.
    if use_gpu_storage:
        for desc in sdfg.arrays.values():
            if isinstance(desc, dace.data.Array) and not desc.transient:
                desc.storage = dace.dtypes.StorageType.GPU_Global

    # Now turn it into a GPU SDFG
    sdfg.apply_gpu_transformations(
        validate=validate,
        validate_all=validate_all,
        simplify=False,
    )
    # The documentation recommends to run simplify afterwards
    gtx_transformations.gt_simplify(sdfg)

    if try_removing_trivial_maps:
        # Because of DaCe's design a Tasklet can not exist outside a Map in a GPU SDFG.
        #  The GPU transformation will thus add trivial maps around them, which
        #  translate to a kernel launch. Our current solution is to promote them and
        #  then fuse it.
        # NOTE: The current implementation has a flaw, because promotion and fusion
        #   are two different steps, this is is inefficient. There are some problems
        #   because the mapped Tasklet might not be fusable at all. However, the real
        #   problem is, that Map fusion does not guarantee a certain order of Map
        #   variables. Currently this is not a problem because of the way it is
        #   implemented.
        # TODO(phimuell): Fix the issue described above.
        # TODO(phimuell): Maybe we should fuse trivial GPU maps before we do anything.
        sdfg.apply_transformations_once_everywhere(
            gtx_transformations.SerialMapPromoterGPU(),
            validate=False,
            validate_all=False,
        )
        sdfg.apply_transformations_repeated(
            gtx_transformations.SerialMapFusion(
                only_toplevel_maps=True,
            ),
            validate=validate,
            validate_all=validate_all,
        )

    # Set the GPU block size if it is known.
    if gpu_block_size is not None:
        gt_set_gpu_blocksize(
            sdfg=sdfg,
            gpu_block_size=gpu_block_size,
            gpu_launch_bounds=gpu_launch_bounds,
            gpu_launch_factor=gpu_launch_factor,
        )

    return sdfg


def gt_set_gpu_blocksize(
    sdfg: dace.SDFG,
    gpu_block_size: Optional[Sequence[int | str] | str],
    gpu_launch_bounds: Optional[int | str] = None,
    gpu_launch_factor: Optional[int] = None,
) -> Any:
    """Set the block sizes of GPU Maps.

    Args:
        sdfg: The SDFG to process.
        gpu_block_size: The block size to use.
        gpu_launch_bounds: The launch bounds to use.
        gpu_launch_factor: The launch factor to use.
    """
    xform = GPUSetBlockSize(
        block_size=gpu_block_size,
        launch_bounds=gpu_launch_bounds,
        launch_factor=gpu_launch_factor,
    )
    return sdfg.apply_transformations_once_everywhere([xform])


def _gpu_block_parser(
    self: "GPUSetBlockSize",
    val: Any,
) -> None:
    """Used by the setter of `GPUSetBlockSize.block_size`."""
    org_val = val
    if isinstance(val, tuple):
        pass
    elif isinstance(val, list):
        val = tuple(val)
    elif isinstance(val, str):
        val = tuple(x.strip() for x in val.split(","))
    else:
        raise TypeError(
            f"Does not know how to transform '{type(val).__name__}' into a proper GPU block size."
        )
    if len(val) == 1:
        val = (*val, 1, 1)
    elif len(val) == 2:
        val = (*val, 1)
    elif len(val) != 3:
        raise ValueError(f"Can not parse block size '{org_val}': wrong length")
    try:
        val = [int(x) for x in val]
    except ValueError:
        raise TypeError(
            f"Currently only block sizes convertible to int are supported, you passed '{val}'."
        ) from None
    self._block_size = val


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

    It is also possible to set the launch bound.

    Args:
        block_size: The block size that should be used.
        launch_bounds: The value for the launch bound that should be used.
        launch_factor: If no `launch_bounds` was given use the number of threads
            in a block multiplied by this number.

    Todo:
        Add the possibility to specify other bounds for 1, 2, or 3 dimensional maps.
    """

    block_size = properties.Property(
        dtype=None,
        allow_none=False,
        default=(32, 1, 1),
        setter=_gpu_block_parser,
        getter=_gpu_block_getter,
        desc="Size of the block size a GPU Map should have.",
    )

    launch_bounds = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="Set the launch bound property of the map.",
    )

    map_entry = transformation.transformation.PatternNode(nodes.MapEntry)

    def __init__(
        self,
        block_size: Sequence[int | str] | str | None = None,
        launch_bounds: int | str | None = None,
        launch_factor: int | None = None,
    ) -> None:
        super().__init__()
        if block_size is not None:
            self.block_size = block_size

        if launch_factor is not None:
            assert launch_bounds is None
            self.launch_bounds = str(
                int(launch_factor) * self.block_size[0] * self.block_size[1] * self.block_size[2]
            )
        elif launch_bounds is None:
            self.launch_bounds = None
        elif isinstance(launch_bounds, (str, int)):
            self.launch_bounds = str(launch_bounds)
        else:
            raise TypeError(
                f"Does not know how to parse '{launch_bounds}' as 'launch_bounds' argument."
            )

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
        """Modify the map as requested."""
        self.map_entry.map.gpu_block_size = self.block_size
        if self.launch_bounds is not None:  # Note empty string has a meaning in DaCe
            self.map_entry.map.gpu_launch_bounds = self.launch_bounds


@properties.make_properties
class SerialMapPromoterGPU(transformation.SingleStateTransformation):
    """Serial Map promoter for empty Maps in case of trivial Maps.

    In CPU mode a Tasklet can be outside of a map, however, this is not
    possible in GPU mode. For this reason DaCe wraps such Tasklets in a
    trivial Map.
    This transformation will look for such Maps and promote them, such
    that they can be fused with downstream maps.

    Note:
        This transformation must be run after the GPU Transformation.

    Todo:
        - The transformation assumes that the upper Map is a trivial Tasklet.
            Which should be the majority of all cases.
        - Combine this transformation such that it can do serial fusion on its own.
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
        from .map_serial_fusion import SerialMapFusion

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
