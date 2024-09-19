# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Functions for turning an SDFG into a GPU SDFG."""

import copy
from typing import Any, Optional, Sequence, Union

import dace
from dace import properties as dace_properties, transformation as dace_transformation
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)


def gt_gpu_transformation(
    sdfg: dace.SDFG,
    try_removing_trivial_maps: bool = True,
    use_gpu_storage: bool = True,
    gpu_block_size: Optional[Sequence[int | str] | str] = None,
    gpu_launch_bounds: Optional[int | str] = None,
    gpu_launch_factor: Optional[int] = None,
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
        use_gpu_storage: Assume that the non global memory is already on the GPU. This
            will avoid the data copy from host to GPU memory.
        gpu_block_size: The size of a thread block on the GPU.
        gpu_launch_bounds: Use this value as `__launch_bounds__` for _all_ GPU Maps.
        gpu_launch_factor: Use the number of threads times this value as `__launch_bounds__`
        validate: Perform validation during the steps.
        validate_all: Perform extensive validation.

    Notes:
        The function might modify the order of the iteration variables of some
        maps.
        In addition it might fuse Maps together that should not be fused. To prevent
        that you should set `try_removing_trivial_maps` to `False`.

    Todo:
        - Solve the fusing problem.
        - Currently only one block size for all maps is given, add more options.
    """
    assert (
        len(kwargs) == 0
    ), f"gt_gpu_transformation(): found unknown arguments: {', '.join(arg for arg in kwargs.keys())}"

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
        # A Tasklet, outside of a Map, that writes into an array on GPU can not work
        #  `sdfg.appyl_gpu_transformations()` puts Map around it (if said Tasklet
        #  would write into a Scalar that then goes into a GPU Map, nothing would
        #  happen. So we might end up with lot of these trivial Maps, that results
        #  in a single kernel launch. To prevent this we will try to fuse them.
        # NOTE: The current implementation has a bug, because promotion and fusion
        #   are two different steps. Because of this the function will implicitly
        #   fuse everything together it can find.
        # TODO(phimuell): Fix the issue described above.
        sdfg.apply_transformations_once_everywhere(
            TrivialGPUMapPromoter(),
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
    """Set the block size related properties of _all_ Maps.

    See `GPUSetBlockSize` for more information.

    Args:
        sdfg: The SDFG to process.
        gpu_block_size: The size of a thread block on the GPU.
        launch_bounds: The value for the launch bound that should be used.
        launch_factor: If no `launch_bounds` was given use the number of threads
            in a block multiplied by this number.
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
    if isinstance(val, (tuple | list)):
        pass
    elif isinstance(val, str):
        val = tuple(x.strip() for x in val.split(","))
    elif isinstance(val, int):
        val = (val,)
    else:
        raise TypeError(
            f"Does not know how to transform '{type(org_val).__name__}' into a proper GPU block size."
        )
    if 0 < len(val) <= 3:
        val = [*val, *([1] * (3 - len(val)))]
    else:
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


@dace_properties.make_properties
class GPUSetBlockSize(dace_transformation.SingleStateTransformation):
    """Sets the GPU block size on GPU Maps.

    The transformation will apply to all Maps that have a GPU schedule, regardless
    of their dimensionality.

    The `gpu_block_size` is either a sequence, of up to three integers or a string
    of up to three numbers, separated by comma (`,`).
    The first number is the size of the block in `x` direction, the second for the
    `y` direction and the third for the `z` direction. Missing values will be filled
    with `1`.

    Args:
        block_size: The size of a thread block on the GPU.
        launch_bounds: The value for the launch bound that should be used.
        launch_factor: If no `launch_bounds` was given use the number of threads
            in a block multiplied by this number.

    Todo:
        Add the possibility to specify other bounds for 1, 2, or 3 dimensional maps.
    """

    block_size = dace_properties.Property(
        dtype=None,
        allow_none=False,
        default=(32, 1, 1),
        setter=_gpu_block_parser,
        getter=_gpu_block_getter,
        desc="Size of the block size a GPU Map should have.",
    )

    launch_bounds = dace_properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="Set the launch bound property of the map.",
    )

    map_entry = dace_transformation.transformation.PatternNode(dace_nodes.MapEntry)

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
        graph: Union[dace.SDFGState, dace.SDFG],
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
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
    ) -> None:
        """Modify the map as requested."""
        self.map_entry.map.gpu_block_size = self.block_size
        if self.launch_bounds is not None:  # Note empty string has a meaning in DaCe
            self.map_entry.map.gpu_launch_bounds = self.launch_bounds


@dace_properties.make_properties
class TrivialGPUMapPromoter(dace_transformation.SingleStateTransformation):
    """Serial Map promoter for empty GPU maps.

    In CPU mode a Tasklet can be outside of a map, however, this is not
    possible in GPU mode. For this reason DaCe wraps such Tasklets in a
    trivial Map.
    This transformation will look for such Maps and promote them, such
    that they can be fused with downstream maps.

    Note:
        - This transformation should not be run on its own, instead it
            is run within the context of `gt_gpu_transformation()`.
        - This transformation must be run after the GPU Transformation.
        - Currently the transformation does not do the fusion on its own.
            Instead map fusion must be run afterwards.
        - The transformation assumes that the upper Map is a trivial Tasklet.
            Which should be the majority of all cases.
    """

    # Pattern Matching
    trivial_map_exit = dace_transformation.transformation.PatternNode(dace_nodes.MapExit)
    access_node = dace_transformation.transformation.PatternNode(dace_nodes.AccessNode)
    second_map_entry = dace_transformation.transformation.PatternNode(dace_nodes.MapEntry)

    @classmethod
    def expressions(cls) -> Any:
        return [
            dace.sdfg.utils.node_path_graph(
                cls.trivial_map_exit, cls.access_node, cls.second_map_entry
            )
        ]

    def can_be_applied(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        """Tests if the promotion is possible.

        The tests includes:
        - Schedule of the maps.
        - If the map is trivial.
        - If the trivial map was not used to define a symbol.
        - Intermediate access node can only have in and out degree of 1.
        - The trivial map exit can only have one output.
        """
        trivial_map_exit: dace_nodes.MapExit = self.trivial_map_exit
        trivial_map: dace_nodes.Map = trivial_map_exit.map
        trivial_map_entry: dace_nodes.MapEntry = graph.entry_node(trivial_map_exit)
        second_map: dace_nodes.Map = self.second_map_entry.map
        access_node: dace_nodes.AccessNode = self.access_node

        # The kind of maps we are interested only have one parameter.
        if len(trivial_map.params) != 1:
            return False

        # Check if it is a GPU map
        for map_to_check in [trivial_map, second_map]:
            if map_to_check.schedule not in [
                dace.dtypes.ScheduleType.GPU_Device,
                dace.dtypes.ScheduleType.GPU_Default,
            ]:
                return False

        # Check if the map is trivial.
        for rng in trivial_map.range.ranges:
            if rng[0] != rng[1]:
                return False

        # Now we have to ensure that the symbol is not used inside the scope of the
        #  map, if it is, then the symbol is just there to define a symbol.
        scope_view = graph.scope_subgraph(
            trivial_map_entry,
            include_entry=False,
            include_exit=False,
        )
        if any(map_param in scope_view.free_symbols for map_param in trivial_map.params):
            return False

        # ensuring that the trivial map exit and the intermediate node have degree
        #  one is a cheap way to ensure that the map can be merged into the
        #  second map.
        if graph.in_degree(access_node) != 1:
            return False
        if graph.out_degree(access_node) != 1:
            return False
        if graph.out_degree(trivial_map_exit) != 1:
            return False

        return True

    def apply(self, graph: Union[dace.SDFGState, dace.SDFG], sdfg: dace.SDFG) -> None:
        """Performs the Map Promoting.

        The function essentially copies the parameters and the ranges from the
        bottom map to the top one.
        """
        trivial_map: dace_nodes.Map = self.trivial_map_exit.map
        second_map: dace_nodes.Map = self.second_map_entry.map

        trivial_map.params = copy.deepcopy(second_map.params)
        trivial_map.range = copy.deepcopy(second_map.range)
