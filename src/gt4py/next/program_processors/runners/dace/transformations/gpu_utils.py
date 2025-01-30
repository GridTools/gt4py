# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Functions for turning an SDFG into a GPU SDFG."""

from __future__ import annotations

import copy
from typing import Any, Callable, Final, Optional, Sequence, Union

import dace
from dace import (
    dtypes as dace_dtypes,
    properties as dace_properties,
    transformation as dace_transformation,
)
from dace.codegen.targets import cpp as dace_cpp
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


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
            Will only take effect if `gpu_block_size` is specified.
        gpu_launch_factor: Use the number of threads times this value as `__launch_bounds__`
            Will only take effect if `gpu_block_size` is specified.
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
        # In DaCe a Tasklet, outside of a Map, can not write into an _array_ that is on
        #  GPU. `sdfg.apply_gpu_transformations()` will wrap such Tasklets in a Map. So
        #  we might end up with lots of these trivial Maps, each requiring a separate
        #  kernel launch. To prevent this we will combine these trivial maps, if
        #  possible, with their downstream maps.
        sdfg.apply_transformations_once_everywhere(
            TrivialGPUMapElimination(),
            validate=False,
            validate_all=False,
        )
        gtx_transformations.gt_simplify(sdfg, validate=validate, validate_all=validate_all)

    # TODO(phimuell): Fixing the stride problem.
    sdfg = gt_gpu_transform_non_standard_memlet(
        sdfg=sdfg,
        map_postprocess=True,
        validate=validate,
        validate_all=validate_all,
    )

    # Set the GPU block size if it is known.
    if gpu_block_size is not None:
        gt_set_gpu_blocksize(
            sdfg=sdfg,
            block_size=gpu_block_size,
            launch_bounds=gpu_launch_bounds,
            launch_factor=gpu_launch_factor,
        )

    if validate_all or validate:
        sdfg.validate()

    return sdfg


def gt_gpu_transform_non_standard_memlet(
    sdfg: dace.SDFG,
    map_postprocess: bool,
    validate: bool = True,
    validate_all: bool = False,
) -> dace.SDFG:
    """Transform some non standard Melets to Maps.

    The GPU code generator is not able to handle certain sets of Memlets. To
    handle them, the code generator transforms them into copy Maps. The main
    issue is that this transformation happens after the auto optimizer, thus
    the copy-Maps will most likely have the wrong iteration order.

    This function allows to perform the preprocessing step before the actual
    code generation. The function will perform the expansion. If
    `map_postprocess` is `True` then the function will also apply MapFusion,
    to these newly created copy-Maps and set their iteration order correctly.

    A user should not call this function directly, instead this function is
    called by the `gt_gpu_transformation()` function.

    Args:
        sdfg: The SDFG that we process.
        map_postprocess: Enable post processing of the maps that are created.
            See the Note section below.
        validate: Perform validation at the end of the function.
        validate_all: Perform validation also on intermediate steps.

    Note:
        - Currently the function applies some crude heuristic to determine the
            correct loop order.
        - This function should be called after `gt_set_iteration_order()` has run.
    """
    new_maps: set[dace_nodes.MapEntry] = set()

    # This code is is copied from DaCe's code generator.
    for e, state in list(sdfg.all_edges_recursive()):
        nsdfg = state.parent
        if (
            isinstance(e.src, dace_nodes.AccessNode)
            and isinstance(e.dst, dace_nodes.AccessNode)
            and e.src.desc(nsdfg).storage == dace_dtypes.StorageType.GPU_Global
            and e.dst.desc(nsdfg).storage == dace_dtypes.StorageType.GPU_Global
        ):
            a: dace_nodes.AccessNode = e.src
            b: dace_nodes.AccessNode = e.dst

            copy_shape, src_strides, dst_strides, _, _ = dace_cpp.memlet_copy_to_absolute_strides(
                None, nsdfg, state, e, a, b
            )
            dims = len(copy_shape)
            if dims == 1:
                continue
            elif dims == 2:
                if src_strides[-1] != 1 or dst_strides[-1] != 1:
                    try:
                        is_src_cont = src_strides[0] / src_strides[1] == copy_shape[1]
                        is_dst_cont = dst_strides[0] / dst_strides[1] == copy_shape[1]
                    except (TypeError, ValueError):
                        is_src_cont = False
                        is_dst_cont = False
                    if is_src_cont and is_dst_cont:
                        continue
                else:
                    continue
            elif dims > 2:
                if not (src_strides[-1] != 1 or dst_strides[-1] != 1):
                    continue

            # For identifying the new map, we first store all neighbors of `a`.
            old_neighbors_of_a: list[dace_nodes.AccessNode] = [
                edge.dst for edge in state.out_edges(a)
            ]

            # Turn unsupported copy to a map
            try:
                dace_transformation.dataflow.CopyToMap.apply_to(
                    nsdfg, save=False, annotate=False, a=a, b=b
                )
            except ValueError:  # If transformation doesn't match, continue normally
                continue

            # We find the new map by comparing the new neighborhood of `a` with the old one.
            new_nodes: set[dace_nodes.MapEntry] = {
                edge.dst for edge in state.out_edges(a) if edge.dst not in old_neighbors_of_a
            }
            assert any(isinstance(new_node, dace_nodes.MapEntry) for new_node in new_nodes)
            assert len(new_nodes) == 1
            new_maps.update(new_nodes)

    # If there are no Memlets that are translated to copy-Maps, then we have nothing to do.
    if len(new_maps) == 0:
        return sdfg

    # This function allows to restrict any fusion operation to the maps
    #  that we have just created.
    def restrict_fusion_to_newly_created_maps(
        self: gtx_transformations.map_fusion_helper.MapFusionHelper,
        map_entry_1: dace_nodes.MapEntry,
        map_entry_2: dace_nodes.MapEntry,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
        permissive: bool,
    ) -> bool:
        return any(new_entry in new_maps for new_entry in [map_entry_1, map_entry_2])

    # Using the callback to restrict the fusing
    sdfg.apply_transformations_repeated(
        [
            gtx_transformations.MapFusionSerial(
                only_toplevel_maps=True,
                apply_fusion_callback=restrict_fusion_to_newly_created_maps,
            ),
            gtx_transformations.MapFusionParallel(
                only_toplevel_maps=True,
                apply_fusion_callback=restrict_fusion_to_newly_created_maps,
            ),
        ],
        validate=validate,
        validate_all=validate_all,
    )

    # Now we have to find the maps that were not fused. We rely here on the fact
    #  that at least one of the map that is involved in fusing still exists.
    maps_to_modify: set[dace_nodes.MapEntry] = set()
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for map_entry in state.nodes():
                if not isinstance(map_entry, dace_nodes.MapEntry):
                    continue
                if map_entry in new_maps:
                    maps_to_modify.add(map_entry)
    assert 0 < len(maps_to_modify) <= len(new_maps)

    # This is a gross hack, but it is needed, for the following reasons:
    #  - The transients have C order while the non-transients have (most
    #       likely) FORTRAN order. So there is not an unique stride dimension.
    #  - The newly created maps have names that does not reflect GT4Py dimensions,
    #       thus we can not use `gt_set_iteration_order()`.
    #  For these reasons we do the simplest thing, which is assuming that the maps
    #  are created in C order and we must make them in FORTRAN order, which means
    #  just swapping the order of the map parameters.
    # TODO(phimuell): Do it properly.
    for me_to_modify in maps_to_modify:
        map_to_modify: dace_nodes.Map = me_to_modify.map
        map_to_modify.params = list(reversed(map_to_modify.params))
        map_to_modify.range = dace.subsets.Range(
            (r1, r2, r3, t)
            for (r1, r2, r3), t in zip(
                reversed(map_to_modify.range.ranges), reversed(map_to_modify.range.tile_sizes)
            )
        )

    return sdfg


def gt_set_gpu_blocksize(
    sdfg: dace.SDFG,
    block_size: Optional[Sequence[int | str] | str],
    launch_bounds: Optional[int | str] = None,
    launch_factor: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Set the block size related properties of _all_ Maps.

    It supports the same arguments as `GPUSetBlockSize`, however it also has
    versions without `_Xd`, these are used as default for the other maps.
    If a version with `_Xd` is specified then it takes precedence.

    Args:
        sdfg: The SDFG to process.
        block_size: The size of a thread block on the GPU.
        launch_bounds: The value for the launch bound that should be used.
        launch_factor: If no `launch_bounds` was given use the number of threads
            in a block multiplied by this number.
    """
    for dim in [1, 2, 3]:
        for arg, val in {
            "block_size": block_size,
            "launch_bounds": launch_bounds,
            "launch_factor": launch_factor,
        }.items():
            if f"{arg}_{dim}d" not in kwargs:
                kwargs[f"{arg}_{dim}d"] = val
    return sdfg.apply_transformations_once_everywhere(GPUSetBlockSize(**kwargs))


def _make_gpu_block_parser_for(
    dim: int,
) -> Callable[["GPUSetBlockSize", Any], None]:
    """Generates a parser for GPU blocks for dimension `dim`.

    The returned function can be used as parser for the `GPUSetBlockSize.block_size_*d`
    properties.
    """

    def _gpu_block_parser(
        self: GPUSetBlockSize,
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
        if len(val) < dim:
            raise ValueError(
                f"The passed block size only covers {len(val)} dimensions, but dimension was {dim}."
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

        # Remove over specification.
        for i in range(dim, 3):
            val[i] = 1
        setattr(self, f"_block_size_{dim}d", tuple(val))

    return _gpu_block_parser


def _make_gpu_block_getter_for(
    dim: int,
) -> Callable[["GPUSetBlockSize"], tuple[int, int, int]]:
    """Makes the getter for the block size of dimension `dim`."""

    def _gpu_block_getter(
        self: "GPUSetBlockSize",
    ) -> tuple[int, int, int]:
        """Used as getter in the `GPUSetBlockSize.block_size` property."""
        return getattr(self, f"_block_size_{dim}d")

    return _gpu_block_getter


def _gpu_launch_bound_parser(
    block_size: tuple[int, int, int],
    launch_bounds: int | str | None,
    launch_factor: int | None = None,
) -> str | None:
    """Used by the `GPUSetBlockSize.__init__()` method to parse the launch bounds."""
    if launch_bounds is None and launch_factor is None:
        return None
    elif launch_bounds is None and launch_factor is not None:
        return str(int(launch_factor) * block_size[0] * block_size[1] * block_size[2])
    elif launch_bounds is not None and launch_factor is None:
        assert isinstance(launch_bounds, (str, int))
        return str(launch_bounds)
    else:
        raise ValueError("Specified both `launch_bounds` and `launch_factor`.")


@dace_properties.make_properties
class GPUSetBlockSize(dace_transformation.SingleStateTransformation):
    """Sets the GPU block size on GPU Maps.

    The `block_size` is either a sequence, of up to three integers or a string
    of up to three numbers, separated by comma (`,`). The first number is the size
    of the block in `x` direction, the second for the `y` direction and the third
    for the `z` direction. Missing values will be filled with `1`.

    A different value for the GPU block size and launch bound can be specified for
    maps of dimension 1, 2 or 3 (all maps with higher dimensions are considered
    three dimensional). If no value is specified then the block size `(32, 1, 1)`
    will be used an no launch bound will be be emitted.

    Args:
        block_size_Xd: The size of a thread block on the GPU for `X` dimensional maps.
        launch_bounds_Xd: The value for the launch bound that should be used for `X`
            dimensional maps.
        launch_factor_Xd: If no `launch_bounds` was given use the number of threads
            in a block multiplied by this number, for maps of dimension `X`.

    Note:
        - You should use the `gt_set_gpu_blocksize()` function.
        - "Over specification" is ignored, i.e. if `(32, 3, 1)` is passed as block
            size for 1 dimensional maps, then it is changed to `(32, 1, 1)`.
    """

    _block_size_default: Final[tuple[int, int, int]] = (32, 1, 1)

    block_size_1d = dace_properties.Property(
        dtype=tuple[int, int, int],
        default=_block_size_default,
        setter=_make_gpu_block_parser_for(1),
        getter=_make_gpu_block_getter_for(1),
        desc="Block size for 1 dimensional GPU maps.",
    )
    launch_bounds_1d = dace_properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="Set the launch bound property for 1 dimensional map.",
    )
    block_size_2d = dace_properties.Property(
        dtype=tuple[int, int, int],
        default=_block_size_default,
        setter=_make_gpu_block_parser_for(2),
        getter=_make_gpu_block_getter_for(2),
        desc="Block size for 2 dimensional GPU maps.",
    )
    launch_bounds_2d = dace_properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="Set the launch bound property for 2 dimensional map.",
    )
    block_size_3d = dace_properties.Property(
        dtype=tuple[int, int, int],
        default=_block_size_default,
        setter=_make_gpu_block_parser_for(3),
        getter=_make_gpu_block_getter_for(3),
        desc="Block size for 3 dimensional GPU maps.",
    )
    launch_bounds_3d = dace_properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="Set the launch bound property for 3 dimensional map.",
    )

    # Pattern matching
    map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)

    def __init__(
        self,
        block_size_1d: Sequence[int | str] | str | None = None,
        block_size_2d: Sequence[int | str] | str | None = None,
        block_size_3d: Sequence[int | str] | str | None = None,
        launch_bounds_1d: int | str | None = None,
        launch_bounds_2d: int | str | None = None,
        launch_bounds_3d: int | str | None = None,
        launch_factor_1d: int | None = None,
        launch_factor_2d: int | None = None,
        launch_factor_3d: int | None = None,
    ) -> None:
        super().__init__()
        if block_size_1d is not None:
            self.block_size_1d = block_size_1d
        if block_size_2d is not None:
            self.block_size_2d = block_size_2d
        if block_size_3d is not None:
            self.block_size_3d = block_size_3d
        self.launch_bounds_1d = _gpu_launch_bound_parser(
            self.block_size_1d, launch_bounds_1d, launch_factor_1d
        )
        self.launch_bounds_2d = _gpu_launch_bound_parser(
            self.block_size_2d, launch_bounds_2d, launch_factor_2d
        )
        self.launch_bounds_3d = _gpu_launch_bound_parser(
            self.block_size_3d, launch_bounds_3d, launch_factor_3d
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
        gpu_map: dace_nodes.Map = self.map_entry.map
        if len(gpu_map.params) == 1:
            block_size = self.block_size_1d
            launch_bounds = self.launch_bounds_1d
        elif len(gpu_map.params) == 2:
            block_size = self.block_size_2d
            launch_bounds = self.launch_bounds_2d
        else:
            block_size = self.block_size_3d
            launch_bounds = self.launch_bounds_3d
        gpu_map.gpu_block_size = block_size
        if launch_bounds is not None:  # Note: empty string has a meaning in DaCe
            gpu_map.gpu_launch_bounds = launch_bounds


@dace_properties.make_properties
class TrivialGPUMapElimination(dace_transformation.SingleStateTransformation):
    """Eliminate certain kind of trivial GPU maps.

    A tasklet outside of map can not write to GPU memory, this can only be done
    from within a map (a scalar is possible). For that reason DaCe's GPU
    transformation wraps such tasklets in trivial maps.
    Under certain condition the transformation will fuse the trivial tasklet with
    a downstream (serial) map.

    Args:
        do_not_fuse: If `True` then the maps are not fused together.
        only_gpu_maps: Only apply to GPU maps; `True` by default.

    Note:
        - This transformation should not be run on its own, instead it
            is run within the context of `gt_gpu_transformation()`.
        - This transformation must be run after the GPU Transformation.
    """

    only_gpu_maps = dace_properties.Property(
        dtype=bool,
        default=True,
        desc="Only promote maps that are GPU maps (debug option).",
    )
    do_not_fuse = dace_properties.Property(
        dtype=bool,
        default=False,
        desc="Only perform the promotion, do not fuse.",
    )

    # Pattern Matching
    trivial_map_exit = dace_transformation.PatternNode(dace_nodes.MapExit)
    access_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
    second_map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)

    def __init__(
        self,
        do_not_fuse: Optional[bool] = None,
        only_gpu_maps: Optional[bool] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if only_gpu_maps is not None:
            self.only_gpu_maps = only_gpu_maps
        if do_not_fuse is not None:
            self.do_not_fuse = do_not_fuse

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
        - Tests if the maps can be fused.
        """
        trivial_map_exit: dace_nodes.MapExit = self.trivial_map_exit
        trivial_map: dace_nodes.Map = trivial_map_exit.map
        trivial_map_entry: dace_nodes.MapEntry = graph.entry_node(trivial_map_exit)
        second_map: dace_nodes.Map = self.second_map_entry.map

        # The kind of maps we are interested only have one parameter.
        if len(trivial_map.params) != 1:
            return False
        for rng in trivial_map.range.ranges:
            if rng[0] != rng[1]:
                return False

        # If we do not not fuse, then the second map can not be trivial.
        #  If we would not prevent that case then we would match these two
        #  maps again and again.
        if self.do_not_fuse and len(second_map.params) <= 1:
            for rng in second_map.range.ranges:
                if rng[0] == rng[1]:
                    return False

        # We now check that the Memlets do not depend on the map parameter.
        #  This is important for the `can_be_applied_to()` check we do below
        #  because we can avoid calling the replace function.
        scope = graph.scope_subgraph(trivial_map_entry)
        trivial_map_param: str = trivial_map.params[0]
        for edge in scope.edges():
            if trivial_map_param in edge.data.free_symbols:
                return False

        # Check if only GPU maps are involved (this is more a testing debug feature).
        if self.only_gpu_maps:
            for map_to_check in [trivial_map, second_map]:
                if map_to_check.schedule not in [
                    dace.dtypes.ScheduleType.GPU_Device,
                    dace.dtypes.ScheduleType.GPU_Default,
                ]:
                    return False

        # Now we check if the two maps can be fused together. For that we have to
        #  do a temporary promotion, it is important that we do not perform the
        #  renaming. If the old symbol is still used, it is used inside a tasklet
        #  so it would show up (temporarily) as free symbol.
        org_trivial_map_params = copy.deepcopy(trivial_map.params)
        org_trivial_map_range = copy.deepcopy(trivial_map.range)
        try:
            self._promote_map(graph, replace_trivail_map_parameter=False)
            if not gtx_transformations.MapFusionSerial.can_be_applied_to(
                sdfg=sdfg,
                map_exit_1=trivial_map_exit,
                intermediate_access_node=self.access_node,
                map_entry_2=self.second_map_entry,
            ):
                return False
        finally:
            trivial_map.params = org_trivial_map_params
            trivial_map.range = org_trivial_map_range

        return True

    def apply(self, graph: Union[dace.SDFGState, dace.SDFG], sdfg: dace.SDFG) -> None:
        """Performs the Map Promoting.

        The function will first perform the promotion of the trivial map and then
        perform the merging of the two maps in one go.
        """
        trivial_map_exit: dace_nodes.MapExit = self.trivial_map_exit
        second_map_entry: dace_nodes.MapEntry = self.second_map_entry
        access_node: dace_nodes.AccessNode = self.access_node

        # Promote the maps.
        self._promote_map(graph)

        # Perform the fusing if requested.
        if not self.do_not_fuse:
            gtx_transformations.MapFusionSerial.apply_to(
                sdfg=sdfg,
                map_exit_1=trivial_map_exit,
                intermediate_access_node=access_node,
                map_entry_2=second_map_entry,
                verify=True,
            )

    def _promote_map(
        self,
        state: dace.SDFGState,
        replace_trivail_map_parameter: bool = True,
    ) -> None:
        """Performs the map promoting.

        Essentially this function will copy the parameters and the range from
        the non trivial map (`self.second_map_entry.map`) to the trivial map
        (`self.trivial_map_exit.map`).

        If `replace_trivail_map_parameter` is `True` (the default value), then the
        function will also remove the trivial map parameter with its value.
        """
        assert isinstance(self.trivial_map_exit, dace_nodes.MapExit)
        assert isinstance(self.second_map_entry, dace_nodes.MapEntry)
        assert isinstance(self.access_node, dace_nodes.AccessNode)

        trivial_map_exit: dace_nodes.MapExit = self.trivial_map_exit
        trivial_map: dace_nodes.Map = self.trivial_map_exit.map
        trivial_map_entry: dace_nodes.MapEntry = state.entry_node(trivial_map_exit)
        second_map: dace_nodes.Map = self.second_map_entry.map

        # If requested then replace the map variable with its value.
        if replace_trivail_map_parameter:
            scope = state.scope_subgraph(trivial_map_entry)
            scope.replace(trivial_map.params[0], trivial_map.range[0][0])

        # Now copy parameter and the ranges from the second to the trivial map.
        trivial_map.params = copy.deepcopy(second_map.params)
        trivial_map.range = copy.deepcopy(second_map.range)
