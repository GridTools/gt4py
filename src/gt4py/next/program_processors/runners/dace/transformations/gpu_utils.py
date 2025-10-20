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
from dace.sdfg import memlet_utils as dace_mutils, nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


def gt_gpu_transformation(
    sdfg: dace.SDFG,
    try_removing_trivial_maps: bool = True,
    use_gpu_storage: bool = True,
    gpu_block_size: Optional[Sequence[int | str] | str] = None,
    gpu_launch_bounds: Optional[int | str] = None,
    gpu_launch_factor: Optional[int] = None,
    gpu_block_size_spec: Optional[dict[str, Sequence[int | str] | str]] = None,
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
        gpu_block_size_spec: Specify thread block size per dimension, see
            `gt_set_gpu_blocksize()` for more.
        validate: Perform validation during the steps.
        validate_all: Perform extensive validation.

    Notes:
        - In addition it might fuse Maps together that should not be fused. To prevent
            that you should set `try_removing_trivial_maps` to `False`.
        - The function assumes that the iteration order has been set correctly before
            it is called.

    Todo:
        - Currently only one block size for all maps is given, add more options.
        - Investigate if the order of iteration is not changed (it should not).
        - Investigate if the trivial GPU map remover is still needed.
    """
    assert len(kwargs) == 0, (
        f"gt_gpu_transformation(): found unknown arguments: {', '.join(arg for arg in kwargs.keys())}"
    )

    # Turn all global arrays (which we identify as input) into GPU memory.
    #  This way the GPU transformation will not create this copying stuff.
    if use_gpu_storage:
        for desc in sdfg.arrays.values():
            if isinstance(desc, dace.data.Array) and not desc.transient:
                desc.storage = dace.dtypes.StorageType.GPU_Global

    # Now turn it into a GPU SDFG
    sdfg.apply_gpu_transformations(
        validate=False,
        validate_all=validate_all,
        simplify=False,
    )

    # The documentation recommends to run simplify afterwards
    gtx_transformations.gt_simplify(sdfg, validate=False, validate_all=validate_all)

    if try_removing_trivial_maps:
        # TODO(phimuell): Figuring out if it is still important/needed to do or if
        #   it can be removed, it should definitely be reworked.
        gt_remove_trivial_gpu_maps(
            sdfg=sdfg,
            validate=False,
            validate_all=validate_all,
        )
        gtx_transformations.gt_simplify(sdfg, validate=False, validate_all=validate_all)

    # TODO(phimuell): Fixing the stride problem in DaCe.
    sdfg = gt_gpu_transform_non_standard_memlet(
        sdfg=sdfg,
        map_postprocess=True,
        validate=False,
        validate_all=validate_all,
    )

    # Set the GPU block size if it is known.
    if gpu_block_size is not None or gpu_block_size_spec is not None:
        gpu_block_size_spec = gpu_block_size_spec or {}
        gt_set_gpu_blocksize(
            sdfg=sdfg,
            block_size=gpu_block_size,
            launch_bounds=gpu_launch_bounds,
            launch_factor=gpu_launch_factor,
            **gpu_block_size_spec,
            validate=False,
            validate_all=validate_all,
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
    """Transform some non-standard Melets to Maps.

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

    # Expand all non-standard memlets and get the new MapEntries.
    new_maps: set[dace_nodes.MapEntry] = _gt_expand_non_standard_memlets(sdfg)

    # If there are no Memlets that are translated to copy-Maps, then we have nothing to do.
    if len(new_maps) == 0:
        return sdfg

    # This function allows to restrict any fusion operation to the maps
    #  that we have just created.
    def restrict_fusion_to_newly_created_maps_vertical(
        self: gtx_transformations.MapFusionVertical,
        map_exit_1: dace_nodes.MapExit,
        map_entry_2: dace_nodes.MapEntry,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
    ) -> bool:
        return (map_entry_2 in new_maps) or (graph.entry_node(map_exit_1) in new_maps)

    def restrict_fusion_to_newly_created_maps_horizontal(
        self: gtx_transformations.MapFusionHorizontal,
        map_entry_1: dace_nodes.MapEntry,
        map_entry_2: dace_nodes.MapEntry,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
    ) -> bool:
        return (map_entry_1 in new_maps) or (map_entry_2 in new_maps)

    # Now try to fuse the maps together, but restrict them that at least one map
    #  needs to be new.
    # TODO(phimuell): Improve this by replacing it by an explicit loop.
    sdfg.apply_transformations_repeated(
        [
            gtx_transformations.MapFusionVertical(
                only_toplevel_maps=True,
                check_fusion_callback=restrict_fusion_to_newly_created_maps_vertical,
            ),
            gtx_transformations.MapFusionHorizontal(
                only_toplevel_maps=True,
                check_fusion_callback=restrict_fusion_to_newly_created_maps_horizontal,
            ),
        ],
        validate=False,
        validate_all=validate_all,
    )

    # This is a gross hack, but it is needed, for the following reasons:
    #  - The transients have C order while the non-transients have (most
    #       likely) FORTRAN order. So there is not an unique stride dimension.
    #  - The newly created maps have names that does not reflect GT4Py dimensions,
    #       thus we can not use `gt_set_iteration_order()`.
    #  For these reasons we do the simplest thing, which is assuming that the maps
    #  are created in C order and we must make them in FORTRAN order, which means
    #  just swapping the order of the map parameters.
    #  We further assume here, that we only have to process the maps that we have
    #  newly created.
    # NOTE: We can stop relying on this once [PR#1913](https://github.com/GridTools/gt4py/pull/1913)
    #   Has been merged, which is currently blocked by a DaCe PR that has not been
    #   merged.

    maps_to_modify: set[dace_nodes.MapEntry] = set()
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for map_entry in state.nodes():
                if not isinstance(map_entry, dace_nodes.MapEntry):
                    continue
                if map_entry in new_maps:
                    maps_to_modify.add(map_entry)

    # We did not found any of the newly created Map. Thus we **hope** that all new
    #  Maps have been integrated into other Maps, that have the correct names.
    #  But as written above, this is a gross hack!
    if len(maps_to_modify) == 0:
        return sdfg

    for me_to_modify in maps_to_modify:
        map_to_modify: dace_nodes.Map = me_to_modify.map
        map_to_modify.params = list(reversed(map_to_modify.params))
        map_to_modify.range = dace.subsets.Range(
            (r1, r2, r3, t)
            for (r1, r2, r3), t in zip(
                reversed(map_to_modify.range.ranges), reversed(map_to_modify.range.tile_sizes)
            )
        )

    if validate or validate_all:
        sdfg.validate()

    return sdfg


def _gt_expand_non_standard_memlets(
    sdfg: dace.SDFG,
) -> set[dace_nodes.MapEntry]:
    """Finds all non-standard Memlet in the SDFG and expand them.

    The function is used by `gt_gpu_transform_non_standard_memlet()` and performs
    the actual expansion of the Memlet, i.e. turning all Memlets that can not be
    expressed as a `memcpy()` into a Map, copy kernel.
    The function will return the MapEntries of all expanded.

    The function will process the SDFG recursively.
    """
    new_maps: set[dace_nodes.MapEntry] = set()
    for nsdfg in sdfg.all_sdfgs_recursive():
        new_maps.update(_gt_expand_non_standard_memlets_sdfg(nsdfg))
    return new_maps


def _gt_expand_non_standard_memlets_sdfg(
    sdfg: dace.SDFG,
) -> set[dace_nodes.MapEntry]:
    """Implementation of `_gt_expand_non_standard_memlets()` that process a single SDFG."""
    new_maps: set[dace_nodes.MapEntry] = set()
    # The implementation is based on DaCe's code generator, see `dace/codegen/targets/cuda.py`
    #  in the function `preprocess()`
    # NOTE: This implementation needs a DaCe version that includes https://github.com/spcl/dace/pull/2033
    for state in sdfg.states():
        for e in state.edges():
            # We are only interested in edges that connects two access nodes of GPU memory.
            #  However, we must exclude Memlets that are empty.
            if not (
                (not e.data.is_empty())
                and isinstance(e.src, dace_nodes.AccessNode)
                and isinstance(e.dst, dace_nodes.AccessNode)
                and e.src.desc(sdfg).storage == dace_dtypes.StorageType.GPU_Global
                and e.dst.desc(sdfg).storage == dace_dtypes.StorageType.GPU_Global
            ):
                continue

            a: dace_nodes.AccessNode = e.src
            b: dace_nodes.AccessNode = e.dst
            copy_shape, src_strides, dst_strides, _, _ = dace_cpp.memlet_copy_to_absolute_strides(
                None, sdfg, state, e, a, b
            )
            dims = len(copy_shape)
            if dims == 1:
                continue
            elif dims == 2:
                is_fortran_order = src_strides[0] == 1 and dst_strides[0] == 1
                is_c_order = src_strides[-1] == 1 and dst_strides[-1] == 1
                if is_c_order or is_fortran_order:
                    continue

                # NOTE: Special case of continuous copy
                # Example: dcol[0:I, 0:J, k] -> datacol[0:I, 0:J]
                # with copy shape [I, J] and strides [J*K, K], [J, 1]
                if src_strides[-1] != 1 or dst_strides[-1] != 1:
                    try:
                        is_src_cont = src_strides[0] / src_strides[1] == copy_shape[1]
                        is_dst_cont = dst_strides[0] / dst_strides[1] == copy_shape[1]
                    except (TypeError, ValueError):
                        is_src_cont = False
                        is_dst_cont = False
                    if is_src_cont and is_dst_cont:
                        continue

            elif dims > 2:
                if not (src_strides[-1] != 1 or dst_strides[-1] != 1):
                    continue

            # For identifying the new map, we first store all neighbors of `a`.
            old_neighbors_of_a: list[dace_nodes.AccessNode] = [
                edge.dst for edge in state.out_edges(a)
            ]

            if not dace_mutils.can_memlet_be_turned_into_a_map(
                edge=e, state=state, sdfg=sdfg, ignore_strides=True
            ):
                # NOTE: In DaCe, they simply ignore that case and continue to the
                #   code generator. In GT4Py we generate an error.
                raise RuntimeError(f"Unable to turn the not supported edge '{e}' into a copy Map.")

            # Turn the not supported Memlet into a copy Map. We have to do it here,
            #  such that we can then set their iteration order correctly.
            dace_mutils.memlet_to_map(
                edge=e,
                state=state,
                sdfg=sdfg,
                ignore_strides=True,
            )

            # We find the new map by comparing the new neighborhood of `a` with the old one.
            new_nodes: set[dace_nodes.MapEntry] = {
                edge.dst for edge in state.out_edges(a) if edge.dst not in old_neighbors_of_a
            }
            assert any(isinstance(new_node, dace_nodes.MapEntry) for new_node in new_nodes)
            new_maps.update(new_nodes)
    return new_maps


def gt_set_gpu_blocksize(
    sdfg: dace.SDFG,
    block_size: Optional[Sequence[int | str] | str],
    launch_bounds: Optional[int | str] = None,
    launch_factor: Optional[int] = None,
    validate: bool = True,
    validate_all: bool = False,
    **kwargs: Any,
) -> int:
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
        validate: Perform validation at the end of the function.
        validate_all: Perform validation also on intermediate steps.

    Note:
        If a Map is found whose range is smaller than the specified block size for
        that dimension then the block size for that map is reduced.
    """
    for dim in [1, 2, 3]:
        for arg, val in {
            "block_size": block_size,
            "launch_bounds": launch_bounds,
            "launch_factor": launch_factor,
        }.items():
            if f"{arg}_{dim}d" not in kwargs:
                kwargs[f"{arg}_{dim}d"] = val

    setter = GPUSetBlockSize(**kwargs)

    configured_maps = 0
    for state in sdfg.states():
        scope_dict: Union[None, dict[Any, Any]] = None
        cfg_id = state.parent_graph.cfg_id
        state_id = state.block_id
        for node in state.nodes():
            if not isinstance(node, dace_nodes.MapEntry):
                continue
            if scope_dict is None:
                scope_dict = state.scope_dict()
            if scope_dict[node] is not None:
                continue
            candidate = {GPUSetBlockSize.map_entry: node}
            setter.setup_match(
                sdfg=sdfg,
                cfg_id=cfg_id,
                state_id=state_id,
                subgraph=candidate,
                expr_index=0,
                override=True,
            )
            if setter.can_be_applied(state, 0, sdfg, False):
                setter.apply(state, sdfg)
                if validate_all:
                    sdfg.validate()
                configured_maps += 1

    if validate and (not validate_all):
        sdfg.validate()

    return configured_maps


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

    If a Map is found whose iteration size in a particular dimension is smaller than
    the block size, the block size for that dimension is reduced to that value. For
    example, the block size for 2D Maps is `(32, 8, 1)` and a Map that only performs
    four iteration in the second dimension, will get a block size of `(32, 4, 1)`.
    Note that this modification will not influence the launch bound value.

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
        - In case the Map has more than 3 dimension, normal DaCe semantic is used.

    Todo:
        - Turn this into a function.
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
        - If launch_bounds_1d is not set then set the launch_bounds of `scan` maps to 512 to limit their register usage.
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
        map_size = gpu_map.range.size()
        dims_to_inspect = len(map_size)
        num_map_params = 0
        for i, axis_size in enumerate(map_size):
            if i > 0 and map_size[i - 1] == 1:
                assert axis_size <= 1, (
                    "GPU thread block size setting currently does not support maps where non-leading "
                    "dimensions have size greater than one if the previous dimension has size one."
                )
            if axis_size != 1:
                num_map_params += 1  # Handle 2D maps where one dimension has range 1 as 1D map

        # Because of a particularity of the DaCe code generator, the iteration
        #  variable that is associated to the `x` dimension of the block is the
        #  last parameter, i.e. `gpu_map.params[-1]`. The one for `y` the second last.
        if num_map_params == 1:
            block_size = list(self.block_size_1d)
            launch_bounds = self.launch_bounds_1d
            if launch_bounds is None:
                for node in graph.scope_subgraph(
                    self.map_entry, include_entry=False, include_exit=False
                ):
                    if isinstance(node, dace_nodes.NestedSDFG) and node.label.startswith("scan_"):
                        launch_bounds = "512"  # Use high launch bound in case of scans to limit register usage and increase occupancy
        elif num_map_params == 2:
            block_size = list(self.block_size_2d)
            launch_bounds = self.launch_bounds_2d
        else:
            block_size = list(self.block_size_3d)
            launch_bounds = self.launch_bounds_3d

            # If there are more than three dimensions DaCe will condense them into
            #  the `z` dimension of the block, so we have to ignore the `z` dimension,
            #  when we modify the block sizes.
            if num_map_params > 3:
                dims_to_inspect = 2

        # block size can only have up to three dimensions
        assert dims_to_inspect <= 3
        # Cut down the block size.
        # TODO(phimuell): Think if it is useful to also modify the launch bounds.
        # TODO(phimuell): Also think of how to connect this with the loop blocking.
        for i in range(dims_to_inspect):
            map_dim_idx_to_inspect = len(gpu_map.params) - 1 - i
            if (map_size[map_dim_idx_to_inspect] < block_size[i]) == True:  # noqa: E712 [true-false-comparison]  # SymPy Fancy comparison.
                block_size[i] = map_size[map_dim_idx_to_inspect]

        gpu_map.gpu_block_size = tuple(block_size)
        if launch_bounds is not None:  # Note: empty string has a meaning in DaCe
            gpu_map.gpu_launch_bounds = launch_bounds


def gt_remove_trivial_gpu_maps(
    sdfg: dace.SDFG,
    validate: bool = True,
    validate_all: bool = False,
) -> dace.SDFG:
    """Removes trivial maps that were created by the GPU transformation.

    The main problem is that a Tasklet outside of a Map cannot write into an
    _array_ that is on GPU. `sdfg.apply_gpu_transformations()` will wrap such
    Tasklets in a Map. The `GT4PyMoveTaskletIntoMap` pass, that runs before,
    but only works if the tasklet is adjacent to a map.

    It first tries to promote them such that they can be fused in other non-trivial
    maps, it will then also perform fusion on them, to reduce the number of kernel
    calls.

    Args:
        sdfg: The SDFG that we process.
        validate: Perform validation at the end of the function.
        validate_all: Perform validation also on intermediate steps.

    Todo: Improve this function.
    """

    # First we try to promote and fuse them with other non-trivial maps.
    sdfg.apply_transformations_once_everywhere(
        TrivialGPUMapElimination(
            do_not_fuse=False,
            only_gpu_maps=True,
        ),
        validate=False,
        validate_all=validate_all,
    )
    gtx_transformations.gt_simplify(sdfg, validate=False, validate_all=validate_all)

    # Now we try to fuse them together, however, we restrict the fusion to trivial
    #  GPU map.
    def restrict_to_trivial_gpu_maps(
        self: Union[gtx_transformations.MapFusionVertical, gtx_transformations.MapFusionHorizontal],
        map_node_1: Union[dace_nodes.MapEntry, dace_nodes.MapExit],
        map_entry_2: dace_nodes.MapEntry,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
    ) -> bool:
        map_entry_1 = (
            map_node_1
            if isinstance(map_node_1, dace_nodes.MapEntry)
            else graph.entry_node(map_node_1)
        )
        for map_entry in [map_entry_1, map_entry_2]:
            _map = map_entry.map
            if len(_map.params) != 1:
                return False
            if _map.range[0][0] != _map.range[0][1]:
                return False
            if _map.schedule not in [
                dace.dtypes.ScheduleType.GPU_Device,
                dace.dtypes.ScheduleType.GPU_Default,
            ]:
                return False
        return True

    # TODO(phimuell): Replace this with a more performant loop.
    sdfg.apply_transformations_repeated(
        [
            gtx_transformations.MapFusionVertical(
                only_toplevel_maps=True,
                check_fusion_callback=restrict_to_trivial_gpu_maps,
            ),
            gtx_transformations.MapFusionHorizontal(
                only_toplevel_maps=True,
                check_fusion_callback=restrict_to_trivial_gpu_maps,
            ),
        ],
        validate=False,
        validate_all=validate_all,
    )

    if validate and (not validate_all):
        sdfg.validate()

    return sdfg


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

    Todo: Figuring out if this transformation is still needed.
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
        # NOTE: We use the same options as in `MapPromoter`, however, since we do
        #   not have the list of single use data available, we have to specify it.
        org_trivial_map_params = copy.deepcopy(trivial_map.params)
        org_trivial_map_range = copy.deepcopy(trivial_map.range)
        try:
            self._promote_map(graph, replace_trivail_map_parameter=False)
            if not gtx_transformations.MapFusionVertical.can_be_applied_to(
                sdfg=sdfg,
                options={
                    "only_toplevel_maps": True,
                    "require_all_intermediates": True,
                    "require_exclusive_intermediates": True,
                },
                first_map_exit=trivial_map_exit,
                array=self.access_node,
                second_map_entry=self.second_map_entry,
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
        #  For the selection of the option, see the `can_be_applied()` function.
        if not self.do_not_fuse:
            gtx_transformations.MapFusionVertical.apply_to(
                sdfg=sdfg,
                options={
                    "only_toplevel_maps": True,
                    "require_all_intermediates": True,
                    "require_exclusive_intermediates": True,
                },
                first_map_exit=trivial_map_exit,
                array=access_node,
                second_map_entry=second_map_entry,
                verify=False,  # Do not rerun `can_be_applied()`.
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


def gt_gpu_apply_mempool(sdfg: dace.SDFG) -> None:
    """Enables the stream ordered memory allocator in CUDA code generation.

    It sets the `pool` flag on all transients allocated in GPU global memory.
    The dace code genarator will handle this flag by calling `cudaMallocAsync`
    and `cudaFreeAsync` in the CUDA code for allocation/release of the buffers.

    Args:
        sdfg: The SDFG that should be processed.
    """
    for _, _, desc in sdfg.arrays_recursive():
        if (
            isinstance(desc, dace.data.Array)
            and desc.storage == dace.StorageType.GPU_Global
            and desc.transient
        ):
            desc.pool = True
