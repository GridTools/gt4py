# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Fast access to the auto optimization on DaCe."""

import enum
import warnings
from typing import Any, Callable, Optional, Sequence, TypeAlias, Union

import dace
from dace import data as dace_data
from dace.sdfg import nodes as dace_nodes, propagation as dace_propagation
from dace.transformation.auto import auto_optimize as dace_aoptimize
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


class GT4PyAutoOptHook(enum.Enum):
    """Allows to hook into different stages of the auto optimizer.

    The hook system allows to inject certain additional behaviour at various places
    during the execution of `gt_auto_optimize()`. See the list below for more specific
    instructions. Most of the hooks are expected to conform to the signature specified
    by `GT4PyAutoOptHookStage`. However, some hooks require a different signature.

    The supported values are:
    - TopLevelDataFlowPre: Called before the top level dataflow is optimized.
    - TopLevelDataFlowStep: Called during the top level optimization stage, which might be
        called multiple times.
    - TopLevelDataFlowMapFusionVerticalCallBack: If provided this function is passed
        as `check_fusion_callback` argument to `MapFusionVertical` during the top level
        dataflow optimization stage. This hook is also forwarded to the
        `gt_vertical_map_split_fusion()` function, which is called with
        `fuse_map_fragments=False` but `run_map_fusion=True`. Note that this hook has
        to meet the requirements of `VerticalMapFusionCallback`, see the `map_fusion`
        module for more information.
    - TopLevelDataFlowMapFusionHorizontalCallBack: If provided this function is passed
        as `check_fusion_callback` argument to `MapFusionHorizontal` during the top
        level dataflow optimization. This hook is also forwarded to the
        `gt_horizontal_map_split_fusion()` function, which is called with
        `fuse_map_fragments=False` but `run_map_fusion=True`. Note that this hook has
        to meet the requirements of `HorizontalMapFusionCallback`, see the `map_fusion`
        module for more information.
    - TopLevelDataFlowMapPromotionCallBack: If provided then pass this function as
        `promotion_callback` argument to the `MapPromoter` during top level dataflow
        optimization. If not provided then `gt4py_default_auto_optimizer_map_promotion_checker()`
        is used. Note that this callback has to meet the requirements of
        `MapPromotionCallBack`.
    - `TopLevelDataFlowVerticalSplitCallBack`: If provided it is passed as `check_split_callback`
        to `gt_vertical_map_split_fusion()` and allows to control how a vertical Map
        split is done. See `VerticalMapSplitCallback` for more details.
        Note that the function is called with `fuse_map_fragments=False` but
        `run_map_fusion=True`.
    - `TopLevelDataFlowHorizontalSplitCallBack`: If provided it is passed as `check_split_callback`
        to `gt_horizontal_map_split_fusion()` and allows to control how a horizontal
        Map split is done. See `HorizontalMapSplitCallback` for more details.
        Note that the function is called with `fuse_map_fragments=False` but
        `run_map_fusion=True`.
    - TopLevelDataFlowPost: Called after the top level dataflow has been optimized.
    - AfterToGPU: This hook is called after the GPU transformation has been called,
        if the SDFG should run on GPU.

    Note:
        The `TopLevelDataFlowMapFusionVerticalCallBack` and `TopLevelDataFlowMapFusionHorizontalCallBack`
        hooks are not passed to the splitting transformations, this is a limitation of
        the toolchain and should be fixed soon. If they are used it is advised to
        pass `disable_splitting` to `gt_auto_optimize()` and call the splitting
        transformations directly if desired.

    Todo:
        - Implement a hook for the splitting transformation.
    """

    TopLevelDataFlowPre = enum.auto()
    TopLevelDataFlowMapFusionVerticalCallBack = enum.auto()
    TopLevelDataFlowMapFusionHorizontalCallBack = enum.auto()
    TopLevelDataFlowMapPromotionCallBack = enum.auto()
    TopLevelDataFlowVerticalSplitCallBack = enum.auto()
    TopLevelDataFlowHorizontalSplitCallBack = enum.auto()
    TopLevelDataFlowStep = enum.auto()
    TopLevelDataFlowPost = enum.auto()
    AfterToGPU = enum.auto()


GT4PyAutoOptHookStage: TypeAlias = Callable[[dace.SDFG], None]
"""Signature of a regular "stage hook" of the optimizer.

The function gets the SDFG as its single argument and processes it in-place.
Most of the hooks `gt_auto_optimize()` provides are expected to be of that type.
"""


GT4PyAutoOptHookFun: TypeAlias = Union[
    GT4PyAutoOptHookStage,
    "gtx_transformations.HorizontalMapFusionCallback",
    "gtx_transformations.VerticalMapFusionCallback",
    "gtx_transformations.MapPromotionCallBack",
    "gtx_transformations.VerticalMapSplitCallback",
    "gtx_transformations.HorizontalMapSplitCallback",
]


def gt_auto_optimize(
    sdfg: dace.SDFG,
    gpu: bool,
    unit_strides_kind: Optional[gtx_common.DimensionKind] = None,
    make_persistent: bool = False,
    gpu_block_size: Optional[Sequence[int | str] | str] = (32, 8, 1),
    gpu_block_size_1d: Optional[Sequence[int | str] | str] = (64, 1, 1),
    gpu_block_size_2d: Optional[Sequence[int | str] | str] = None,
    gpu_block_size_3d: Optional[Sequence[int | str] | str] = None,
    blocking_dim: Optional[gtx_common.Dimension] = None,
    blocking_size: int = 10,
    blocking_only_if_independent_nodes: bool = True,
    disable_splitting: bool = False,
    reuse_transients: bool = False,
    gpu_launch_bounds: Optional[int | str] = None,
    gpu_launch_factor: Optional[int] = None,
    gpu_memory_pool: bool = True,
    constant_symbols: Optional[dict[str, Any]] = None,
    assume_pointwise: bool = True,
    optimization_hooks: Optional[dict[GT4PyAutoOptHook, GT4PyAutoOptHookFun]] = None,
    demote_fields: Optional[list[str]] = None,
    validate: bool = True,
    validate_all: bool = False,
    **kwargs: Any,
) -> dace.SDFG:
    """Performs GT4Py specific optimizations on the SDFG in place.

    This function expects that the input SDFG follows the principles that are
    outlined in [ADR0018](https://github.com/GridTools/gt4py/tree/main/docs/development/ADRs/0018-Canonical_SDFG_in_GT4Py_Transformations.md).
    It is important to notice, that if `reuse_transients` is active then the
    optimized SDFG no longer conforms to these rules.

    The auto optimization works in different phases, that focuses each on
    different aspects of the SDFG. The initial SDFG is assumed to have a
    very large number of rather simple Maps.

    Note, because of how `gt_auto_optimizer()` works it is not save to call
    it twice on the same SDFG.

    A user should not rely on a particular order in which the SDFG is processed.
    However, in broad terms the function performs roughly the following steps:
    - An initial simplification is performed. This also includes the substitution
        of symbols whose value is known, see `constant_symbols`.
    - The top level Maps are optimized, the goal of this step is to increase
        the computational intensity. This is mostly, but not limited to fusing
        then together.
    - The interior dataflow of the top level Maps are optimized.
    - The Maps are configured. This involves setting their iteration order, such
        that the unit stride dimension is associated to the `x` block (GPU) or
        inner most loop-nest (CPU). But it also involves moving everything to GPU,
        if needed and setting the block size.
    - The transients are processed, this involves steps such as determine their
        strides or if they are persistently allocated.

    Args:
        sdfg: The SDFG that should be optimized in place.
        gpu: Optimize for GPU or CPU.
        unit_strides_kind: All dimensions of this kind are considered to have unit
            strides, see `gt_set_iteration_order()` for more.
        make_persistent: Turn all transients to persistent lifetime, thus they are
            allocated over the whole lifetime of the program, even if the kernel exits.
            Thus the SDFG can not be called by different threads.
        gpu_block_size: This is used as default thread block size for GPU Maps. See
            also the `gpu_block_size_*d` arguments
        gpu_block_size_{1, 2, 3}d: Allows to specify the GPU thread block size for
            1, 2 and 3 dimension Maps individually. See the `gpu_block_size_spec`
            argument of `gt_gpu_transformation()` for more.
        blocking_dim: On which dimension blocking should be applied.
        blocking_size: How many elements each block should process.
        blocking_only_if_independent_nodes: If `True`, the default, only apply loop
            blocking if there are independent nodes in the Map, see the
            `require_independent_nodes` option of the `LoopBlocking` transformation.
        disable_splitting: Disable the splitting transformations.
        reuse_transients: Run the `TransientReuse` transformation, might reduce memory footprint.
        gpu_launch_bounds: Use this value as `__launch_bounds__` for _all_ GPU Maps.
        gpu_launch_factor: Use the number of threads times this value as `__launch_bounds__`
            for _all_ GPU Maps.
        gpu_memory_pool: Enable CUDA memory pool in gpu codegen.
        constant_symbols: Symbols listed in this `dict` will be replaced by the
            respective value inside the SDFG. This might increase performance.
        assume_pointwise: Assume that the SDFG has no risk for race condition in
            global data access. See the `GT4PyMapBufferElimination` transformation for more.
        optimization_hooks: A `dict` containing the hooks that should be called,
            see `GT4PyAutoOptHook` for more information.
        demote_fields: Consider these fields as transients for the purpose of optimization.
            Use at your own risk. See Notes for all implications.
        validate: Perform validation during the steps.
        validate_all: Perform extensive validation.

    Note:
        - For identifying symbols that can be treated as compile time constants
            `gt_find_constant_arguments()` function can be used.
        - When `unit_strides_kind` is `None` the function assumes that on GPU the
            horizontal dimension has unit strides, while on CPU the vertical dimension
            has unit strides.
        - In case GPU optimizations are enabled, the function assumes that all
            global fields are already on the GPU.
        - When using the `demote_fields` feature the user has to ensure that the fields
            meet all constraints of a transient, see ADR18 for more. Furthermore, there
            is no guarantee whether the demoted field remains part of the SDFG's call
            signature or not.

    Todo:
        - Update the description. The Phases are nice, but they have lost their
            link to reality a little bit.
        - Improve the determination of the strides and iteration order of the
            transients.
        - Set padding of transients, i.e. alignment, the DaCe datadescriptor
            can do that.
        - Handle nested SDFGs better.
        - Specify arguments to set the size of GPU thread blocks depending on the
            dimensions. I.e. be able to use a different size for 1D than 2D Maps.
        - Implement some model to further guide to determine what we want to fuse.
            Something along the line "Fuse if operational intensity goes up, but
            not if we have too much internal space (register pressure).
    """
    device = dace.DeviceType.GPU if gpu else dace.DeviceType.CPU
    optimization_hooks = optimization_hooks or {}

    with dace.config.temporary_config():
        # Do not store which transformations were applied inside the SDFG.
        dace.Config.set("store_history", value=False)

        # If there is an exception during `can_be_applied()` propagate it instead
        #  of ignoring it.
        dace.Config.set("optimizer", "match_exception", value=True)

        # Do not assume that symbols are positive. The only reason for this is
        #  that the origin can potentially be negative.
        #  See also [issue#2095](https://github.com/spcl/dace/issues/2095)
        dace.Config.set("optimizer", "symbolic_positive", value=False)

        # Initial Cleanup
        # NOTE: The initial simplification stage must be synchronized with the one that
        #   `gt_substitute_compiletime_symbols()` performs!
        gtx_transformations.gt_simplify(
            sdfg=sdfg,
            validate=False,
            skip=gtx_transformations.constants._GT_AUTO_OPT_INITIAL_STEP_SIMPLIFY_SKIP_LIST,
            validate_all=validate_all,
        )

        if constant_symbols:
            gtx_transformations.gt_substitute_compiletime_symbols(
                sdfg=sdfg,
                repl=constant_symbols,
                simplify=True,  # Simplify again after.
                skip=gtx_transformations.constants._GT_AUTO_OPT_INITIAL_STEP_SIMPLIFY_SKIP_LIST,
                simplify_at_entry=False,
                validate=False,
                validate_all=validate_all,
            )

        # Demote the fields.
        #  Actually they should probably be at the very start of this function, however,
        #  they have to be after constant substitution in case the descriptor is modified.
        original_demoted_descriptors: dict[str, dace_data.Data] = {}
        if demote_fields is not None:
            for field_to_demote in demote_fields:
                if field_to_demote not in sdfg.arrays:
                    warnings.warn(
                        f"Requested the demotion of field '{field_to_demote}' but the field is unknown.",
                        stacklevel=0,
                    )
                    continue
                field_desc = sdfg.arrays[field_to_demote]

                if field_desc.transient:
                    warnings.warn(
                        f"Requested the demotion of field '{field_to_demote}' but the field is a transient.",
                        stacklevel=0,
                    )
                    continue

                # Demote the field but keep a copy of it such that we can restore it later.
                original_demoted_descriptors[field_to_demote] = field_desc.clone()
                field_desc.transient = True

            if len(original_demoted_descriptors) != 0:
                gtx_transformations.gt_simplify(
                    sdfg=sdfg,
                    validate=False,
                    skip=gtx_transformations.constants._GT_AUTO_OPT_INITIAL_STEP_SIMPLIFY_SKIP_LIST,
                    validate_all=validate_all,
                )

        gtx_transformations.gt_reduce_distributed_buffering(
            sdfg, validate=False, validate_all=validate_all
        )

        # Process top level Maps
        sdfg = _gt_auto_process_top_level_maps(
            sdfg=sdfg,
            assume_pointwise=assume_pointwise,
            disable_splitting=disable_splitting,
            optimization_hooks=optimization_hooks,
            validate_all=validate_all,
        )

        # We now ensure that point wise computations are properly double buffered,
        #  this ensures that rule 3 of ADR-18 is maintained.
        # TODO(phimuell): Figuring out if it is important to do it before the inner
        #   Map optimization. I think it is, especially when we apply `LoopBlocking`.
        gtx_transformations.gt_create_local_double_buffering(sdfg)

        # Optimize the interior of the Maps:
        sdfg = _gt_auto_process_dataflow_inside_maps(
            sdfg=sdfg,
            blocking_dim=blocking_dim,
            blocking_size=blocking_size,
            blocking_only_if_independent_nodes=blocking_only_if_independent_nodes,
            validate_all=validate_all,
        )

        # Configure the Maps:
        #  Will also perform the GPU transformation.
        # TODO(phimuell): Maybe switch it with the inside map optimization.
        gpu_block_size_spec: dict[str, Sequence[int | str] | str] = {}
        if gpu_block_size_1d is not None:
            gpu_block_size_spec["block_size_1d"] = gpu_block_size_1d
        if gpu_block_size_2d is not None:
            gpu_block_size_spec["block_size_2d"] = gpu_block_size_2d
        if gpu_block_size_3d is not None:
            gpu_block_size_spec["block_size_3d"] = gpu_block_size_3d

        sdfg = _gt_auto_configure_maps_and_strides(
            sdfg=sdfg,
            gpu=gpu,
            unit_strides_kind=unit_strides_kind,
            gpu_block_size=gpu_block_size,
            gpu_launch_factor=gpu_launch_factor,
            gpu_launch_bounds=gpu_launch_bounds,
            optimization_hooks=optimization_hooks,
            gpu_block_size_spec=gpu_block_size_spec if gpu_block_size_spec else None,
            validate_all=validate_all,
        )

        # Transients
        sdfg = _gt_auto_post_processing(
            sdfg=sdfg,
            gpu=gpu,
            make_persistent=make_persistent,
            # TODO(phimuell): In general `TransientReuse` is a good idea, but the
            #   current implementation also reuses transients scalars inside Map
            #   scopes, which I do not like. Thus we should fix the transformation
            #   to avoid that.
            reuse_transients=reuse_transients,
            gpu_memory_pool=gpu_memory_pool,
            validate_all=validate_all,
        )

        # Set the implementation of the library nodes.
        dace_aoptimize.set_fast_implementations(sdfg, device)

        # We now turn the demoted fields back into globals if possible, for ABI compatibility.
        for demoted_field, original_field_desc in original_demoted_descriptors.items():
            if demoted_field not in sdfg.arrays:
                # The demoted field is not inside the SDFG anymore, just insert it.
                sdfg.add_datadesc(demoted_field, original_field_desc)

            elif original_field_desc.is_equivalent(sdfg.arrays[demoted_field]) and all(
                (ostride == cstride) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
                for ostride, cstride in zip(
                    sdfg.arrays[demoted_field].strides, original_field_desc.strides
                )
            ):
                # The demoted field is still inside the SDFG, we only add it back if it
                #  still has the same layout as before.
                # NOTE: Technically we could ignore the strides, the problem are nested
                #   SDFG, where they might be mapped into.
                sdfg.arrays[demoted_field].transient = False

            else:
                warnings.warn(
                    f"Could not restore the demoted field '{demoted_field}' back to a global.",
                    stacklevel=0,
                )

        if validate:
            sdfg.validate()

        return sdfg


def _gt_auto_process_top_level_maps(
    sdfg: dace.SDFG,
    assume_pointwise: bool,
    optimization_hooks: dict[GT4PyAutoOptHook, GT4PyAutoOptHookFun],
    disable_splitting: bool,
    validate_all: bool,
) -> dace.SDFG:
    """Optimize the Maps at the top level of the SDFG inplace.

    In broad terms the function tries to reduce the number of of kernels/maps
    by fusing/inlining them, however, the function will also perform some other
    transformations, with the goal to promote high computational intensity and
    occupancy (on GPU).

    It is important to note that the fusion will only affect top level maps, i.e.
    everything whose scope is `None`. Some parts of the optimization process are
    iteratively.

    For a full description of the argument see the `gt_auto_optimize()` function.

    Note:
        The function assumes that `gt_simplify()` has been called on the SDFG
        before it is passed to this function.
    """

    # NOTE: Inside this function we have to disable the consolidation of edges.
    #   This is because it might block the application of `SpliAccessNode`. As
    #   an example consider a Map that writes `tmp[:, 80]` and a second Map that
    #   writes `tmp[:, 0]`, if these two Maps are now horizontally fussed and
    #   edge consolidation is on, then the resulting Map would "write", at least
    #   according to the subset, after Memlet propagation, into `tmp[:, 0:80]`.
    #   For that reason we block edge consolidation inside this function.
    #   However, we allow allow the consolidation, in MapFusion if this does
    #   not lead to an extension. This is because it causes some issues with
    #   MapFusion.
    # TODO(phimuell): Find a better way as blocking edge consolidation might
    #   limit what MapFusion can do.
    # TODO(phimuell): Maybe disable edge consolidation by default?

    # Compute the SDFG hash to see if something has changed.
    #  We use the hash instead of the return values of the transformation, because
    #  computing the hash invalidates some caches that are not properly updated in Dace.
    # TODO(phimuell): Remove this hack as soon as DaCe is fixed.
    # TODO(phimuell): Maybe switch to `reset_cfg_list()`?
    sdfg_hash = sdfg.hash_sdfg()

    if GT4PyAutoOptHook.TopLevelDataFlowPre in optimization_hooks:
        optimization_hooks[GT4PyAutoOptHook.TopLevelDataFlowPre](sdfg)  # type: ignore[call-arg]

    while True:
        # First we do scan the entire SDFG to figure out which data is only
        #  used once and can be deleted.
        find_single_use_data = dace_analysis.FindSingleUseData()
        single_use_data = find_single_use_data.apply_pass(sdfg, None)

        # TODO(phimuell): Use a cost measurement to decide if fusion should be done.

        vertical_map_fusion = gtx_transformations.MapFusionVertical(
            only_toplevel_maps=True,
            consolidate_edges_only_if_not_extending=True,
            check_fusion_callback=optimization_hooks.get(  # type: ignore[arg-type]
                GT4PyAutoOptHook.TopLevelDataFlowMapFusionVerticalCallBack, None
            ),
        )
        # TODO(phimuell): Remove that hack once [issue#1911](https://github.com/spcl/dace/issues/1911)
        #   has been solved.
        vertical_map_fusion._single_use_data = single_use_data

        sdfg.apply_transformations_repeated(
            vertical_map_fusion,
            validate=False,
            validate_all=validate_all,
        )

        # Promote Maps. This will remove transients between 1D and 2D Maps, at the
        #  cost of more data loads from memory. Empirical observations have shown
        #  that this is beneficial; especially for Nabla4-type kernel in conjunction
        #  with `LoopBlocking`.
        # NOTE: We have to promote before we do horizontal Map fusion.
        sdfg.apply_transformations_repeated(
            gtx_transformations.MapPromoter(
                only_toplevel_maps=True,
                promote_vertical=True,
                promote_horizontal=True,
                promote_local=False,
                promotion_callback=optimization_hooks.get(
                    GT4PyAutoOptHook.TopLevelDataFlowMapPromotionCallBack,
                    gt4py_default_auto_optimizer_map_promotion_checker,
                ),
                single_use_data=single_use_data,
            ),
            validate=False,
            validate_all=validate_all,
        )

        # Now run horizontal Map fusion, also run vertical Map fusion, because it
        #  might have become possible.
        # TODO(phimuell): Think of moving horizontal Map fusion after the splitting.
        horizontal_map_fusion = gtx_transformations.MapFusionHorizontal(
            only_toplevel_maps=True,
            consolidate_edges_only_if_not_extending=True,
            check_fusion_callback=optimization_hooks.get(  # type: ignore[arg-type]
                GT4PyAutoOptHook.TopLevelDataFlowMapFusionHorizontalCallBack, None
            ),
            # NOTE: Setting this argument to `False` has the side effect that Maps
            #   without any connection to each other are fused together and by this
            #   creating dependencies between otherwise unrelated Maps. However,
            #   also has the side effect of integrating Maps generated in
            #   `broadcast+concat_where` situations into other Maps. We should handle
            #   these cases through the splitting transformations.
            # TODO(phimuell): Update the other transformations such that we can
            #   set the falg to `True`.
            only_if_common_ancestor=False,
        )
        sdfg.apply_transformations_repeated(
            [horizontal_map_fusion, vertical_map_fusion],
            validate=False,
            validate_all=validate_all,
        )

        # Now do some cleanup task, that may enable further fusion opportunities.
        #  Note for performance reasons simplify is deferred.
        gtx_transformations.gt_reduce_distributed_buffering(
            sdfg, validate=False, validate_all=validate_all
        )

        if not disable_splitting:
            # TODO(phimuell): Find out how to skip the propagation and integrating it
            #   into the split transformation.
            sdfg.apply_transformations_repeated(
                [
                    gtx_transformations.MapSplitter(
                        single_use_data=single_use_data,
                        remove_dead_dataflow=True,
                    ),
                    gtx_transformations.SplitConsumerMemlet(single_use_data=single_use_data),
                ],
                validate=False,
                validate_all=validate_all,
            )
            # TODO(phimuell): Implement a data cleaner.
            dace_propagation.propagate_memlets_sdfg(sdfg)

            sdfg.apply_transformations_repeated(
                [
                    # TODO(phimuell): The transformation is also active inside Maps.
                    #   Which is against the description of this function, but it should
                    #   not matter that much.
                    gtx_transformations.SplitAccessNode(
                        single_use_data=single_use_data,
                    ),
                    gtx_transformations.GT4PyMapBufferElimination(
                        assume_pointwise=assume_pointwise,
                    ),
                ],
                validate=False,
                validate_all=validate_all,
            )

            # Call vertical and horizontal map fusion to fuse together maps on partially
            #  overlapping range. This is an iterative process that splits the maps to
            #  expose overlapping range and applies serial/parallel map fusion.
            gtx_transformations.gt_vertical_map_split_fusion(
                sdfg=sdfg,
                run_simplify=False,
                run_map_fusion=True,
                fuse_map_fragments=False,  # To avoid uncontrolled Map fusing.
                skip=gtx_transformations.constants._GT_AUTO_OPT_TOP_LEVEL_STAGE_SIMPLIFY_SKIP_LIST,
                consolidate_edges_only_if_not_extending=True,
                single_use_data=single_use_data,
                check_split_callback=optimization_hooks.get(  # type: ignore[arg-type]
                    GT4PyAutoOptHook.TopLevelDataFlowVerticalSplitCallBack, None
                ),
                check_fusion_callback=optimization_hooks.get(  # type: ignore[arg-type]
                    GT4PyAutoOptHook.TopLevelDataFlowMapFusionVerticalCallBack, None
                ),
                validate=False,
                validate_all=validate_all,
            )
            gtx_transformations.gt_horizontal_map_split_fusion(
                sdfg=sdfg,
                run_simplify=False,
                run_map_fusion=True,
                fuse_map_fragments=False,  # To avoid uncontrolled Map fusing.
                skip=gtx_transformations.constants._GT_AUTO_OPT_TOP_LEVEL_STAGE_SIMPLIFY_SKIP_LIST,
                consolidate_edges_only_if_not_extending=True,
                check_split_callback=optimization_hooks.get(  # type: ignore[arg-type]
                    GT4PyAutoOptHook.TopLevelDataFlowHorizontalSplitCallBack, None
                ),
                check_fusion_callback=optimization_hooks.get(  # type: ignore[arg-type]
                    GT4PyAutoOptHook.TopLevelDataFlowMapFusionHorizontalCallBack, None
                ),
                validate=False,
                validate_all=validate_all,
            )

        sdfg.apply_transformations_repeated(
            gtx_transformations.GT4PyMapBufferElimination(
                assume_pointwise=assume_pointwise,
            ),
            validate=False,
            validate_all=validate_all,
        )

        # TODO(phimuell): Figuring out if this is is the correct location for doing it.
        if GT4PyAutoOptHook.TopLevelDataFlowStep in optimization_hooks:
            optimization_hooks[GT4PyAutoOptHook.TopLevelDataFlowStep](sdfg)  # type: ignore[call-arg]

        # Determine if the SDFG has been modified by comparing the hash.
        old_sdfg_hash, sdfg_hash = sdfg_hash, sdfg.hash_sdfg()
        if old_sdfg_hash == sdfg_hash:
            break

        # The SDFG was modified by the transformations above. The SDFG was
        #  modified. Call Simplify and try again to further optimize.
        gtx_transformations.gt_simplify(
            sdfg,
            validate=False,
            validate_all=validate_all,
            skip=gtx_transformations.constants._GT_AUTO_OPT_TOP_LEVEL_STAGE_SIMPLIFY_SKIP_LIST,
        )

    if GT4PyAutoOptHook.TopLevelDataFlowPost in optimization_hooks:
        optimization_hooks[GT4PyAutoOptHook.TopLevelDataFlowPost](sdfg)  # type: ignore[call-arg]

    return sdfg


def _gt_auto_process_dataflow_inside_maps(
    sdfg: dace.SDFG,
    blocking_dim: Optional[gtx_common.Dimension],
    blocking_size: int,
    blocking_only_if_independent_nodes: Optional[bool],
    validate_all: bool,
) -> dace.SDFG:
    """Optimizes the dataflow inside the top level Maps of the SDFG inplace.

    For a description of the arguments see `gt_auto_optimize()`.

    It is important that nested Maps are not fused together. The main reason is that
    in most cases it is not beneficial and might prevent other optimization, as the
    fusion (especially the parallel version) will create dependency of otherwise
    unrelated dataflow. Furthermore, the majority, if not all, of these maps are
    over a constant range, e.g. the number of neighbours, which is known at compile
    time, so the compiler will fully unroll them anyway.
    """

    # Constants (tasklets are needed to write them into a variable) should not be
    #  arguments to a kernel but be present inside the body.
    sdfg.apply_transformations_once_everywhere(
        gtx_transformations.GT4PyMoveTaskletIntoMap,
        validate=False,
        validate_all=validate_all,
    )
    gtx_transformations.gt_simplify(
        sdfg,
        skip=gtx_transformations.constants._GT_AUTO_OPT_INNER_DATAFLOW_STAGE_SIMPLIFY_SKIP_LIST,
        validate=False,
        validate_all=validate_all,
    )

    # Blocking is performed first, because this ensures that as much as possible
    #  is moved into the k independent part.
    if blocking_dim is not None:
        sdfg.apply_transformations_once_everywhere(
            gtx_transformations.LoopBlocking(
                blocking_size=blocking_size,
                blocking_parameter=blocking_dim,
                require_independent_nodes=blocking_only_if_independent_nodes,
            ),
            validate=False,
            validate_all=validate_all,
        )

    # Move dataflow into the branches of the `if` such that they are only evaluated
    #  if they are needed. Important to call it repeatedly.
    # TODO(phimuell): It is unclear if `MoveDataflowIntoIfBody` should be called
    #   before or after `LoopBlocking`. In cases where the condition is `False`
    #   most of the times calling it before is better, but if the condition is
    #   `True` then this order is better. Solve that issue.
    sdfg.apply_transformations_repeated(
        gtx_transformations.MoveDataflowIntoIfBody(
            ignore_upstream_blocks=False,
        ),
        validate=False,
        validate_all=validate_all,
    )
    gtx_transformations.gt_simplify(
        sdfg,
        skip=gtx_transformations.constants._GT_AUTO_OPT_INNER_DATAFLOW_STAGE_SIMPLIFY_SKIP_LIST,
        validate=False,
        validate_all=validate_all,
    )

    # After some transformation we see in the SDFG that there are pointwise views
    # generated after reduction nodes. These views are unecessary and might produce
    # sub optimal GPU code thus we remove them.
    sdfg.apply_transformations_once_everywhere(
        gtx_transformations.RemovePointwiseViews,
        validate=False,
        validate_all=validate_all,
    )

    return sdfg


def _gt_auto_configure_maps_and_strides(
    sdfg: dace.SDFG,
    gpu: bool,
    unit_strides_kind: Optional[gtx_common.DimensionKind],
    gpu_block_size: Optional[Sequence[int | str] | str],
    gpu_launch_bounds: Optional[int | str],
    gpu_launch_factor: Optional[int],
    optimization_hooks: dict[GT4PyAutoOptHook, GT4PyAutoOptHookFun],
    gpu_block_size_spec: Optional[dict[str, Sequence[int | str] | str]],
    validate_all: bool,
) -> dace.SDFG:
    """Configure the Maps and the strides of the SDFG inplace.

    Essentially the function will set the properties of the Map accordingly, most
    importantly the iteration order of the loops, to ensure that the unit stride
    is associated to the `x` dimension of a block (GPU) or the inner most loop-nest.
    Furthermore, it will also apply the GPU transformation.
    It will also determine the strides of the transients.

    For a description of the arguments see the `gt_auto_optimize()` function.
    """

    # We now set the iteration order of the Maps. For that we use `unit_strides_kind`
    #  argument and if not supplied we guess depending if we are on the GPU or not.
    if unit_strides_kind is None:
        unit_strides_kind = (
            gtx_common.DimensionKind.HORIZONTAL if gpu else gtx_common.DimensionKind.VERTICAL
        )
    # It is not possible to use the `unit_strides_dim` argument of the
    #  function, because `LoopBlocking`, if run, changed the name of the
    #  parameter but the dimension can still be identified by its "kind".
    gtx_transformations.gt_set_iteration_order(
        sdfg=sdfg,
        unit_strides_kind=unit_strides_kind,
        validate=False,
        validate_all=validate_all,
    )

    # NOTE: We have to set the strides of transients before the non-standard Memlets
    #   get expanded, i.e. turned into Maps because no `cudaMemcpy*()` call exists,
    #   which requires that the final strides are there. Furthermore, Memlet expansion
    #   has to happen before the GPU block size is set. There are several possible
    #   solutions for that, of which none is really good. The one that is the least
    #   bad thing is to set the strides of the transients here. The main downside
    #   is that this and the `_gt_auto_post_processing()` function has these weird
    #   names.
    gtx_transformations.gt_change_transient_strides(sdfg, gpu=gpu)

    if gpu:
        # TODO(phimuell): The GPU function might modify the map iteration order.
        #   This is because how it is implemented (promotion and fusion). However,
        #   because of its current state, this should not happen, but we have to look
        #   into it.
        gtx_transformations.gt_gpu_transformation(
            sdfg,
            gpu_block_size=gpu_block_size,
            gpu_launch_bounds=gpu_launch_bounds,
            gpu_launch_factor=gpu_launch_factor,
            gpu_block_size_spec=gpu_block_size_spec,
            validate=False,
            validate_all=validate_all,
            try_removing_trivial_maps=True,
        )
        if GT4PyAutoOptHook.AfterToGPU in optimization_hooks:
            optimization_hooks[GT4PyAutoOptHook.AfterToGPU](sdfg)  # type: ignore[call-arg]

    return sdfg


def _gt_auto_post_processing(
    sdfg: dace.SDFG,
    gpu: bool,
    make_persistent: bool,
    reuse_transients: bool,
    gpu_memory_pool: bool,
    validate_all: bool,
) -> dace.SDFG:
    """Perform post processing on the SDFG.

    Apply the finishing touch to the optimized SDFG.
    For a full description of the arguments see the `gt_auto_optimize()` function.
    """

    if reuse_transients:
        # TODO(phimuell): Investigate if we should enable it, it may make things
        #   harder for the compiler. Maybe write our own to only consider big
        #   transients and not small ones (~60B)
        transient_reuse = dace.transformation.passes.TransientReuse()
        transient_reuse.apply_pass(sdfg, {})

    # TODO(phimuell): Fix the bug, it uses the tile value and not the stack array value.
    dace_aoptimize.move_small_arrays_to_stack(sdfg)

    if make_persistent and gpu_memory_pool:
        raise ValueError("Cannot set both 'make_persistent' and 'gpu_memory_pool'.")

    if make_persistent:
        device = dace.DeviceType.GPU if gpu else dace.DeviceType.CPU
        gtx_transformations.gt_make_transients_persistent(sdfg=sdfg, device=device)

        if device == dace.DeviceType.GPU:
            # NOTE: For unknown reasons the counterpart of the
            #   `gt_make_transients_persistent()` function in DaCe, resets the
            #   `wcr_nonatomic` property of every memlet, i.e. makes it atomic.
            #   However, it does this only for edges on the top level and on GPU.
            #   For compatibility with DaCe (and until we found out why) the GT4Py
            #   auto optimizer will emulate this behaviour.
            for state in sdfg.states():
                assert isinstance(state, dace.SDFGState)
                for edge in state.edges():
                    edge.data.wcr_nonatomic = False

    if gpu and gpu_memory_pool:
        gtx_transformations.gpu_utils.gt_gpu_apply_mempool(sdfg)

    if validate_all:
        sdfg.validate()

    return sdfg


def gt4py_default_auto_optimizer_map_promotion_checker(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    first_map_exit: dace_nodes.MapExit,
    second_map_entry: dace_nodes.MapEntry,
    missing_map_parameters: list[str],
) -> bool:
    """Helper function to check if horizontal Map promotion should be performed.

    The function is used by `_gt_auto_process_top_level_maps()` to ensure that
    only horizontal (the function will not check any other promotion) promotions
    that are beneficial will be accepted.

    The implementation checks if the complexity of the first map (the one that
    lacks the horizontal dimension) is not too large, such that inlining is
    beneficial. The rules, which are empirical, are summarized as follows:
    - The number of nodes inside the Map scope is below a certain value (currently
        20 nodes).
    - The Map only contain Tasklets and AccessNodes refering to transient data.
    - There are no WCR edges.

    For arguments see the `MapPromotionCallBack`.
    """

    # We only apply the rule to horizontal promotion. Vertical promotion is known
    #  to be beneficial, especially in conjunction with `LoopBloking`.
    if not any(
        missing_param.endswith("_gtx_horizontal") for missing_param in missing_map_parameters
    ):
        return True

    # NOTE: Inclusion of the exit is important that we get the edges that
    #   actually write the result.
    first_map_entry = state.entry_node(first_map_exit)
    first_map_scope = state.scope_subgraph(first_map_entry, include_entry=True, include_exit=True)

    # If there are more than 20 nodes we assume that the promotion is not
    #  beneficial. This is an arbitrary value.
    if first_map_scope.number_of_nodes() > 22:
        return False

    for node in first_map_scope.nodes():
        if node in (first_map_entry, first_map_exit):
            continue

        # All AccessNodes inside must be transient scalars. Because everything else
        #  indicate (empirical consideration) a more complex computation.
        if isinstance(node, dace_nodes.AccessNode):
            desc = node.desc(sdfg)
            if (not isinstance(desc, dace_data.Scalar)) and (not desc.transient):
                return False
            continue

        # A tasklet is considered to be a cheap computation.
        if isinstance(node, dace_nodes.Tasklet):
            continue

        # Everything else is considered to be a complex computation,
        #  and thus the Map should not be promoted.
        return False

    for edge in first_map_scope.edges():
        # If we have WCR then we have complex computations, most likely
        #   reductions, thus we will refuse the fusing.
        if edge.data.wcr is not None:
            return False

    return True
