# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Fast access to the auto optimization on DaCe."""

from typing import Any, Optional, Sequence

import dace
from dace.transformation.auto import auto_optimize as dace_aoptimize
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


def gt_auto_optimize(
    sdfg: dace.SDFG,
    gpu: bool,
    unit_strides_kind: Optional[gtx_common.DimensionKind] = None,
    make_persistent: bool = False,
    gpu_block_size: Optional[Sequence[int | str] | str] = None,
    blocking_dim: Optional[gtx_common.Dimension] = None,
    blocking_size: int = 10,
    blocking_only_if_independent_nodes: Optional[bool] = None,
    reuse_transients: bool = False,
    gpu_launch_bounds: Optional[int | str] = None,
    gpu_launch_factor: Optional[int] = None,
    constant_symbols: Optional[dict[str, Any]] = None,
    assume_pointwise: bool = True,
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
        gpu_block_size: The thread block size for maps in GPU mode, currently only
            one for all.
        blocking_dim: On which dimension blocking should be applied.
        blocking_size: How many elements each block should process.
        blocking_only_if_independent_nodes: If `True` only apply loop blocking if
            there are independent nodes in the Map, see the `require_independent_nodes`
            option of the `LoopBlocking` transformation.
        reuse_transients: Run the `TransientReuse` transformation, might reduce memory footprint.
        gpu_launch_bounds: Use this value as `__launch_bounds__` for _all_ GPU Maps.
        gpu_launch_factor: Use the number of threads times this value as `__launch_bounds__`
            for _all_ GPU Maps.
        constant_symbols: Symbols listed in this `dict` will be replaced by the
            respective value inside the SDFG. This might increase performance.
        assume_pointwise: Assume that the SDFG has no risk for race condition in
            global data access. See the `GT4PyMapBufferElimination` transformation for more.
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

    with dace.config.temporary_config():
        dace.Config.set("optimizer", "match_exception", value=True)
        dace.Config.set("store_history", value=False)

        # Initial Cleanup
        if constant_symbols:
            gtx_transformations.gt_substitute_compiletime_symbols(
                sdfg=sdfg,
                repl=constant_symbols,
                simplify=False,
                validate=validate,
                validate_all=validate_all,
            )
        gtx_transformations.gt_simplify(sdfg)
        gtx_transformations.gt_reduce_distributed_buffering(sdfg)

        # Process top level Maps
        sdfg = _gt_auto_process_top_level_maps(
            sdfg=sdfg,
            assume_pointwise=assume_pointwise,
            validate=validate,
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
            validate=validate,
            validate_all=validate_all,
        )

        # Configure the Maps:
        #  Will also perform the GPU transformation.
        sdfg = _gt_auto_configure_maps_and_strides(
            sdfg=sdfg,
            gpu=gpu,
            unit_strides_kind=unit_strides_kind,
            gpu_block_size=gpu_block_size,
            gpu_launch_factor=gpu_launch_factor,
            gpu_launch_bounds=gpu_launch_bounds,
            validate=validate,
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
            validate=validate,
            validate_all=validate_all,
        )

        # Set the implementation of the library nodes.
        dace_aoptimize.set_fast_implementations(sdfg, device)

        return sdfg


def _gt_auto_process_top_level_maps(
    sdfg: dace.SDFG,
    assume_pointwise: bool,
    validate: bool,
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
    # Compute the SDFG hash to see if something has changed.
    #  We use the hash instead of the return values of the transformation, because
    #  computing the hash invalidates some caches that are not properly updated in Dace.
    # TODO(phimuell): Remove this hack as soon as DaCe is fixed.
    sdfg_hash = sdfg.hash_sdfg()

    while True:
        # First we do scan the entire SDFG to figure out which data is only
        #  used once and can be deleted. MapFusion could do this on its own but
        #  it is more efficient to do it once and then reuse it.
        find_single_use_data = dace_analysis.FindSingleUseData()
        single_use_data = find_single_use_data.apply_pass(sdfg, None)

        # TODO(phimuell): Switch to `FullMapFusion` once DaCe has
        #   [parallel map fusion](https://github.com/spcl/dace/pull/1965).
        # TODO(phimuell): Use a cost measurement to decide if fusion should be done.
        # TODO(phimuell): Should we restrict parallel fusing to Maps that have
        #   common data?
        fusion_transformation = gtx_transformations.MapFusion(
            only_toplevel_maps=True,
            allow_parallel_map_fusion=True,
            allow_serial_map_fusion=True,
            only_if_common_ancestor=False,
        )
        # TODO(phimuell): Remove that hack once [issue#1911](https://github.com/spcl/dace/issues/1911)
        #   has been solved.
        fusion_transformation._single_use_data = single_use_data

        sdfg.apply_transformations_repeated(
            fusion_transformation,
            validate=validate,
            validate_all=validate_all,
        )

        # Now do some cleanup task, that may enable further fusion opportunities.
        #  Note for performance reasons simplify is deferred.
        cleanup_stages = [
            gtx_transformations.SplitAccessNode(
                single_use_data=single_use_data,
            ),
            gtx_transformations.GT4PyMapBufferElimination(
                assume_pointwise=assume_pointwise,
            ),
            # TODO(phimuell): Add a criteria to decide if we should promote or not.
            gtx_transformations.SerialMapPromoter(
                only_toplevel_maps=True,
                promote_vertical=True,
                promote_horizontal=False,
                promote_local=False,
            ),
        ]

        # Perform the clean up.
        gtx_transformations.gt_reduce_distributed_buffering(sdfg)
        sdfg.apply_transformations_repeated(
            cleanup_stages,
            validate=validate,
            validate_all=validate_all,
        )

        # Determine if the SDFG has been modified by comparing the hash.
        old_sdfg_hash, sdfg_hash = sdfg_hash, sdfg.hash_sdfg()
        if old_sdfg_hash == sdfg_hash:
            break

        # The SDFG was modified by the transformations above. The SDFG was
        #  modified. Call Simplify and try again to further optimize.
        gtx_transformations.gt_simplify(sdfg, validate=validate, validate_all=validate_all)

    return sdfg


def _gt_auto_process_dataflow_inside_maps(
    sdfg: dace.SDFG,
    blocking_dim: Optional[gtx_common.Dimension],
    blocking_size: int,
    blocking_only_if_independent_nodes: Optional[bool],
    validate: bool,
    validate_all: bool,
) -> dace.SDFG:
    """Optimizes the dataflow inside the top level Maps of teh SDFG inplace.

    For a description of the arguments see `gt_auto_optimize()`.
    """

    # Constants should bot be arguments to a kernel but be present inside the body.
    sdfg.apply_transformations_once_everywhere(
        gtx_transformations.GT4PyMoveTaskletIntoMap,
        validate=validate,
        validate_all=validate_all,
    )
    gtx_transformations.gt_simplify(sdfg, validate=validate, validate_all=validate_all)

    # Blocking is performed first, because this ensures that as much as possible
    #  is moved into the k independent part.
    if blocking_dim is not None:
        sdfg.apply_transformations_once_everywhere(
            gtx_transformations.LoopBlocking(
                blocking_size=blocking_size,
                blocking_parameter=blocking_dim,
                require_independent_nodes=blocking_only_if_independent_nodes,
            ),
            validate=validate,
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
        validate=validate,
        validate_all=validate_all,
    )

    fusion_transformation = gtx_transformations.MapFusion(
        only_toplevel_maps=False,
        only_inner_maps=True,  # TODO(phimuell): What about Maps in nested SDFGs?
        allow_parallel_map_fusion=True,
        allow_serial_map_fusion=True,
        only_if_common_ancestor=False,  # TODO(phimuell): Should we?
    )
    # TODO(phimuell): Remove that hack once [issue#1911](https://github.com/spcl/dace/issues/1911)
    #   has been solved.
    fusion_transformation._single_use_data = dace_analysis.FindSingleUseData().apply_pass(sdfg, {})

    while True:
        nb_applied = sdfg.apply_transformations_repeated(
            fusion_transformation,
            validate=validate,
            validate_all=validate_all,
        )
        if nb_applied == 0:
            break
        gtx_transformations.gt_simplify(sdfg, validate=validate, validate_all=validate_all)

    return sdfg


def _gt_auto_configure_maps_and_strides(
    sdfg: dace.SDFG,
    gpu: bool,
    unit_strides_kind: Optional[gtx_common.DimensionKind],
    gpu_block_size: Optional[Sequence[int | str] | str],
    gpu_launch_bounds: Optional[int | str],
    gpu_launch_factor: Optional[int],
    validate: bool,
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

    if unit_strides_kind is None:
        unit_strides_kind = (
            gtx_common.DimensionKind.HORIZONTAL if gpu else gtx_common.DimensionKind.VERTICAL
        )
    if unit_strides_kind is not None:
        # It is not possible to use the `unit_strides_dim` argument of the
        #  function, because `LoopBlocking` will change the name of the parameter
        #  but the dimension can still be identified by its "kind".
        gtx_transformations.gt_set_iteration_order(
            sdfg=sdfg,
            unit_strides_kind=unit_strides_kind,
            validate=validate,
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
            validate=validate,
            validate_all=validate_all,
            try_removing_trivial_maps=True,
        )

    return sdfg


def _gt_auto_post_processing(
    sdfg: dace.SDFG,
    gpu: bool,
    make_persistent: bool,
    reuse_transients: bool,
    validate: bool,
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

    return sdfg
