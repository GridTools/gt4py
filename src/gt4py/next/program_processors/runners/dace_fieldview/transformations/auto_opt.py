# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Fast access to the auto optimization on DaCe."""

from typing import Any, Final, Iterable, Optional, Sequence

import dace
from dace.transformation import dataflow as dace_dataflow, passes as dace_passes
from dace.transformation.auto import auto_optimize as dace_aoptimize

from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)


GT_SIMPLIFY_DEFAULT_SKIP_SET: Final[set[str]] = {"ScalarToSymbolPromotion", "ConstantPropagation"}
"""Set of simplify passes `gt_simplify()` skips by default.

The following passes are included:
- `ScalarToSymbolPromotion`: The lowering has sometimes to turn a scalar into a
    symbol or vice versa and at a later point to invert this again. However, this
    pass has some problems with this pattern so for the time being it is disabled.
- `ConstantPropagation`: Same reasons as `ScalarToSymbolPromotion`.
"""


def gt_simplify(
    sdfg: dace.SDFG,
    validate: bool = True,
    validate_all: bool = False,
    skip: Optional[Iterable[str]] = None,
) -> Any:
    """Performs simplifications on the SDFG in place.

    Instead of calling `sdfg.simplify()` directly, you should use this function,
    as it is specially tuned for GridTool based SDFGs.

    This function runs the DaCe simplification pass, but the following passes are
    replaced:
    - `InlineSDFGs`: Instead `gt_inline_nested_sdfg()` will be called.

    Furthermore, by default, or if `None` is passed fro `skip` the passes listed in
    `GT_SIMPLIFY_DEFAULT_SKIP_SET` will be skipped.

    Args:
        sdfg: The SDFG to optimize.
        validate: Perform validation after the pass has run.
        validate_all: Perform extensive validation.
        skip: List of simplify passes that should not be applied, defaults
            to `GT_SIMPLIFY_DEFAULT_SKIP_SET`.
    """
    # Ensure that `skip` is a `set`
    skip = GT_SIMPLIFY_DEFAULT_SKIP_SET if skip is None else set(skip)

    if "InlineSDFGs" not in skip:
        gt_inline_nested_sdfg(
            sdfg=sdfg,
            multistate=True,
            permissive=False,
            validate=validate,
            validate_all=validate_all,
        )

    return dace_passes.SimplifyPass(
        validate=validate,
        validate_all=validate_all,
        verbose=False,
        skip=(skip | {"InlineSDFGs"}),
    ).apply_pass(sdfg, {})


def gt_set_iteration_order(
    sdfg: dace.SDFG,
    leading_dim: gtx_common.Dimension,
    validate: bool = True,
    validate_all: bool = False,
) -> Any:
    """Set the iteration order of the Maps correctly.

    Modifies the order of the Map parameters such that `leading_dim`
    is the fastest varying one, the order of the other dimensions in
    a Map is unspecific. `leading_dim` should be the dimensions were
    the stride is one.

    Args:
        sdfg: The SDFG to process.
        leading_dim: The leading dimensions.
        validate: Perform validation during the steps.
        validate_all: Perform extensive validation.
    """
    return sdfg.apply_transformations_once_everywhere(
        gtx_transformations.MapIterationOrder(
            leading_dim=leading_dim,
        ),
        validate=validate,
        validate_all=validate_all,
    )


def gt_inline_nested_sdfg(
    sdfg: dace.SDFG,
    multistate: bool = True,
    permissive: bool = False,
    validate: bool = True,
    validate_all: bool = False,
) -> dace.SDFG:
    """Perform inlining of nested SDFG into their parent SDFG.

    The function uses DaCe's `InlineSDFG` transformation, the same used in simplify.
    However, before the inline transformation is run the function will run some
    cleaning passes that allows inlining nested SDFGs.
    As a side effect, the function will split stages into more states.

    Args:
        sdfg: The SDFG that should be processed, will be modified in place and returned.
        multistate: Allow inlining of multistate nested SDFG, defaults to `True`.
        permissive: Be less strict on the accepted SDFGs.
        validate: Perform validation after the transformation has finished.
        validate_all: Performs extensive validation.
    """
    first_iteration = True
    i = 0
    while True:
        print(f"ITERATION: {i}")
        nb_preproccess = sdfg.apply_transformations_repeated(
            [dace_dataflow.PruneSymbols, dace_dataflow.PruneConnectors],
            validate=False,
            validate_all=validate_all,
        )
        if (nb_preproccess == 0) and (not first_iteration):
            break

        # Create and configure the inline pass
        inline_sdfg = dace_passes.InlineSDFGs()
        inline_sdfg.progress = False
        inline_sdfg.permissive = permissive
        inline_sdfg.multistate = multistate

        # Apply the inline pass
        nb_inlines = inline_sdfg.apply_pass(sdfg, {})

        # Check result, if needed and test if we can stop
        if validate_all or validate:
            sdfg.validate()
        if nb_inlines == 0:
            break
        first_iteration = False

    return sdfg


def gt_auto_optimize(
    sdfg: dace.SDFG,
    gpu: bool,
    leading_dim: Optional[gtx_common.Dimension] = None,
    aggressive_fusion: bool = True,
    max_optimization_rounds_p2: int = 100,
    make_persistent: bool = True,
    gpu_block_size: Optional[Sequence[int | str] | str] = None,
    blocking_dim: Optional[gtx_common.Dimension] = None,
    blocking_size: int = 10,
    reuse_transients: bool = False,
    gpu_launch_bounds: Optional[int | str] = None,
    gpu_launch_factor: Optional[int] = None,
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

    1. Some general simplification transformations, beyond classical simplify,
        are applied to the SDFG.
    2. Tries to create larger kernels by fusing smaller ones, see
        `gt_auto_fuse_top_level_maps()` for more details.
    3. After the function created big kernels/maps it will apply some optimization,
        inside the kernels itself. For example fuse maps inside them.
    4. Afterwards it will process the map ranges and iteration order. For this
        the function assumes that the dimension indicated by `leading_dim` is the
        one with stride one.
    5. If requested the function will now apply loop blocking, on the dimension
        indicated by `leading_dim`.
    6. If requested the SDFG will be transformed to GPU. For this the
        `gt_gpu_transformation()` function is used, that might apply several other
        optimizations.
    7. Afterwards some general transformations to the SDFG are applied.
        This includes:
        - Use fast implementation for library nodes.
        - Move small transients to stack.
        - Make transients persistent (if requested).
        - Apply DaCe's `TransientReuse` transformation (if requested).

    Args:
        sdfg: The SDFG that should be optimized in place.
        gpu: Optimize for GPU or CPU.
        leading_dim: Leading dimension, indicates where the stride is 1.
        aggressive_fusion: Be more aggressive during phase 2, will lead to the promotion
            of certain maps (over computation) but will lead to larger kernels.
        max_optimization_rounds_p2: Maximum number of optimization rounds in phase 2.
        make_persistent: Turn all transients to persistent lifetime, thus they are
            allocated over the whole lifetime of the program, even if the kernel exits.
            Thus the SDFG can not be called by different threads.
        gpu_block_size: The thread block size for maps in GPU mode, currently only
            one for all.
        blocking_dim: On which dimension blocking should be applied.
        blocking_size: How many elements each block should process.
        reuse_transients: Run the `TransientReuse` transformation, might reduce memory footprint.
        gpu_launch_bounds: Use this value as `__launch_bounds__` for _all_ GPU Maps.
        gpu_launch_factor: Use the number of threads times this value as `__launch_bounds__`
            for _all_ GPU Maps.
        validate: Perform validation during the steps.
        validate_all: Perform extensive validation.

    Todo:
        - Make sure that `SDFG.simplify()` is not called indirectly, by temporarily
            overwriting it with `gt_simplify()`.
        - Specify arguments to set the size of GPU thread blocks depending on the
            dimensions. I.e. be able to use a different size for 1D than 2D Maps.
        - Add a parallel version of Map fusion.
        - Implement some model to further guide to determine what we want to fuse.
            Something along the line "Fuse if operational intensity goes up, but
            not if we have too much internal space (register pressure).
        - Create a custom array elimination pass that honors rule 1.
        - Check if a pipeline could be used to speed up some computations.
    """
    device = dace.DeviceType.GPU if gpu else dace.DeviceType.CPU

    with dace.config.temporary_config():
        dace.Config.set("optimizer", "match_exception", value=True)
        dace.Config.set("store_history", value=False)

        # TODO(phimuell): Should there be a zeroth phase, in which we generate
        #       a chanonical form of the SDFG, for example move all local maps
        #       to internal serial maps, such that they do not block fusion?

        # Phase 1: Initial Cleanup
        gt_simplify(
            sdfg=sdfg,
            validate=validate,
            validate_all=validate_all,
        )
        sdfg.apply_transformations_repeated(
            [
                dace_dataflow.TrivialMapElimination,
                # TODO(phimuell): The transformation are interesting, but they have
                #  a bug as they assume that they are not working inside a map scope.
                #  Before we use them we have to fix them.
                #  https://chat.spcl.inf.ethz.ch/spcl/pl/8mtgtqjb378hfy7h9a96sy3nhc
                # dace_dataflow.MapReduceFusion,
                # dace_dataflow.MapWCRFusion,
            ],
            validate=validate,
            validate_all=validate_all,
        )

        # Phase 2: Kernel Creation
        #   Try to create kernels as large as possible.
        sdfg = gt_auto_fuse_top_level_maps(
            sdfg=sdfg,
            aggressive_fusion=aggressive_fusion,
            max_optimization_rounds=max_optimization_rounds_p2,
            validate=validate,
            validate_all=validate_all,
        )

        # Phase 3: Optimizing the kernels, i.e. the larger maps, themselves.
        #   Currently this only applies fusion inside Maps.
        sdfg.apply_transformations_repeated(
            gtx_transformations.SerialMapFusion(
                only_inner_maps=True,
            ),
            validate=validate,
            validate_all=validate_all,
        )
        gt_simplify(sdfg)

        # Phase 4: Iteration Space
        #   This essentially ensures that the stride 1 dimensions are handled
        #   by the inner most loop nest (CPU) or x-block (GPU)
        if leading_dim is not None:
            gt_set_iteration_order(
                sdfg=sdfg,
                leading_dim=leading_dim,
                validate=validate,
                validate_all=validate_all,
            )

        # Phase 5: Apply blocking
        if blocking_dim is not None:
            sdfg.apply_transformations_once_everywhere(
                gtx_transformations.LoopBlocking(
                    blocking_size=blocking_size,
                    blocking_parameter=blocking_dim,
                ),
                validate=validate,
                validate_all=validate_all,
            )

        # Phase 6: Going to GPU
        if gpu:
            # TODO(phimuell): The GPU function might modify the map iteration order.
            #                   This is because how it is implemented (promotion and
            #                   fusion). However, because of its current state, this
            #                   should not happen, but we have to look into it.
            gtx_transformations.gt_gpu_transformation(
                sdfg,
                gpu_block_size=gpu_block_size,
                gpu_launch_bounds=gpu_launch_bounds,
                gpu_launch_factor=gpu_launch_factor,
                validate=validate,
                validate_all=validate_all,
                try_removing_trivial_maps=True,
            )

        # Phase 7: General Optimizations
        #   The following operations apply regardless if we have a GPU or CPU.
        #   The DaCe auto optimizer also uses them. Note that the reuse transient
        #   is not done by DaCe.
        if reuse_transients:
            # TODO(phimuell): Investigate if we should enable it, it may make things
            #                   harder for the compiler. Maybe write our own to
            #                   only consider big transients and not small ones (~60B)
            transient_reuse = dace.transformation.passes.TransientReuse()
            transient_reuse.apply_pass(sdfg, {})

        # Set the implementation of the library nodes.
        dace_aoptimize.set_fast_implementations(sdfg, device)
        # TODO(phimuell): Fix the bug, it uses the tile value and not the stack array value.
        dace_aoptimize.move_small_arrays_to_stack(sdfg)
        if make_persistent:
            # TODO(phimuell): Allow to also to set the lifetime to `SDFG`.
            dace_aoptimize.make_transients_persistent(sdfg, device)

        return sdfg


def gt_auto_fuse_top_level_maps(
    sdfg: dace.SDFG,
    aggressive_fusion: bool = True,
    max_optimization_rounds: int = 100,
    validate: bool = True,
    validate_all: bool = False,
) -> dace.SDFG:
    """Performs the second phase of the auto optimization process.

    As noted in the doc of `gt_auto_optimize()` the function tries to reduce the
    number of kernels/maps by fusing maps. This process essentially builds on
    the map fusion transformations and some clean up transformations.

    It is important to note that the fusion will only affect top level maps, i.e.
    nested maps are ignored. Furthermore, the function will iteratively perform
    optimizations until a fix point is reached. If this does not happen within
    `max_optimization_rounds` iterations an error is generated.

    Return:
        The function optimizes the SDFG in place and returns it.

    Args:
        sdfg: The SDFG to optimize.
        aggressive_fusion: allow more aggressive fusion by promoting maps (over
            computing).
        max_optimization_rounds: Maximal number of optimization rounds should be
            performed.
        validate: Perform validation during the steps.
        validate_all: Perform extensive validation.

    Note:
        Calling this function directly is most likely an error. Instead you should
        call `gt_auto_optimize()` directly.
    """
    # Compute the SDFG hash to see if something has changed.
    sdfg_hash = sdfg.hash_sdfg()

    # We use a loop to optimize because we are using multiple transformations
    #  after the other, thus new opportunities might arise in the next round.
    #  We use the hash of the SDFG to detect if we have reached a fix point.
    for _ in range(max_optimization_rounds):
        # Use map fusion to reduce their number and to create big kernels
        # TODO(phimuell): Use a cost measurement to decide if fusion should be done.
        # TODO(phimuell): Add parallel fusion transformation. Should it run after
        #                   or with the serial one?
        sdfg.apply_transformations_repeated(
            gtx_transformations.SerialMapFusion(
                only_toplevel_maps=True,
            ),
            validate=validate,
            validate_all=validate_all,
        )

        # Now do some cleanup task, that may enable further fusion opportunities.
        #  Note for performance reasons simplify is deferred.
        phase2_cleanup = []
        phase2_cleanup.append(dace_dataflow.TrivialTaskletElimination())

        # If requested perform map promotion, this will lead to more fusion.
        if aggressive_fusion:
            # TODO(phimuell): Should we do this all the time or only once?
            # TODO(phimuell): Add a criteria to decide if we should promote or not.
            # TODO(phimuell): Add parallel map promotion?
            phase2_cleanup.append(
                gtx_transformations.SerialMapPromoter(
                    only_toplevel_maps=True,
                    promote_vertical=True,
                    promote_horizontal=False,
                    promote_local=False,
                )
            )

        # Perform the phase 2 cleanup.
        sdfg.apply_transformations_once_everywhere(
            phase2_cleanup,
            validate=validate,
            validate_all=validate_all,
        )

        # Use the hash to determine if the transformations did modify the SDFG.
        #  If not we have optimized the SDFG as much as we could, in this phase.
        old_sdfg_hash = sdfg_hash
        sdfg_hash = sdfg.hash_sdfg()
        if old_sdfg_hash == sdfg_hash:
            break

        # The SDFG was modified by the transformations above. The SDFG was
        #  modified. Call Simplify and try again to further optimize.
        gt_simplify(sdfg, validate=validate, validate_all=validate_all)

    else:
        raise RuntimeWarning("Optimization of the SDFG did not converge.")

    return sdfg
