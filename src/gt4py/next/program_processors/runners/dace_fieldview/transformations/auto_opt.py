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

"""Fast access to the auto optimization on DaCe."""

from typing import Any, Optional, Sequence

import dace
from dace.transformation import dataflow as dace_dataflow
from dace.transformation.auto import auto_optimize as dace_aoptimize

from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)


def dace_auto_optimize(
    sdfg: dace.SDFG,
    device: dace.DeviceType = dace.DeviceType.CPU,
    use_gpu_storage: bool = True,
    **kwargs: Any,
) -> dace.SDFG:
    """This is a convenient wrapper arround DaCe's `auto_optimize` function.

    Args:
        sdfg: The SDFG that should be optimized in place.
        device: the device for which optimizations should be done, defaults to CPU.
        use_gpu_storage: Assumes that the SDFG input is already on the GPU.
            This parameter is `False` in DaCe but here is changed to `True`.
        kwargs: Are forwarded to the underlying auto optimized exposed by DaCe.
    """
    return dace_aoptimize.auto_optimize(
        sdfg,
        device=device,
        use_gpu_storage=use_gpu_storage,
        **kwargs,
    )


def gt_simplify(
    sdfg: dace.SDFG,
    validate: bool = True,
    validate_all: bool = False,
    skip: Optional[set[str]] = None,
) -> Any:
    """Performs simplifications on the SDFG in place.

    Instead of calling `sdfg.simplify()` directly, you should use this function,
    as it is specially tuned for GridTool based SDFGs.

    Args:
        sdfg: The SDFG to optimize.
        validate: Perform validation after the pass has run.
        validate_all: Perform extensive validation.
        skip: List of simplify passes that should not be applied.

    Note:
        The reason for this function is that we can influence how simplify works.
        Since some parts in simplify might break things in the SDFG.
        However, currently nothing is customized yet, and the function just calls
        the simplification pass directly.
    """
    from dace.transformation.passes.simplify import SimplifyPass

    return SimplifyPass(
        validate=validate,
        validate_all=validate_all,
        verbose=False,
        skip=skip,
    ).apply_pass(sdfg, {})


def gt_auto_optimize(
    sdfg: dace.SDFG,
    gpu: bool,
    leading_dim: Optional[gtx_common.Dimension] = None,
    aggressive_fusion: bool = True,
    make_persistent: bool = True,
    gpu_block_size: Optional[Sequence[int | str] | str] = None,
    block_dim: Optional[gtx_common.Dimension] = None,
    blocking_size: int = 10,
    reuse_transients: bool = False,
    validate: bool = True,
    validate_all: bool = False,
    **kwargs: Any,
) -> dace.SDFG:
    """Performs GT4Py specific optimizations on the SDFG in place.

    The auto optimization works in different phases, that focuses each on
    different aspects of the SDFG. The initial SDFG is assumed to have a
    very large number of rather simple Maps.

    1. Some general simplification transformations, beyond classical simplify,
        are applied to the SDFG.
    2. In this phase the function tries to reduce the number of maps. This
        process mostly relies on the map fusion transformation. If
        `aggressive_fusion` is set the function will also promote certain Maps, to
        make them fusable. For this it will add dummy dimensions. However, currently
        the function will only add horizonal dimensions.
        In this phase some optimizations inside the bigger kernels themselves might
        be applied as well.
    3. After the function created big kernels it will apply some optimization,
        inside the kernels itself. For example fuse maps inside them.
    4. Afterwards it will process the map ranges and iteration order. For this
        the function assumes that the dimension indicated by `leading_dim` is the
        one with stride one.
    5. If requested the function will now apply blocking, on the dimension indicated
        by `leading_dim`. (The reason that it is not done in the kernel optimization
        phase is a restriction dictated by the implementation.)
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
        aggressive_fusion: Be more aggressive in fusion, will lead to the promotion
            of certain maps.
        make_persistent: Turn all transients to persistent lifetime, thus they are
            allocated over the whole lifetime of the program, even if the kernel exits.
            Thus the SDFG can not be called by different threads.
        gpu_block_size: The thread block size for maps in GPU mode, currently only
            one for all.
        block_dim: On which dimension blocking should be applied.
        blocking_size: How many elements each block should process.
        reuse_transients: Run the `TransientReuse` transformation, might reduce memory footprint.
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
    """
    device = dace.DeviceType.GPU if gpu else dace.DeviceType.CPU

    with dace.config.temporary_config():
        dace.Config.set("optimizer", "match_exception", value=True)
        dace.Config.set("store_history", value=False)

        # TODO(phimuell): Should there be a zeroth phase, in which we generate
        #       a chanonical form of the SDFG, for example move all local maps
        #       to internal serial maps, such that they do not block fusion?

        # Phase 1: Initial Cleanup
        gt_simplify(sdfg)
        sdfg.apply_transformations_repeated(
            [
                dace_dataflow.TrivialMapElimination,
                # TODO(phimuell): Investigate if these two are appropriate.
                dace_dataflow.MapReduceFusion,
                dace_dataflow.MapWCRFusion,
            ],
            validate=validate,
            validate_all=validate_all,
        )

        # Compute the SDFG hash to see if something has changed.
        sdfg_hash = sdfg.hash_sdfg()

        # Phase 2: Kernel Creation
        #   We will now try to reduce the number of kernels and create large Maps/kernels.
        #   For this we essentially use Map fusion. We do this is a loop because
        #   after a graph modification followed by simplify new fusing opportunities
        #   might arise. We use the hash of the SDFG to detect if we have reached a
        #   fix point.
        # TODO(phimuell): Find a better upper bound for the starvation protection.
        for _ in range(100):
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

            # TODO(phimuell): Should we do this all the time or only once? (probably the later)
            # TODO(phimuell): Add a criteria to decide if we should promote or not.
            phase2_cleanup.append(
                gtx_transformations.SerialMapPromoter(
                    only_toplevel_maps=True,
                    promote_vertical=True,
                    promote_horizontal=False,
                    promote_local=False,
                )
            )

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
            gt_simplify(sdfg)

        else:
            raise RuntimeWarning("Optimization of the SDFG did not converge.")

        # Phase 3: Optimizing the kernels themselves.
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
            sdfg.apply_transformations_once_everywhere(
                gtx_transformations.MapIterationOrder(
                    leading_dim=leading_dim,
                )
            )

        # Phase 5: Apply blocking
        if block_dim is not None:
            sdfg.apply_transformations_once_everywhere(
                gtx_transformations.KBlocking(
                    blocking_size=blocking_size,
                    block_dim=block_dim,
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
            gpu_launch_factor: Optional[int] = kwargs.get("gpu_launch_factor", None)
            gpu_launch_bounds: Optional[int] = kwargs.get("gpu_launch_bounds", None)
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
