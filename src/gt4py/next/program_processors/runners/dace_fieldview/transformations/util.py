# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Common functionality for the transformations/optimization pipeline."""

from typing import Any, Final, Iterable, Optional

import dace
from dace.transformation import dataflow as dace_dataflow, passes as dace_passes

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


def gt_make_transients_persistent(
    sdfg: dace.SDFG,
    device: dace.DeviceType,
    make_wcr_atomic_on_gpu: bool = True,
) -> dict[int, set[str]]:
    """
    Changes the lifetime of certain transients to `Persistent`.

    A persistent lifetime means that the transient is allocated only the very first
    time and only deallocated if the underlying `CompiledSDFG` object goes out of
    scope or if the exit handler of the SDFG is called. The main advantage is,
    that memory must not be allocated, however, the SDFG can not be called by
    different threads.
    This function was copied from DaCe.

    Args:
        sdfg: The SDFG to process.
        device: The device type.
        make_wcr_atomic_on_gpu: Reset the `wcr_nonatomic` property of
            all memlets, enabled by default. See note for more info.

    Returns:
        A dictionary mapping SDFG IDs to a set of transient arrays that
        were made persistent.

    Notes:
        For unknown reasons the DaCe counterpart of this function, resets the
        `wcr_nonatomic` property of every memlet, but only on GPU and only on the
        top level SDFG, i.e. not on the nested SDFGs. This means that every
        reduction (on the top level on GPU) becomes atomic in every situation.
        However, this behaviour might change in GT4Py, but if desired, it can
        disabled explicitly.
    """
    result: dict[int, set[str]] = {}
    for nsdfg in sdfg.all_sdfgs_recursive():
        fsyms: set[str] = nsdfg.free_symbols
        modify_lifetime: set[str] = set()
        not_modify_lifetime: set[str] = set()

        for state in nsdfg.states():
            for dnode in state.data_nodes():
                if dnode.data in not_modify_lifetime:
                    continue

                if dnode.data in nsdfg.constants_prop:
                    not_modify_lifetime.add(dnode.data)
                    continue

                desc = dnode.desc(nsdfg)
                if not desc.transient or type(desc) not in {dace.data.Array, dace.data.Scalar}:
                    not_modify_lifetime.add(dnode.data)
                    continue
                if desc.storage == dace.StorageType.Register:
                    not_modify_lifetime.add(dnode.data)
                    continue

                if desc.lifetime == dace.AllocationLifetime.External:
                    not_modify_lifetime.add(dnode.data)
                    continue

                try:
                    # The symbols describing the total size must be a subset of the
                    #  free symbols of the SDFG (symbols passed as argument).
                    # NOTE: This ignores the renaming of symbols through the
                    #   `symbol_mapping` property of nested SDFGs.
                    if not set(map(str, desc.total_size.free_symbols)).issubset(fsyms):
                        not_modify_lifetime.add(dnode.data)
                        continue
                except AttributeError:  # total_size is an integer / has no free symbols
                    pass

                # Make it persistent.
                modify_lifetime.add(dnode.data)

        # Now setting the lifetime.
        result[nsdfg.cfg_id] = modify_lifetime - not_modify_lifetime
        for aname in result[nsdfg.cfg_id]:
            nsdfg.arrays[aname].lifetime = dace.AllocationLifetime.Persistent

    if device == dace.DeviceType.GPU and make_wcr_atomic_on_gpu:
        # Make all WCR atomic.
        # NOTE: This is not an "error", DaCe does this only on GPU and only on
        #   the top levels.
        for state in sdfg.states():
            for edge in state.edges():
                edge.data.wcr_nonatomic = False

    return result
