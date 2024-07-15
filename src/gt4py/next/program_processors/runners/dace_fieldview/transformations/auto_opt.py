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

from typing import Any, Optional

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
    **kwargs: Any,
) -> dace.SDFG:
    """This is a convenient wrapper arround DaCe's `auto_optimize` function.

    By default it uses the `CPU` device type. Furthermore, it will first run the
    `{In, Out}LocalStorage` transformations of the SDFG. The reason for this is that
    empirical observations have shown, that the current auto optimizer has problems
    in certain cases and this should prevent some of them.

    Args:
        sdfg: The SDFG that should be optimized in place.
        device: the device for which optimizations should be done, defaults to CPU.
        kwargs: Are forwarded to the underlying auto optimized exposed by DaCe.
    """

    # Now put output storages everywhere to make auto optimizer less likely to fail.
    # sdfg.apply_transformations_repeated([InLocalStorage, OutLocalStorage])  # noqa: ERA001 [commented-out-code]

    # Now the optimization.
    sdfg = dace_aoptimize(sdfg, device=device, **kwargs)

    # Now the simplification step.
    #  This should get rid of some of teh additional transients we have added.
    sdfg.simplify()

    return sdfg


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
    device: dace.DeviceType = dace.DeviceType.CPU,
    leading_dim: Optional[gtx_common.Dimension] = None,
    **kwargs: Any,
) -> dace.SDFG:
    """Performs GT4Py specific optimizations in place.

    Args:
        sdfg: The SDFG that should ve optimized in place.
        device: The device for which we should optimize.
    """

    with dace.config.temporary_config():
        dace.Config.set("optimizer", "match_exception", value=True)

        # Initial cleaning
        gt_simplify(sdfg)

        # Compute the SDFG to see if something has changed.
        sdfg_hash = sdfg.hash_sdfg()

        for _ in range(100):
            # Due to the structure of the generated SDFG getting rid of Maps,
            #  i.e. fusing them, is the best we can currently do.
            if kwargs.get("use_dace_fusion", False):
                sdfg.apply_transformations_repeated([dace_dataflow.MapFusion])
            else:
                xform = gtx_transformations.SerialMapFusion()
                sdfg.apply_transformations_repeated([xform], validate=True, validate_all=True)

            sdfg.apply_transformations_repeated(
                [gtx_transformations.SerialMapPromoter(promote_horizontal=False)],
                validate=True,
                validate_all=True,
            )

            # Maybe running the fusion has opened more opportunities.
            gt_simplify(sdfg)

            # check if something has changed and if so end it here.
            old_sdfg_hash = sdfg_hash
            sdfg_hash = sdfg.hash_sdfg()

            if old_sdfg_hash == sdfg_hash:
                break

        else:
            raise RuntimeWarning("Optimization of the SDFG did not converged.")

        # After we have optimized the SDFG as good as we can, we will now do some
        #  lower level optimization.
        if leading_dim is not None:
            sdfg.apply_transformations_once_everywhere(
                gtx_transformations.MapIterationOrder(
                    leading_dim=leading_dim,
                )
            )

        # These are the part that we copy from DaCe built in auto optimization.
        dace_aoptimize.set_fast_implementations(sdfg, device)
        dace_aoptimize.make_transients_persistent(sdfg, device)
        dace_aoptimize.move_small_arrays_to_stack(sdfg)

        # Final simplify
        gt_simplify(sdfg)

        return sdfg
