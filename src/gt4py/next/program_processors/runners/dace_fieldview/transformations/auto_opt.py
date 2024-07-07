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

from typing import Any

import dace
from dace.transformation.auto import auto_optimize as dace_aoptimize
from dace.transformation import dataflow as dace_dataflow


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


def gt_auto_optimize(
    sdfg: dace.SDFG,
    device: dace.DeviceType = dace.DeviceType.CPU,
    **kwargs: Any,
) -> dace.SDFG:
    """Performs GT4Py specific optimizations in place.

    Args:
        sdfg: The SDFG that should ve optimized in place.
        device: The device for which we should optimize.
    """

    # Initial cleaning
    sdfg.simplify()



    # Due to the structure of the generated SDFG getting rid of Maps,
    #  i.e. fusing them, is the best we can currently do.
    sdfg.apply_transformations_repeated([dace_dataflow.MapFusion])

    # These are the part that we copy from DaCe built in auto optimization.
    dace_aoptimize.set_fast_implementations(sdfg, device)
    dace_aoptimize.make_transients_persistent(sdfg, device)
    dace_aoptimize.move_small_arrays_to_stack(sdfg)

    sdfg.simplify()

    return sdfg

