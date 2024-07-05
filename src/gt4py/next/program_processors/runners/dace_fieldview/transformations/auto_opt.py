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
    from dace.transformation.auto.auto_optimize import auto_optimize as _auto_optimize

    # Now put output storages everywhere to make auto optimizer less likely to fail.
    # sdfg.apply_transformations_repeated([InLocalStorage, OutLocalStorage])  # noqa: ERA001 [commented-out-code]

    # Now the optimization.
    sdfg = _auto_optimize(sdfg, device=device, **kwargs)

    # Now the simplification step.
    #  This should get rid of some of teh additional transients we have added.
    sdfg.simplify()

    return sdfg
