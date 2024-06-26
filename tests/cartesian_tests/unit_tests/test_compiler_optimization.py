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

import pytest

from gt4py.cartesian.gtscript import Field, PARALLEL, computation, interval, stencil


def apb2c(a: Field[float], b: Field[float], c: Field[float]):  # type: ignore
    with computation(PARALLEL), interval(...):  # type: ignore
        c = a + b


@pytest.mark.parametrize("extra_opt_flags", ["", "-ftree-vectorize -DNDEBUG"])
@pytest.mark.parametrize("opt_level", ["0", "1", "2", "3", "s"])
def test_stencil_options(opt_level, extra_opt_flags):
    stencil_obj = stencil(
        backend="gt:cpu_kfirst",
        definition=apb2c,
        opt_level=opt_level,
        extra_opt_flags=extra_opt_flags,
    )
    backend_opts = stencil_obj.options["backend_opts"]
    assert "opt_level" in backend_opts
    assert backend_opts["opt_level"] == opt_level
    assert "extra_opt_flags" in backend_opts
    assert backend_opts["extra_opt_flags"] == extra_opt_flags
