# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
