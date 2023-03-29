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
import numpy as np

from gt4py.next import common
from gt4py.next.ffront import decorator, fbuiltins
from gt4py.next.iterator.embedded import np_as_located_field
from gt4py.next.program_processors.runners import gtfn_cpu


IDim = common.Dimension("IDim")
JDim = common.Dimension("JDim")


def test_different_buffer_sizes():
    """
    Ensure correct bindings are generated for buffers of different sizes (strides).

    Note:
      gridtools::sid::composite share the same runtime strides if they are of the same kind at compile-time.
      This test would fail if the same strides kind is used for both storages.
    """
    nx = 6
    ny = 7
    out_nx = 5
    out_ny = 5

    inp = np_as_located_field(IDim, JDim)(np.reshape(np.arange(nx * ny, dtype=np.int32), (nx, ny)))
    out = np_as_located_field(IDim, JDim)(np.zeros((out_nx, out_ny), dtype=np.int32))

    @decorator.field_operator(backend=gtfn_cpu.run_gtfn)
    def copy(
        inp: common.Field[[IDim, JDim], fbuiltins.int32]
    ) -> common.Field[[IDim, JDim], fbuiltins.int32]:
        return inp

    copy(inp, out=out, offset_provider={})

    assert np.allclose(inp[:out_nx, :out_ny], out)
