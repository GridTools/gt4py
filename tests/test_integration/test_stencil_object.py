# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration tests for StencilObjects."""

import time

import pytest

from gt4py import gtscript
from gt4py import storage as gt_storage


@pytest.fixture
def backend() -> str:
    return "gtc:numpy"


def test_stencil_object_cache(backend: str):
    @gtscript.stencil(backend=backend)
    def stencil(
        in_field: gtscript.Field[float], out_field: gtscript.Field[float], *, offset: float
    ):
        with computation(PARALLEL), interval(...):
            out_field = in_field + offset

    shape = (4, 4, 4)
    in_storage = gt_storage.ones(
        backend=backend, default_origin=(0, 0, 0), shape=shape, dtype=float
    )
    out_storage = gt_storage.ones(
        backend=backend, default_origin=(0, 0, 0), shape=shape, dtype=float
    )

    exec_info = {}
    start_time = time.perf_counter()
    stencil(in_storage, out_storage, offset=1.0, exec_info=exec_info)
    end_time = time.perf_counter()
    run_time = exec_info["run_end_time"] - exec_info["run_start_time"]
    base_time = end_time - start_time - run_time

    start_time = time.perf_counter()
    stencil(in_storage, out_storage, offset=1.0, exec_info=exec_info)
    end_time = time.perf_counter()
    run_time = exec_info["run_end_time"] - exec_info["run_start_time"]
    fast_time = end_time - start_time - run_time

    assert fast_time < base_time

    # When an origin changes, it needs to recompute more, so the time should increase
    out_storage = gt_storage.ones(
        backend=backend, default_origin=(1, 0, 0), shape=shape, dtype=float
    )

    start_time = time.perf_counter()
    stencil(in_storage, out_storage, offset=1.0, exec_info=exec_info)
    end_time = time.perf_counter()
    run_time = exec_info["run_end_time"] - exec_info["run_start_time"]
    third_time = end_time - start_time - run_time

    assert third_time > fast_time
