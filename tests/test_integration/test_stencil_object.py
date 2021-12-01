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

from typing import Any, Dict

import pytest

from gt4py import gtscript
from gt4py import storage as gt_storage
from gt4py.gtscript import PARALLEL, Field, computation, interval


@pytest.mark.parametrize("backend", ["gtc:numpy"])
def test_stencil_object_cache(backend: str):
    @gtscript.stencil(backend=backend)
    def stencil(in_field: Field[float], out_field: Field[float], *, offset: float):
        with computation(PARALLEL), interval(...):
            out_field = (  # noqa: F841 # local variable 'out_field' is assigned to but never used
                in_field + offset
            )

    shape = (4, 4, 4)
    in_storage = gt_storage.ones(
        backend=backend, default_origin=(0, 0, 0), shape=shape, dtype=float
    )
    out_storage = gt_storage.ones(
        backend=backend, default_origin=(0, 0, 0), shape=shape, dtype=float
    )

    def runit(*args, **kwargs) -> float:
        exec_info: Dict[str, Any] = {}
        stencil(*args, **kwargs, exec_info=exec_info)
        run_time: float = exec_info["run_end_time"] - exec_info["run_start_time"]
        call_time: float = exec_info["call_run_end_time"] - exec_info["call_run_start_time"]
        return call_time - run_time

    base_time = runit(in_storage, out_storage, offset=1.0)
    fast_time = runit(in_storage, out_storage, offset=1.0)
    assert fast_time < base_time

    # When an origin changes, it needs to recompute more, so the time should increase
    out_storage = gt_storage.ones(
        backend=backend, default_origin=(1, 0, 0), shape=shape, dtype=float
    )
    third_time = runit(in_storage, out_storage, offset=1.0)
    assert third_time > fast_time
