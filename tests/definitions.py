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

try:
    import cupy as cp

    cp.cuda.Device()
except (ImportError, RuntimeError):
    cp = None

import datetime

import pytest

import gt4py.backend as gt_backend
import gt4py.utils as gt_utils


ALL_BACKENDS = list(gt_backend.REGISTRY.keys())
if cp is None:
    # Skip gpu backends
    ALL_BACKENDS = [
        name for name in ALL_BACKENDS if gt_backend.from_name(name).compute_device != "gpu"
    ]

CPU_BACKENDS = [name for name in ALL_BACKENDS if gt_backend.from_name(name).compute_device == "cpu"]
GPU_BACKENDS = list(set(ALL_BACKENDS) - set(CPU_BACKENDS))
INTERNAL_BACKENDS = ["debug", "numpy"] + [name for name in ALL_BACKENDS if name.startswith("gt")]
DAWN_BACKENDS = [name for name in ALL_BACKENDS if "dawn:" in name]
DAWN_CPU_BACKENDS = [name for name in CPU_BACKENDS if "dawn:" in name]
DAWN_GPU_BACKENDS = [name for name in GPU_BACKENDS if "dawn:" in name]


@pytest.fixture()
def id_version():
    return gt_utils.shashed_id(str(datetime.datetime.now()))
