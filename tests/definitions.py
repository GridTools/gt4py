# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2022, ETH Zurich
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

from gt4py import backend as gt_backend
from gt4py import utils as gt_utils


def _backend_name_as_param(name):
    marks = []
    if gt_backend.from_name(name).storage_info["device"] == "gpu":
        marks.append(pytest.mark.requires_gpu)
    if "dace" in name:
        marks.append(pytest.mark.requires_dace)
    return pytest.param(name, marks=marks)


_ALL_BACKEND_NAMES = list(gt_backend.REGISTRY.keys())


CPU_BACKENDS = [
    _backend_name_as_param(name)
    for name in _ALL_BACKEND_NAMES
    if gt_backend.from_name(name).storage_info["device"] == "cpu"
]
GPU_BACKENDS = [
    _backend_name_as_param(name)
    for name in _ALL_BACKEND_NAMES
    if gt_backend.from_name(name).storage_info["device"] == "gpu"
]
ALL_BACKENDS = CPU_BACKENDS + GPU_BACKENDS


@pytest.fixture()
def id_version():
    return gt_utils.shashed_id(str(datetime.datetime.now()))
