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

try:
    import cupy as cp

    cp.cuda.Device()
except (ImportError, RuntimeError):
    cp = None

import datetime

import pytest

from gt4py import cartesian as gt4pyc
from gt4py.cartesian import utils as gt_utils


def _backend_name_as_param(name):
    marks = []
    if gt4pyc.backend.from_name(name).storage_info["device"] == "gpu":
        marks.append(pytest.mark.requires_gpu)
    if "dace" in name:
        marks.append(pytest.mark.requires_dace)
    return pytest.param(name, marks=marks)


_ALL_BACKEND_NAMES = list(gt4pyc.backend.REGISTRY.keys())


def _get_backends_with_storage_info(storage_info_kind: str):
    res = []
    for name in _ALL_BACKEND_NAMES:
        backend = gt4pyc.backend.from_name(name)
        if not getattr(backend, "disabled", False):
            if backend.storage_info["device"] == storage_info_kind:
                res.append(_backend_name_as_param(name))
    return res


CPU_BACKENDS = _get_backends_with_storage_info("cpu")
GPU_BACKENDS = _get_backends_with_storage_info("gpu")
ALL_BACKENDS = CPU_BACKENDS + GPU_BACKENDS

_PERFORMANCE_BACKEND_NAMES = [name for name in _ALL_BACKEND_NAMES if name != "numpy"]
PERFORMANCE_BACKENDS = [_backend_name_as_param(name) for name in _PERFORMANCE_BACKEND_NAMES]


@pytest.fixture()
def id_version():
    return gt_utils.shashed_id(str(datetime.datetime.now()))
