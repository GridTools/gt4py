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

from gt4py import backend as gt_backend
from gt4py import utils as gt_utils


def _backend_name_as_param(name):
    if gt_backend.from_name(name).storage_info["device"] == "gpu":
        return pytest.param(name, marks=[pytest.mark.requires_gpu])
    else:
        return pytest.param(name)


def make_backend_params(*names):
    return map(_backend_name_as_param, names)


_ALL_BACKEND_NAMES = list(gt_backend.REGISTRY.keys())
_INTERNAL_BACKEND_NAMES = ["debug", "numpy"] + [
    name for name in _ALL_BACKEND_NAMES if name.startswith("gt")
]


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

INTERNAL_CPU_BACKENDS = [
    _backend_name_as_param(name)
    for name in _INTERNAL_BACKEND_NAMES
    if gt_backend.from_name(name).storage_info["device"] == "cpu"
]
INTERNAL_GPU_BACKENDS = [
    _backend_name_as_param(name)
    for name in _INTERNAL_BACKEND_NAMES
    if gt_backend.from_name(name).storage_info["device"] == "gpu"
]

INTERNAL_BACKENDS = INTERNAL_CPU_BACKENDS + INTERNAL_GPU_BACKENDS

OLD_BACKENDS = [
    _backend_name_as_param(name) for name in _ALL_BACKEND_NAMES if not name.startswith("gtc:")
]
OLD_INTERNAL_BACKENDS = [
    _backend_name_as_param(name) for name in _INTERNAL_BACKEND_NAMES if not name.startswith("gtc:")
]

LEGACY_GRIDTOOLS_BACKENDS = [_backend_name_as_param(name) for name in ("gtx86", "gtmc", "gtcuda")]

GTC_BACKENDS = [
    _backend_name_as_param(name) for name in _ALL_BACKEND_NAMES if name.startswith("gtc:")
]


@pytest.fixture()
def id_version():
    return gt_utils.shashed_id(str(datetime.datetime.now()))
