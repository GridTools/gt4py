# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

try:
    import cupy as cp

    cp.cuda.Device()
except (ImportError, RuntimeError):
    cp = None

import datetime

import numpy as np
import pytest

from gt4py import cartesian as gt4pyc
from gt4py.cartesian import utils as gt_utils


def _backend_name_as_param(name: str):
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


PERFORMANCE_BACKENDS = [
    _backend_name_as_param(name) for name in _ALL_BACKEND_NAMES if name not in ("numpy", "debug")
]


@pytest.fixture()
def id_version():
    return gt_utils.shashed_id(str(datetime.datetime.now()))


def get_array_library(backend: str):
    """Return device ready array maker library"""
    backend_cls = gt4pyc.backend.from_name(backend)
    if backend_cls.storage_info["device"] == "gpu":
        assert cp is not None
        return cp
    else:
        return np
