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
from typing import Callable, List

import numpy as np
import pytest

from gt4py import cartesian as gt4pyc
from gt4py.cartesian import utils as gt_utils
from gt4py.cartesian.backend.base import Backend, from_name
from gt4py.cartesian.testing.definitions import (
    ALL_BACKEND_NAMES,
    CPU_BACKEND_NAMES,
    GPU_BACKEND_NAMES,
    PERFORMANCE_BACKEND_NAMES,
)
from gt4py.cartesian.testing.suites import ParameterSet
from gt4py.storage.cartesian.layout import is_gpu_device


def _backend_name_as_param(backend_name: str) -> ParameterSet:
    marks = []
    if backend_name in GPU_BACKEND_NAMES:
        marks.append(pytest.mark.requires_gpu)
    if "dace" in backend_name:
        marks.append(pytest.mark.requires_dace)
    return pytest.param(backend_name, marks=marks)


def _filter_backends(filter: Callable[[Backend], bool]) -> List[str]:
    res = []
    for name in ALL_BACKEND_NAMES:
        backend = from_name(name)
        if not getattr(backend, "disabled", False) and filter(backend):
            res.append(_backend_name_as_param(name))
    return res


CPU_BACKENDS = _filter_backends(lambda backend: backend.name in CPU_BACKEND_NAMES)
GPU_BACKENDS = _filter_backends(lambda backend: backend.name in GPU_BACKEND_NAMES)
ALL_BACKENDS = CPU_BACKENDS + GPU_BACKENDS

PERFORMANCE_BACKENDS = [_backend_name_as_param(name) for name in PERFORMANCE_BACKEND_NAMES]


@pytest.fixture()
def id_version():
    return gt_utils.shashed_id(str(datetime.datetime.now()))


def get_array_library(backend: str):
    """Return device ready array maker library"""
    backend_cls = gt4pyc.backend.from_name(backend)
    assert backend_cls is not None
    if is_gpu_device(backend_cls.storage_info):
        assert cp is not None
        return cp
    else:
        return np
