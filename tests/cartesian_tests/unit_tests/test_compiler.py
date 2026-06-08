# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Generator, Any

import pytest
import os

from gt4py.cartesian.utils.compiler import gpu_configuration


@pytest.fixture(
    scope="function",
    autouse=True,
    params=[
        pytest.param("", id="no-extra-args"),
        pytest.param("arg1 arg2", id="extra-args"),
    ],
)
def extra_cuda_args(request) -> Generator[Any, None, None]:
    # keep a snapshot of old state
    old_extra_args = os.environ.get("GT4PY_CARTESIAN_EXTRA_CUDA_COMPILE_ARGS", "")

    # run test with extra args
    os.environ["GT4PY_CARTESIAN_EXTRA_CUDA_COMPILE_ARGS"] = request.param
    yield

    # reset old state
    if old_extra_args:
        os.environ["GT4PY_CARTESIAN_EXTRA_CUDA_COMPILE_ARGS"] = old_extra_args
    else:
        del os.environ["GT4PY_CARTESIAN_EXTRA_CUDA_COMPILE_ARGS"]


@pytest.mark.parametrize("optimization_level", ["0", "1", "2", "3", "s"])
def test_gpu_configuration(optimization_level: str) -> None:
    config = gpu_configuration(optimization_level)

    assert isinstance(config.gpu_compile_flags, list)
    for flag in config.gpu_compile_flags:
        assert len(flag) > 0
        assert flag.strip() == flag

    if optimization_level == "0":
        assert "-fmad=false" in config.gpu_compile_flags

    extra_args = os.environ.get("GT4PY_CARTESIAN_EXTRA_CUDA_COMPILE_ARGS", "").split(" ")
    for arg in extra_args:
        if len(arg.strip()) > 0:
            assert arg in config.gpu_compile_flags
