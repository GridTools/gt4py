# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared build system functionality."""

from __future__ import annotations

import functools
import importlib
import os
import warnings

from gt4py._core import definitions as core_defs


def python_module_suffix() -> str:
    return importlib.machinery.EXTENSION_SUFFIXES[0][1:]


@functools.cache
def _query_device_arch() -> str | None:
    # Cached: the device does not change within a process, and querying early is
    # load-bearing — on a saturated GPU (e.g. many test processes sharing one
    # device) a late query can fail with cudaErrorMemoryAllocation even though
    # an early one succeeded.
    if core_defs.CUPY_DEVICE_TYPE == core_defs.DeviceType.CUDA:
        # use `cp` from core_defs to avoid trying to re-import cupy
        try:
            return core_defs.cp.cuda.Device(0).compute_capability  # type: ignore[attr-defined]
        except core_defs.cp.cuda.runtime.CUDARuntimeError as e:  # type: ignore[attr-defined]
            warnings.warn(
                UserWarning(f"Could not determine the CUDA compute capability: {e}"), stacklevel=2
            )
            return None
    elif core_defs.CUPY_DEVICE_TYPE == core_defs.DeviceType.ROCM:
        # TODO(egparedes): Implement this properly, either parsing the output of `$ rocminfo`
        # or using the HIP low level bindings.
        # Check: https://rocm.docs.amd.com/projects/hip-python/en/latest/user_guide/1_usage.html
        return "gfx942"  # MI300A

    return None


def get_device_arch() -> str | None:
    """CMake-style architecture(s) of the current device, or None if unknown.

    Single source for device-architecture detection: an explicit ``CUDAARCHS`` /
    ``HIPARCHS`` environment variable takes precedence over querying the device
    (which is impossible where no device is visible, e.g. in a process-pool
    worker).
    """
    match core_defs.CUPY_DEVICE_TYPE:
        case core_defs.DeviceType.CUDA:
            env_archs = os.environ.get("CUDAARCHS", "").strip()
        case core_defs.DeviceType.ROCM:
            # `HIPARCHS` is not officially supported by CMake yet, but it might be in the future
            env_archs = os.environ.get("HIPARCHS", "").strip()
        case _:
            return None
    return env_archs or _query_device_arch()
