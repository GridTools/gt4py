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
"""
Test that the high level gtfn interface respects user config.

Note: certain features of the config system can not be tested.

These features include:
- build cache location
- debug mode

Because monkey patching the config variables is not enough, as
other variables are computed at import time based on them.
"""

import gt4py._core.definitions as core_defs
from gt4py.next import allocators, config
from gt4py.next.program_processors.runners import gtfn
from gt4py.otf import workflow


def test_backend_factory_set_device():
    cpu_version = gtfn.GTFNBackendFactory(gpu=False, cached=False)
    gpu_version = gtfn.GTFNBackendFactory(gpu=True, cached=False)

    assert cpu_version.executor.__name__ == "run_gtfn_cpu"
    assert gpu_version.executor.__name__ == "run_gtfn_gpu"

    assert cpu_version.executor.otf_workflow.translation.device_type is core_defs.DeviceType.CPU
    assert gpu_version.executor.otf_workflow.translation.device_type is core_defs.DeviceType.CUDA

    assert cpu_version.executor.decoration.keywords["device"] is core_defs.DeviceType.CPU
    assert gpu_version.executor.decoration.keywords["device"] is core_defs.DeviceType.CUDA

    assert allocators.is_allocator_for(cpu_version.allocator, core_defs.DeviceType.CPU)
    assert allocators.is_allocator_for(gpu_version.allocator, core_defs.DeviceType.CUDA)


def test_backend_factory_set_cached():
    cached_version = gtfn.GTFNBackendFactory(gpu=False, cached=True)
    assert isinstance(cached_version.executor.otf_workflow, workflow.CachedStep)
    assert cached_version.executor.__name__ == "run_gtfn_cpu_cached"


def test_backend_factory_build_cache_config(monkeypatch):
    monkeypatch.setattr(config, "BUILD_CACHE_LIFETIME", config.BuildCacheLifetime.SESSION)
    session_version = gtfn.GTFNBackendFactory()
    monkeypatch.setattr(config, "BUILD_CACHE_LIFETIME", config.BuildCacheLifetime.PERSISTENT)
    persistent_version = gtfn.GTFNBackendFactory()

    assert (
        session_version.executor.otf_workflow.compilation.cache_lifetime
        is config.BuildCacheLifetime.SESSION
    )
    assert (
        persistent_version.executor.otf_workflow.compilation.cache_lifetime
        is config.BuildCacheLifetime.PERSISTENT
    )


def test_backend_factory_build_type_config(monkeypatch):
    monkeypatch.setattr(config, "CMAKE_BUILD_TYPE", config.CMakeBuildType.RELEASE)
    release_version = gtfn.GTFNBackendFactory()
    monkeypatch.setattr(config, "CMAKE_BUILD_TYPE", config.CMakeBuildType.MIN_SIZE_REL)
    min_size_version = gtfn.GTFNBackendFactory()

    assert (
        release_version.executor.otf_workflow.compilation.builder_factory.cmake_build_type
        is config.CMakeBuildType.RELEASE
    )
    assert (
        min_size_version.executor.otf_workflow.compilation.builder_factory.cmake_build_type
        is config.CMakeBuildType.MIN_SIZE_REL
    )
