# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
from gt4py.next import _allocators, config
from gt4py.next.iterator import transforms
from gt4py.next.iterator.transforms import global_tmps
from gt4py.next.otf import workflow
from gt4py.next.program_processors.runners import gtfn


def test_backend_factory_trait_device():
    cpu_version = gtfn.GTFNBackendFactory(gpu=False, cached=False)
    gpu_version = gtfn.GTFNBackendFactory(gpu=True, cached=False)

    assert cpu_version.name == "run_gtfn_cpu"
    assert gpu_version.name == "run_gtfn_gpu"

    assert cpu_version.executor.translation.device_type is core_defs.DeviceType.CPU
    assert gpu_version.executor.translation.device_type is core_defs.DeviceType.CUDA

    assert cpu_version.executor.decoration.keywords["device"] is core_defs.DeviceType.CPU
    assert gpu_version.executor.decoration.keywords["device"] is core_defs.DeviceType.CUDA

    assert _allocators.is_field_allocator_for(cpu_version.allocator, core_defs.DeviceType.CPU)
    assert _allocators.is_field_allocator_for(gpu_version.allocator, core_defs.DeviceType.CUDA)


def test_backend_factory_trait_cached():
    cached_version = gtfn.GTFNBackendFactory(gpu=False, cached=True)
    assert isinstance(cached_version.executor, workflow.CachedStep)
    assert cached_version.name == "run_gtfn_cpu_cached"


def test_backend_factory_build_cache_config(monkeypatch):
    monkeypatch.setattr(config, "BUILD_CACHE_LIFETIME", config.BuildCacheLifetime.SESSION)
    session_version = gtfn.GTFNBackendFactory()
    monkeypatch.setattr(config, "BUILD_CACHE_LIFETIME", config.BuildCacheLifetime.PERSISTENT)
    persistent_version = gtfn.GTFNBackendFactory()

    assert session_version.executor.compilation.cache_lifetime is config.BuildCacheLifetime.SESSION
    assert (
        persistent_version.executor.compilation.cache_lifetime
        is config.BuildCacheLifetime.PERSISTENT
    )


def test_backend_factory_build_type_config(monkeypatch):
    monkeypatch.setattr(config, "CMAKE_BUILD_TYPE", config.CMakeBuildType.RELEASE)
    release_version = gtfn.GTFNBackendFactory()
    monkeypatch.setattr(config, "CMAKE_BUILD_TYPE", config.CMakeBuildType.MIN_SIZE_REL)
    min_size_version = gtfn.GTFNBackendFactory()

    assert (
        release_version.executor.compilation.builder_factory.cmake_build_type
        is config.CMakeBuildType.RELEASE
    )
    assert (
        min_size_version.executor.compilation.builder_factory.cmake_build_type
        is config.CMakeBuildType.MIN_SIZE_REL
    )
