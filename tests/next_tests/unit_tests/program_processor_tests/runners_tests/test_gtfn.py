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
from gt4py.next import allocators, config
from gt4py.next.iterator import transforms
from gt4py.next.iterator.transforms import global_tmps
from gt4py.next.otf import workflow
from gt4py.next.program_processors.runners import gtfn


def test_backend_factory_trait_device():
    cpu_version = gtfn.GTFNBackendFactory(gpu=False, cached=False)
    gpu_version = gtfn.GTFNBackendFactory(gpu=True, cached=False)

    assert cpu_version.executor.__name__ == "run_gtfn_cpu"
    assert gpu_version.executor.__name__ == "run_gtfn_gpu"

    assert cpu_version.executor.otf_workflow.translation.device_type is core_defs.DeviceType.CPU
    assert gpu_version.executor.otf_workflow.translation.device_type is core_defs.DeviceType.CUDA

    assert (
        cpu_version.executor.otf_workflow.decoration.keywords["device"] is core_defs.DeviceType.CPU
    )
    assert (
        gpu_version.executor.otf_workflow.decoration.keywords["device"] is core_defs.DeviceType.CUDA
    )

    assert allocators.is_field_allocator_for(cpu_version.allocator, core_defs.DeviceType.CPU)
    assert allocators.is_field_allocator_for(gpu_version.allocator, core_defs.DeviceType.CUDA)


def test_backend_factory_trait_cached():
    cached_version = gtfn.GTFNBackendFactory(gpu=False, cached=True)
    assert isinstance(cached_version.executor.otf_workflow, workflow.CachedStep)
    assert cached_version.executor.__name__ == "run_gtfn_cpu_cached"


def test_backend_factory_trait_temporaries():
    inline_version = gtfn.GTFNBackendFactory(cached=False)
    temps_version = gtfn.GTFNBackendFactory(cached=False, use_temporaries=True)

    assert inline_version.executor.otf_workflow.translation.lift_mode is None
    assert (
        temps_version.executor.otf_workflow.translation.lift_mode
        is transforms.LiftMode.USE_TEMPORARIES
    )

    assert inline_version.executor.otf_workflow.translation.temporary_extraction_heuristics is None
    assert (
        temps_version.executor.otf_workflow.translation.temporary_extraction_heuristics
        is global_tmps.SimpleTemporaryExtractionHeuristics
    )


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
