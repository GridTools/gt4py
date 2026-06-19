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
from gt4py.next import config, custom_layout_allocators
from gt4py.next.iterator import transforms
from gt4py.next.iterator.transforms import global_tmps
from gt4py.next.otf import workflow
from gt4py.next.program_processors.runners import gtfn


def test_backend_factory_trait_device():
    cpu_version = gtfn.GTFNBackendFactory(gpu=False)
    gpu_version = gtfn.GTFNBackendFactory(gpu=True)

    assert cpu_version.name == "run_gtfn_cpu"
    assert isinstance(cpu_version.executor, workflow.CachedStep)
    otf_workflow_cpu = cpu_version.executor.step
    assert gpu_version.name == "run_gtfn_gpu"
    assert isinstance(gpu_version.executor, workflow.CachedStep)
    otf_workflow_gpu = gpu_version.executor.step

    assert isinstance(otf_workflow_cpu.translation, workflow.CachedStep)
    assert otf_workflow_cpu.translation.step.device_type is core_defs.DeviceType.CPU
    assert isinstance(otf_workflow_gpu.translation, workflow.CachedStep)
    assert otf_workflow_gpu.translation.step.device_type is core_defs.DeviceType.CUDA

    assert otf_workflow_cpu.decoration.keywords["device"] is core_defs.DeviceType.CPU
    assert otf_workflow_gpu.decoration.keywords["device"] is core_defs.DeviceType.CUDA

    assert custom_layout_allocators.is_field_allocator_for(
        cpu_version.allocator, core_defs.DeviceType.CPU
    )
    assert custom_layout_allocators.is_field_allocator_for(
        gpu_version.allocator, core_defs.DeviceType.CUDA
    )


def test_backend_factory_build_cache_config(monkeypatch):
    monkeypatch.setattr(config, "BUILD_CACHE_LIFETIME", config.BuildCacheLifetime.SESSION)
    session_version = gtfn.GTFNBackendFactory()
    assert isinstance(session_version.executor, workflow.CachedStep)
    session_otf_workflow = session_version.executor.step
    monkeypatch.setattr(config, "BUILD_CACHE_LIFETIME", config.BuildCacheLifetime.PERSISTENT)
    persistent_version = gtfn.GTFNBackendFactory()
    assert isinstance(persistent_version.executor, workflow.CachedStep)
    persistent_otf_workflow = persistent_version.executor.step

    assert session_otf_workflow.compilation.cache_lifetime is config.BuildCacheLifetime.SESSION
    assert (
        persistent_otf_workflow.compilation.cache_lifetime is config.BuildCacheLifetime.PERSISTENT
    )


def test_backend_factory_build_type_config(monkeypatch):
    monkeypatch.setattr(config, "CMAKE_BUILD_TYPE", config.CMakeBuildType.RELEASE)
    release_version = gtfn.GTFNBackendFactory()
    assert isinstance(release_version.executor, workflow.CachedStep)
    release_otf_workflow = release_version.executor.step
    monkeypatch.setattr(config, "CMAKE_BUILD_TYPE", config.CMakeBuildType.MIN_SIZE_REL)
    min_size_version = gtfn.GTFNBackendFactory()
    assert isinstance(min_size_version.executor, workflow.CachedStep)
    min_size_otf_workflow = min_size_version.executor.step

    assert (
        release_otf_workflow.compilation.builder_factory.cmake_build_type
        is config.CMakeBuildType.RELEASE
    )
    assert (
        min_size_otf_workflow.compilation.builder_factory.cmake_build_type
        is config.CMakeBuildType.MIN_SIZE_REL
    )
