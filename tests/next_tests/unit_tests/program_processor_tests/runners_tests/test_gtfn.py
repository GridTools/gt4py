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

import functools
import pathlib
import unittest.mock

import pytest

import gt4py._core.definitions as core_defs
import gt4py.next as gtx
from gt4py.next import config, custom_layout_allocators
from gt4py.next.otf import arguments, artifacts, workflow
from gt4py.next.otf.compilation import build_data, cache, compiler, importer
from gt4py.next.program_processors.codegens.gtfn import gtfn_module
from gt4py.next.program_processors.runners import gtfn
from gt4py.next.type_system import type_specifications as ts


def test_make_gtfn_toolchain_device():
    cpu_version = gtfn.make_gtfn_toolchain(gtfn.GTFNConfig(gpu=False))
    gpu_version = gtfn.make_gtfn_toolchain(gtfn.GTFNConfig(gpu=True))

    assert cpu_version.name == "run_gtfn_cpu"
    assert isinstance(cpu_version.backend.translation, workflow.CachedStep)
    assert cpu_version.backend.translation.step.device_type is core_defs.DeviceType.CPU
    assert gpu_version.name == "run_gtfn_gpu"
    assert isinstance(gpu_version.backend.translation, workflow.CachedStep)
    assert gpu_version.backend.translation.step.device_type is core_defs.DeviceType.CUDA

    # The compilation step now also carries device_type so it can stamp the artifact.
    assert cpu_version.backend.compilation.device_type is core_defs.DeviceType.CPU
    assert gpu_version.backend.compilation.device_type is core_defs.DeviceType.CUDA

    assert custom_layout_allocators.is_field_allocator_for(
        cpu_version.allocator, core_defs.DeviceType.CPU
    )
    assert custom_layout_allocators.is_field_allocator_for(
        gpu_version.allocator, core_defs.DeviceType.CUDA
    )


def test_make_gtfn_toolchain_build_cache_config(monkeypatch):
    monkeypatch.setattr(config, "BUILD_CACHE_LIFETIME", config.BuildCacheLifetime.SESSION)
    session_version = gtfn.make_gtfn_toolchain()
    monkeypatch.setattr(config, "BUILD_CACHE_LIFETIME", config.BuildCacheLifetime.PERSISTENT)
    persistent_version = gtfn.make_gtfn_toolchain()

    assert session_version.backend.compilation.cache_lifetime is config.BuildCacheLifetime.SESSION
    assert (
        persistent_version.backend.compilation.cache_lifetime
        is config.BuildCacheLifetime.PERSISTENT
    )


def test_make_gtfn_toolchain_build_type_config(monkeypatch):
    monkeypatch.setattr(config, "CMAKE_BUILD_TYPE", config.CMakeBuildType.RELEASE)
    release_version = gtfn.make_gtfn_toolchain()
    monkeypatch.setattr(config, "CMAKE_BUILD_TYPE", config.CMakeBuildType.MIN_SIZE_REL)
    min_size_version = gtfn.make_gtfn_toolchain()

    assert (
        release_version.backend.compilation.builder_factory.cmake_build_type
        is config.CMakeBuildType.RELEASE
    )
    assert (
        min_size_version.backend.compilation.builder_factory.cmake_build_type
        is config.CMakeBuildType.MIN_SIZE_REL
    )


def test_cmake_build_type_changes_build_folder(monkeypatch, tmp_path):
    """Different cmake build types must yield different build folders.

    The compiler passes a fingerprint of the builder factory to `get_cache_folder`
    as `build_context_id`. Since the builder factory embeds the cmake build type,
    changing it must result in a different id, so that `debug` and `release` builds
    land in different cache folders.
    """
    monkeypatch.setattr(config, "CMAKE_BUILD_TYPE", config.CMakeBuildType.RELEASE)
    release_version = gtfn.make_gtfn_toolchain()
    monkeypatch.setattr(config, "CMAKE_BUILD_TYPE", config.CMakeBuildType.DEBUG)
    debug_version = gtfn.make_gtfn_toolchain()

    release_compiler = release_version.backend.compilation
    debug_compiler = debug_version.backend.compilation

    build_context_ids: list[str] = []

    def fake_get_cache_folder(
        ext_source: object,
        lifetime: config.BuildCacheLifetime,
        build_context_id: str = "",
    ) -> pathlib.Path:
        build_context_ids.append(build_context_id)
        return tmp_path / build_context_id

    fake_build_data = build_data.BuildData(
        status=build_data.BuildStatus.COMPILED,
        module=pathlib.Path("fake_module.so"),
        entry_point_name="entry_point",
    )

    with (
        unittest.mock.patch.object(cache, "get_cache_folder", side_effect=fake_get_cache_folder),
        unittest.mock.patch.object(build_data, "read_data", return_value=fake_build_data),
        unittest.mock.patch.object(compiler, "module_exists", return_value=True),
        unittest.mock.patch.object(
            importer, "import_from_path", return_value=unittest.mock.MagicMock()
        ),
    ):
        release_compiler(unittest.mock.MagicMock())
        debug_compiler(unittest.mock.MagicMock())

    assert len(build_context_ids) == 2
    assert build_context_ids[0] != build_context_ids[1]


def test_step_builder_partial_keeps_config_settings():
    """A partial of a default step builder changes a step-local setting only."""
    cfg = gtfn.GTFNConfig(gpu=True, cmake_build_type=config.CMakeBuildType.DEBUG)

    toolchain = gtfn.make_gtfn_toolchain(
        cfg,
        translation=functools.partial(gtfn.make_gtfn_translation, enable_itir_transforms=False),
        compilation=functools.partial(
            gtfn.make_gtfn_compiler,
            build_system=functools.partial(
                gtfn.make_gtfn_build_system, cmake_extra_flags=["-DEXTRA=ON"]
            ),
        ),
    )

    translation = toolchain.backend.translation
    assert isinstance(translation, workflow.CachedStep)
    assert translation.step.enable_itir_transforms is False
    assert translation.step.device_type is cfg.device_type
    compilation = toolchain.backend.compilation
    assert compilation.device_type is cfg.device_type
    assert compilation.builder_factory.cmake_extra_flags == ["-DEXTRA=ON"]
    assert compilation.builder_factory.cmake_build_type is config.CMakeBuildType.DEBUG


def test_custom_step_builder_output_is_cached():
    """The cache wraps whatever the translation step builder returns."""
    custom_step = gtfn_module.GTFNTranslationStep(
        device_type=core_defs.DeviceType.CPU, use_max_domain_range_on_unstructured_shift=True
    )

    toolchain = gtfn.make_gtfn_toolchain(translation=lambda cfg: custom_step)

    assert isinstance(toolchain.backend.translation, workflow.CachedStep)
    assert toolchain.backend.translation.step is custom_step


def test_uncached_translation():
    toolchain = gtfn.make_gtfn_toolchain(gtfn.GTFNConfig(cached_translation=False))

    assert isinstance(toolchain.backend.translation, gtfn_module.GTFNTranslationStep)


def test_step_builder_ignoring_config_device_raises():
    def cpu_only_translation(cfg: gtfn.GTFNConfig) -> gtfn_module.GTFNTranslationStep:
        return gtfn_module.GTFNTranslationStep(device_type=core_defs.DeviceType.CPU)

    with pytest.raises(ValueError, match="toolchain is being built for 'CUDA'"):
        gtfn.make_gtfn_toolchain(gtfn.GTFNConfig(gpu=True), translation=cpu_only_translation)


def test_step_builder_cannot_override_config_setting():
    with pytest.raises(TypeError, match="device_type"):
        gtfn.make_gtfn_toolchain(
            gtfn.GTFNConfig(gpu=True),
            translation=functools.partial(
                gtfn.make_gtfn_translation, device_type=core_defs.DeviceType.CPU
            ),
        )


def test_prebuilt_toolchain_names_are_unique():
    names = [gtfn.run_gtfn.name, gtfn.run_gtfn_gpu.name, gtfn.run_gtfn_no_transforms.name]

    assert gtfn.run_gtfn_no_transforms.name == "run_gtfn_cpu_no_transforms"
    assert len(set(names)) == len(names)


IDim = gtx.Dimension("IDim")


@gtx.field_operator
def _copy_op(a: gtx.Field[gtx.Dims[IDim], gtx.float64]) -> gtx.Field[gtx.Dims[IDim], gtx.float64]:
    return a


@gtx.program
def _copy_prog(
    a: gtx.Field[gtx.Dims[IDim], gtx.float64], out: gtx.Field[gtx.Dims[IDim], gtx.float64]
):
    _copy_op(a, out=out)


def test_translate_produces_cpp_source():
    """`Toolchain.translate` is the sanctioned partial run: frontend + translation only."""
    field_type = ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
    compile_time_args = arguments.CompileTimeArgs(
        args=(field_type, field_type),
        kwargs={},
        offset_provider={},
        column_axis=None,
        argument_descriptor_contexts={},
    )

    source = gtfn.run_gtfn.translate(_copy_prog.definition_stage, compile_time_args)

    assert isinstance(source, artifacts.ProgramSource)
    assert source.code_spec.file_extension == "cpp"
    assert source.entry_point.name == "_copy_prog"
    assert "_copy_prog" in source.source_code
