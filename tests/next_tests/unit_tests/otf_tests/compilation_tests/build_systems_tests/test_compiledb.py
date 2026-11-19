# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import os
import pathlib
import shutil
import tempfile
from unittest import mock

import pytest

from gt4py._core import definitions as core_defs
from gt4py.next import config, fingerprinting
from gt4py.next.otf import artifacts
from gt4py.next.otf.compilation import build_data, cache, compiler, importer
from gt4py.next.otf.compilation.build_systems import compiledb


@pytest.fixture
def clean_compiledb_cache(extension_source_example):
    cache_dir = cache.get_cache_folder(
        ext_source=extension_source_example,
        lifetime=config.BuildCacheLifetime.SESSION,
        build_context_id=fingerprinting.strict_fingerprinter(compiledb.CompiledbFactory()),
    )
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    yield


def test_default_compiledb_factory(extension_source_example, clean_compiledb_cache):
    otf_builder = compiledb.CompiledbFactory()(
        extension_source_example, cache_lifetime=config.BuildCacheLifetime.SESSION
    )

    # make sure the example project has not been written yet
    assert not build_data.contains_data(otf_builder.root_path)

    otf_builder.build()
    data = build_data.read_data(otf_builder.root_path)

    assert data.status == build_data.BuildStatus.COMPILED
    assert (otf_builder.root_path / data.module).exists()
    assert hasattr(
        importer.import_from_path(otf_builder.root_path / data.module), data.entry_point_name
    )

    assert (otf_builder.root_path / "build.sh").exists()


def test_compiledb_project_is_relocatable(extension_source_example, clean_compiledb_cache):
    builder = compiledb.CompiledbFactory()(
        extension_source_example, cache_lifetime=config.BuildCacheLifetime.SESSION
    )

    # make sure the example project has not been written yet
    assert not build_data.contains_data(builder.root_path)

    builder.build()

    with tempfile.TemporaryDirectory() as tmpdir:
        # copy the project to a new location
        relocated_dir = pathlib.Path(tmpdir) / "relocated"
        shutil.copytree(
            builder.root_path,
            relocated_dir,
            ignore=shutil.ignore_patterns("*.so", "*.dylib", "*.o"),
        )
        shutil.rmtree(builder.root_path)
        # make sure the orignal project has been removed
        assert not build_data.contains_data(builder.root_path)

        data = build_data.read_data(relocated_dir)
        assert data.status == build_data.BuildStatus.COMPILED
        # unset it's build status
        build_data.update_status(build_data.BuildStatus.CONFIGURED, relocated_dir)

        # rebuild (note this test uses internal API)
        empty_compiledb_project = compiledb.CompiledbProject(
            relocated_dir,
            source_files={},
            program_name="",
            compile_commands_cache="",
            bindings_file_name="",
        )
        empty_compiledb_project._run_build()

        new_data = build_data.read_data(relocated_dir)
        assert new_data.status == build_data.BuildStatus.COMPILED
        assert hasattr(
            importer.import_from_path(relocated_dir / new_data.module), new_data.entry_point_name
        )


def _with_code_spec(
    source: artifacts.ExtensionSource, code_spec: artifacts.CPPLikeCodeSpec
) -> artifacts.ExtensionSource:
    return dataclasses.replace(
        source, program_source=dataclasses.replace(source.program_source, code_spec=code_spec)
    )


def test_build_context_id_of_cpu_source_is_the_builder_fingerprint(extension_source_example):
    factory = compiledb.CompiledbFactory()
    with (
        mock.patch("gt4py._core.definitions.CUPY_DEVICE_TYPE", core_defs.DeviceType.CUDA),
        mock.patch.dict(os.environ, {"CUDAARCHS": "90"}),
    ):
        assert compiler.build_context_id(
            factory, extension_source_example
        ) == fingerprinting.strict_fingerprinter(factory)


def test_build_context_id_of_gpu_source_depends_on_device_arch(extension_source_example):
    factory = compiledb.CompiledbFactory()
    cuda_source = _with_code_spec(extension_source_example, artifacts.CUDACodeSpec())
    with mock.patch("gt4py._core.definitions.CUPY_DEVICE_TYPE", core_defs.DeviceType.CUDA):
        with mock.patch.dict(os.environ, {"CUDAARCHS": "80"}):
            id_80 = compiler.build_context_id(factory, cuda_source)
        with mock.patch.dict(os.environ, {"CUDAARCHS": "90"}):
            id_90 = compiler.build_context_id(factory, cuda_source)
            assert compiler.build_context_id(factory, cuda_source) == id_90

    assert id_80 != id_90


def test_compiledb_prototype_name_depends_on_device_arch(extension_source_example):
    factory = compiledb.CompiledbFactory()
    cuda_source = _with_code_spec(extension_source_example, artifacts.CUDACodeSpec())
    prototype_names = {}
    with (
        mock.patch("gt4py._core.definitions.CUPY_DEVICE_TYPE", core_defs.DeviceType.CUDA),
        mock.patch.object(compiledb, "_cc_get_compiledb") as get_compiledb,
    ):
        for arch in ("80", "90"):
            with mock.patch.dict(os.environ, {"CUDAARCHS": arch}):
                project = factory(cuda_source, cache_lifetime=config.BuildCacheLifetime.SESSION)
                context_id = compiler.build_context_id(factory, cuda_source)
            prototype_names[arch] = get_compiledb.call_args.args[1].entry_point.name
            assert project.root_path.name.endswith(context_id)

    assert prototype_names["80"] != prototype_names["90"]
    assert factory.cmake_extra_flags == []
