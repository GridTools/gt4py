# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import pathlib
import shutil
import tempfile

import pytest

from gt4py.next import config, fingerprinting
from gt4py.next.otf.compilation import build_data, cache, importer
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


def test_compiledb_prototype_ignores_format_source(monkeypatch, extension_source_example):
    prototypes = []

    def fake_get_compiledb(renew_compiledb, prototype_program_source, **kwargs):
        prototypes.append(prototype_program_source)
        return pathlib.Path("compile_commands.json")

    monkeypatch.setattr(compiledb, "_cc_get_compiledb", fake_get_compiledb)

    program_source = extension_source_example.program_source
    for format_source in (True, False):
        code_spec = dataclasses.replace(program_source.code_spec, format_source=format_source)
        compiledb.CompiledbFactory()(
            dataclasses.replace(
                extension_source_example,
                program_source=dataclasses.replace(program_source, code_spec=code_spec),
            ),
            cache_lifetime=config.BuildCacheLifetime.SESSION,
        )

    assert fingerprinting.strict_fingerprinter(
        prototypes[0]
    ) == fingerprinting.strict_fingerprinter(prototypes[1])
