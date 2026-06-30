# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

from gt4py.next import config
from gt4py.next.otf.compilation import cache

from next_tests.unit_tests.otf_tests.compilation_tests.build_systems_tests.conftest import (
    extension_source_example,
    program_source_example,
)


def test_cache_folder_includes_program_name_and_version_id(extension_source_example):
    cache_dir = cache.get_cache_folder(extension_source_example, config.BuildCacheLifetime.SESSION)

    assert extension_source_example.program_source.entry_point.name in cache_dir.name
    assert config.BUILD_CACHE_VERSION_ID in cache_dir.name


def test_version_id_busts_cache_folder(monkeypatch, extension_source_example):
    folder_a = cache.get_cache_folder(extension_source_example, config.BuildCacheLifetime.SESSION)

    monkeypatch.setattr(
        config, "BUILD_CACHE_VERSION_ID", f"{config.BUILD_CACHE_VERSION_ID}-changed"
    )
    folder_b = cache.get_cache_folder(extension_source_example, config.BuildCacheLifetime.SESSION)

    # A different version id must land the build in a different folder, even though the
    # extension source (and therefore its fingerprint) is unchanged.
    assert folder_a != folder_b


def test_pyext_marker_only_when_bindings_present(extension_source_example):
    with_bindings = cache.get_cache_folder(
        extension_source_example, config.BuildCacheLifetime.SESSION
    )
    assert "_pyext" in with_bindings.name

    without_bindings = cache.get_cache_folder(
        dataclasses.replace(extension_source_example, binding_source=None),
        config.BuildCacheLifetime.SESSION,
    )
    assert "_pyext" not in without_bindings.name


def test_build_context_id_busts_cache_folder(extension_source_example):
    folder_a = cache.get_cache_folder(extension_source_example, config.BuildCacheLifetime.SESSION)
    folder_b = cache.get_cache_folder(
        extension_source_example, config.BuildCacheLifetime.SESSION, build_context_id="ctx"
    )

    assert folder_a != folder_b
    assert folder_b.name.endswith("_ctx")
