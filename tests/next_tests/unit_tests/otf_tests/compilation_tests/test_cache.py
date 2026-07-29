# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import re

import pytest

from gt4py.next import config, fingerprinting
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


@pytest.mark.parametrize("with_bindings", [True, False])
@pytest.mark.parametrize("with_context_id", [True, False])
def test_folder_name_round_trips_through_documented_pattern(
    extension_source_example, with_bindings, with_context_id
):
    """Names built by `get_cache_folder` must parse back through `CACHE_FOLDER_NAME_PATTERN`.

    External tooling (e.g. `scripts/python/dace_determinism.py`) recognizes cached
    program folders and recovers the program name through this pattern, so any
    change to the folder naming scheme must update the pattern together with it.
    """
    ext_source = (
        extension_source_example
        if with_bindings
        else dataclasses.replace(extension_source_example, binding_source=None)
    )
    context_id = (
        fingerprinting.strict_fingerprinter("build configuration") if with_context_id else ""
    )
    folder = cache.get_cache_folder(
        ext_source, config.BuildCacheLifetime.SESSION, build_context_id=context_id
    )

    match = re.fullmatch(cache.CACHE_FOLDER_NAME_PATTERN, folder.name)
    assert match is not None
    expected_name = ext_source.program_source.entry_point.name + ("_pyext" if with_bindings else "")
    assert match["name"] == expected_name
    assert match["version_id"] == config.BUILD_CACHE_VERSION_ID
    assert match["build_context_id"] == (context_id if with_context_id else None)
