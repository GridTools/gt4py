# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

import pytest

from gt4py._core import definitions as core_defs
from gt4py.next import config
from gt4py.next.otf import artifacts
from gt4py.next.otf.compilation import cache, common, compiler, toolchain
from gt4py.next.otf.compilation.build_systems import compiledb

from next_tests.unit_tests.otf_tests.compilation_tests.build_systems_tests.conftest import (
    extension_source_example,
    program_source_example,
)


@pytest.fixture
def fresh_toolchain():
    toolchain.describe.cache_clear()
    yield
    toolchain.describe.cache_clear()


def _fake_compiler(tmp_path: pathlib.Path, name: str, version: str) -> str:
    path = tmp_path / name
    path.write_text(f"#!/bin/sh\necho '{version}'\n")
    path.chmod(0o755)
    return str(path)


def _cache_folder(source) -> pathlib.Path:
    return cache.get_cache_folder(
        source,
        config.BuildCacheLifetime.SESSION,
        compiler.cpp_build_context_id(compiledb.CompiledbFactory(), source),
    )


def test_same_toolchain_same_folder(fresh_toolchain, extension_source_example):
    folder = _cache_folder(extension_source_example)
    toolchain.describe.cache_clear()

    assert _cache_folder(extension_source_example) == folder


def test_changed_cxx_changes_folder(
    fresh_toolchain, monkeypatch, tmp_path, extension_source_example
):
    monkeypatch.setenv("CXX", _fake_compiler(tmp_path, "cxx_a", "fake c++ 1.0"))
    folder_a = _cache_folder(extension_source_example)
    toolchain.describe.cache_clear()
    monkeypatch.setenv("CXX", _fake_compiler(tmp_path, "cxx_b", "fake c++ 2.0"))
    folder_b = _cache_folder(extension_source_example)

    assert folder_a != folder_b
    assert dict(toolchain.describe("CXX"))["CXX"] == "fake c++ 2.0"


def test_changed_cxx_changes_compiledb_prototype(fresh_toolchain, monkeypatch, tmp_path):
    def prototype_folder() -> pathlib.Path:
        prototype = compiledb._cc_prototype_program_source(
            deps=(),
            build_type=config.CMakeBuildType.DEBUG,
            cmake_flags=[],
            code_spec=artifacts.CPPCodeSpec(),
        )
        return cache.get_cache_folder(
            artifacts.ExtensionSource(prototype, artifacts.BindingSource("", ())),
            config.BuildCacheLifetime.SESSION,
        )

    monkeypatch.setenv("CXX", _fake_compiler(tmp_path, "cxx_a", "fake c++ 1.0"))
    folder_a = prototype_folder()
    toolchain.describe.cache_clear()
    monkeypatch.setenv("CXX", _fake_compiler(tmp_path, "cxx_b", "fake c++ 2.0"))

    assert prototype_folder() != folder_a
    assert prototype_folder().name.startswith(compiledb.COMPILEDB_PROTOTYPE_NAME_PREFIX)


def test_cpu_ignores_device_toolchain(fresh_toolchain, monkeypatch, tmp_path):
    monkeypatch.setattr(common, "get_device_arch", lambda: "90")
    monkeypatch.setenv(
        "CUDACXX", _fake_compiler(tmp_path, "nvcc_a", "Cuda compilation tools, release 12.6")
    )
    cpu_a, cuda_a = toolchain.fingerprint("CXX"), toolchain.fingerprint("CUDA")
    toolchain.describe.cache_clear()
    monkeypatch.setenv(
        "CUDACXX", _fake_compiler(tmp_path, "nvcc_b", "Cuda compilation tools, release 12.9")
    )
    cpu_b, cuda_b = toolchain.fingerprint("CXX"), toolchain.fingerprint("CUDA")

    assert cpu_a == cpu_b
    assert cuda_a != cuda_b
    assert dict(toolchain.describe("CUDA"))["CUDA"] == "Cuda compilation tools, release 12.9"


def test_device_arch_changes_device_fingerprint(fresh_toolchain, monkeypatch):
    monkeypatch.setattr(common, "get_device_arch", lambda: "80")
    cuda_80 = toolchain.fingerprint("CUDA")
    toolchain.describe.cache_clear()
    monkeypatch.setattr(common, "get_device_arch", lambda: "90")

    assert toolchain.fingerprint("CUDA") != cuda_80


def test_distribution_version_changes_fingerprint(fresh_toolchain, monkeypatch):
    versions = {"nanobind": "2.0.0"}
    monkeypatch.setattr(toolchain, "_distribution_version", versions.get)
    before = toolchain.fingerprint("CXX", ("nanobind",))
    toolchain.describe.cache_clear()
    versions["nanobind"] = "2.1.0"

    assert toolchain.fingerprint("CXX", ("nanobind",)) != before
    assert toolchain.fingerprint("CXX") == toolchain.fingerprint("CXX", ())


@pytest.mark.parametrize(
    "device_type, language",
    [
        (core_defs.DeviceType.CPU, "CXX"),
        (core_defs.DeviceType.CUDA, "CUDA"),
        (core_defs.DeviceType.ROCM, "HIP"),
    ],
)
def test_device_language(device_type, language):
    assert toolchain.device_language(device_type) == language
