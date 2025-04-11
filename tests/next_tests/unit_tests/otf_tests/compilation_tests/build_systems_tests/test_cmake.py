# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
import pathlib
from unittest import mock

from gt4py._core import definitions as core_defs
from gt4py.next import config
from gt4py.next.otf.compilation import build_data, importer
from gt4py.next.otf.compilation.build_systems import cmake


def test_get_cmake_device_arch_option_cuda():
    with (
        mock.patch("gt4py._core.definitions.CUPY_DEVICE_TYPE", core_defs.DeviceType.CUDA),
        mock.patch("gt4py.next.otf.compilation.build_systems.cmake.get_device_arch", lambda: "90"),
    ):
        # Test CUDA device without user-provided environment variable
        with mock.patch.dict(os.environ, {}):
            assert cmake.get_cmake_device_arch_option() == "-DCMAKE_CUDA_ARCHITECTURES=90"
        with mock.patch.dict(os.environ, {"CUDAARCHS": ""}):
            assert cmake.get_cmake_device_arch_option() == "-DCMAKE_CUDA_ARCHITECTURES=90"

        # Test CUDA device with CUDAARCHS environment variable
        with mock.patch.dict(os.environ, {"CUDAARCHS": "80", "HIPARCHS": "gfx90a"}):
            assert cmake.get_cmake_device_arch_option() == "-DCMAKE_CUDA_ARCHITECTURES=80"


def test_get_cmake_device_arch_option_rocm():
    with (
        mock.patch("gt4py._core.definitions.CUPY_DEVICE_TYPE", core_defs.DeviceType.ROCM),
        mock.patch(
            "gt4py.next.otf.compilation.build_systems.cmake.get_device_arch", lambda: "gfx942"
        ),
    ):
        # Test ROCM device without user-provided environment variable
        with mock.patch.dict(os.environ, {}):
            assert cmake.get_cmake_device_arch_option() == "-DCMAKE_HIP_ARCHITECTURES=gfx942"
        with mock.patch.dict(os.environ, {"HIPARCHS": ""}):
            assert cmake.get_cmake_device_arch_option() == "-DCMAKE_HIP_ARCHITECTURES=gfx942"

        # Test ROCM device with HIPARCHS environment variable
        with mock.patch.dict(os.environ, {"CUDAARCHS": "80", "HIPARCHS": "gfx90a"}):
            assert cmake.get_cmake_device_arch_option() == "-DCMAKE_HIP_ARCHITECTURES=gfx90a"


def test_default_cmake_factory(compilable_source_example, clean_example_session_cache):
    otf_builder = cmake.CMakeFactory()(
        source=compilable_source_example, cache_lifetime=config.BuildCacheLifetime.SESSION
    )
    assert not build_data.contains_data(otf_builder.root_path)

    otf_builder.build()

    data = build_data.read_data(otf_builder.root_path)

    assert data.status == build_data.BuildStatus.COMPILED
    assert pathlib.Path(otf_builder.root_path / data.module).exists()
    assert hasattr(
        importer.import_from_path(otf_builder.root_path / data.module), data.entry_point_name
    )

    assert (otf_builder.root_path / "configure.sh").exists()
