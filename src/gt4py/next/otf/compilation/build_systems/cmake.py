# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import os
import pathlib
import subprocess

from gt4py._core import definitions as core_defs
from gt4py.next import config
from gt4py.next.otf import languages, stages
from gt4py.next.otf.compilation import build_data, cache, common, compiler
from gt4py.next.otf.compilation.build_systems import cmake_lists


def get_device_arch() -> str | None:
    if core_defs.CUPY_DEVICE_TYPE == core_defs.DeviceType.CUDA:
        # use `cp` from core_defs to avoid trying to re-import cupy
        return core_defs.cp.cuda.Device(0).compute_capability  # type: ignore[attr-defined]
    elif core_defs.CUPY_DEVICE_TYPE == core_defs.DeviceType.ROCM:
        # TODO(egparedes): Implement this properly, either parsing the output of `$ rocminfo`
        # or using the HIP low level bindings.
        # Check: https://rocm.docs.amd.com/projects/hip-python/en/latest/user_guide/1_usage.html
        return "gfx90a"

    return None


def get_cmake_device_arch_option() -> str:
    cmake_flag = ""

    match core_defs.CUPY_DEVICE_TYPE:
        case core_defs.DeviceType.CUDA:
            device_archs = os.environ.get("CUDAARCHS", "").strip() or get_device_arch()
            cmake_flag = f"-DCMAKE_CUDA_ARCHITECTURES={device_archs}"
        case core_defs.DeviceType.ROCM:
            # `HIPARCHS` is not officially supported by CMake yet, but it might be in the future
            device_archs = os.environ.get("HIPARCHS", "").strip() or get_device_arch()
            cmake_flag = f"-DCMAKE_HIP_ARCHITECTURES={device_archs}"

    return cmake_flag


@dataclasses.dataclass
class CMakeFactory(
    compiler.BuildSystemProjectGenerator[
        languages.CPP | languages.CUDA | languages.HIP,
        languages.LanguageWithHeaderFilesSettings,
        languages.Python,
    ]
):
    """Create a CMakeProject from a ``CompilableSource`` stage object with given CMake settings."""

    cmake_generator_name: str = "Ninja"
    cmake_build_type: config.CMakeBuildType = config.CMakeBuildType.DEBUG
    cmake_extra_flags: list[str] = dataclasses.field(default_factory=list)

    def __call__(
        self,
        source: stages.CompilableSource[
            languages.CPP | languages.CUDA | languages.HIP,
            languages.LanguageWithHeaderFilesSettings,
            languages.Python,
        ],
        cache_lifetime: config.BuildCacheLifetime,
    ) -> CMakeProject:
        if not source.binding_source:
            raise NotImplementedError(
                "CMake build system project requires separate bindings code file."
            )
        name = source.program_source.entry_point.name
        header_name = f"{name}.{source.program_source.language_settings.header_extension}"
        bindings_name = f"{name}_bindings.{source.program_source.language_settings.file_extension}"
        cmake_languages = [cmake_lists.Language(name="CXX")]
        if (src_lang := source.program_source.language) in [languages.CUDA, languages.HIP]:
            cmake_languages = [*cmake_languages, cmake_lists.Language(name=src_lang.__name__)]
        cmake_lists_src = cmake_lists.generate_cmakelists_source(
            name,
            source.library_deps,
            [header_name, bindings_name],
            languages=cmake_languages,
        )

        if device_arch_flag := get_cmake_device_arch_option():
            self.cmake_extra_flags.append(device_arch_flag)

        return CMakeProject(
            root_path=cache.get_cache_folder(source, cache_lifetime),
            source_files={
                header_name: source.program_source.source_code,
                bindings_name: source.binding_source.source_code,
                "CMakeLists.txt": cmake_lists_src,
            },
            program_name=name,
            generator_name=self.cmake_generator_name,
            build_type=self.cmake_build_type,
            extra_cmake_flags=self.cmake_extra_flags,
        )


@dataclasses.dataclass
class CMakeProject(
    stages.BuildSystemProject[
        languages.CPP, languages.LanguageWithHeaderFilesSettings, languages.Python
    ]
):
    """
    CMake build system for gt4py programs.

    Write source files to disk and run cmake on them, keeping the build data updated.
    The ``.build()`` method runs all the necessary steps to compile the code to a python extension module.
    The location of the module can be found by accessing the ``gt4py.next.otf.compilation.build_data.BuildData``
    on the ``.root_path`` after building.
    """

    root_path: pathlib.Path
    source_files: dict[str, str]
    program_name: str
    generator_name: str = "Ninja"
    build_type: config.CMakeBuildType = config.CMakeBuildType.DEBUG
    extra_cmake_flags: list[str] = dataclasses.field(default_factory=list)

    def build(self) -> None:
        self._write_files()
        self._run_config()
        self._run_build()

    def _write_files(self) -> None:
        for name, content in self.source_files.items():
            (self.root_path / name).write_text(content, encoding="utf-8")

        build_data.write_data(
            build_data.BuildData(
                status=build_data.BuildStatus.INITIALIZED,
                module=pathlib.Path(
                    f"build/bin/{self.program_name}.{common.python_module_suffix()}"
                ),
                entry_point_name=self.program_name,
            ),
            self.root_path,
        )

        self._write_configure_script()

    @property
    def _config_command(self) -> list[str]:
        return [
            "cmake",
            "-G",
            self.generator_name,
            "-S",
            str(self.root_path),
            "-B",
            str(self.root_path / "build"),
            f"-DCMAKE_BUILD_TYPE={self.build_type.value}",
            *self.extra_cmake_flags,
        ]

    def _write_configure_script(self) -> None:
        # TODO(havogt): additionally we could store all `env` vars
        configure_script_path = self.root_path / "configure.sh"
        with configure_script_path.open("w") as build_script_pointer:
            build_script_pointer.write("#!/bin/sh\n")
            build_script_pointer.write(" ".join(self._config_command))
        try:
            configure_script_path.chmod(0o755)
        except OSError:
            # if setting permissions fails, it's not a problem
            pass

    def _run_config(self) -> None:
        logfile = self.root_path / "log_config.txt"
        with logfile.open(mode="w") as log_file_pointer:
            subprocess.check_call(
                self._config_command,
                stdout=log_file_pointer,
                stderr=log_file_pointer,
            )

        build_data.update_status(new_status=build_data.BuildStatus.CONFIGURED, path=self.root_path)

    def _run_build(self) -> None:
        logfile = self.root_path / "log_build.txt"
        with logfile.open(mode="w") as log_file_pointer:
            subprocess.check_call(
                ["cmake", "--build", self.root_path / "build"],
                stdout=log_file_pointer,
                stderr=log_file_pointer,
            )

        build_data.update_status(new_status=build_data.BuildStatus.COMPILED, path=self.root_path)
