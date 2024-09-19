# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import pathlib
import subprocess
from typing import Optional

from gt4py.next import config
from gt4py.next.otf import languages, stages
from gt4py.next.otf.compilation import build_data, cache, common, compiler
from gt4py.next.otf.compilation.build_systems import cmake_lists


@dataclasses.dataclass
class CMakeFactory(
    compiler.BuildSystemProjectGenerator[
        languages.Cpp | languages.Cuda, languages.LanguageWithHeaderFilesSettings, languages.Python
    ]
):
    """Create a CMakeProject from a ``CompilableSource`` stage object with given CMake settings."""

    cmake_generator_name: str = "Ninja"
    cmake_build_type: config.CMakeBuildType = config.CMakeBuildType.DEBUG
    cmake_extra_flags: Optional[list[str]] = None

    def __call__(
        self,
        source: stages.CompilableSource[
            languages.Cpp | languages.Cuda,
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
        if source.program_source.language is languages.Cuda:
            cmake_languages = [*cmake_languages, cmake_lists.Language(name="HIP")]
        cmake_lists_src = cmake_lists.generate_cmakelists_source(
            name, source.library_deps, [header_name, bindings_name], languages=cmake_languages
        )
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
            extra_cmake_flags=self.cmake_extra_flags or [],
        )


@dataclasses.dataclass
class CMakeProject(
    stages.BuildSystemProject[
        languages.Cpp, languages.LanguageWithHeaderFilesSettings, languages.Python
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

    def _run_config(self) -> None:
        logfile = self.root_path / "log_config.txt"
        with logfile.open(mode="w") as log_file_pointer:
            subprocess.check_call(
                [
                    "cmake",
                    "-G",
                    self.generator_name,
                    "-S",
                    str(self.root_path),
                    "-B",
                    str(self.root_path / "build"),
                    f"-DCMAKE_BUILD_TYPE={self.build_type.value}",
                    *self.extra_cmake_flags,
                ],
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
