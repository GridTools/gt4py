# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import dataclasses
import enum
import pathlib
import subprocess
from typing import Optional

from gt4py.next.otf import languages, stages
from gt4py.next.otf.compilation import build_data, cache, common, compiler
from gt4py.next.otf.compilation.build_systems import cmake_lists


class BuildType(enum.Enum):
    def _generate_next_value_(name, start, count, last_values):
        return "".join(part.capitalize() for part in name.split("_"))

    DEBUG = enum.auto()
    RELEASE = enum.auto()
    REL_WITH_DEB_INFO = enum.auto()
    MIN_SIZE_REL = enum.auto()


@dataclasses.dataclass
class CMakeFactory(
    compiler.BuildSystemProjectGenerator[
        languages.Cpp, languages.LanguageWithHeaderFilesSettings, languages.Python
    ]
):
    """Create a CMakeProject from a ``CompilableSource`` stage object with given CMake settings."""

    cmake_generator_name: str = "Ninja"
    cmake_build_type: BuildType = BuildType.DEBUG
    cmake_extra_flags: Optional[list[str]] = None

    def __call__(
        self,
        source: stages.CompilableSource[
            languages.Cpp,
            languages.LanguageWithHeaderFilesSettings,
            languages.Python,
        ],
        cache_strategy: cache.Strategy,
    ) -> CMakeProject:
        if not source.binding_source:
            raise NotImplementedError(
                "CMake build system project requires separate bindings code file."
            )
        name = source.program_source.entry_point.name
        header_name = f"{name}.{source.program_source.language_settings.header_extension}"
        bindings_name = f"{name}_bindings.{source.program_source.language_settings.file_extension}"
        return CMakeProject(
            root_path=cache.get_cache_folder(source, cache_strategy),
            source_files={
                header_name: source.program_source.source_code,
                bindings_name: source.binding_source.source_code,
                "CMakeLists.txt": cmake_lists.generate_cmakelists_source(
                    name,
                    source.library_deps,
                    [header_name, bindings_name],
                ),
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
    build_type: BuildType = BuildType.DEBUG
    extra_cmake_flags: list[str] = dataclasses.field(default_factory=list)

    def build(self):
        self._write_files()
        self._run_config()
        self._run_build()

    def _write_files(self):
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

    def _run_config(self):
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

    def _run_build(self):
        logfile = self.root_path / "log_build.txt"
        with logfile.open(mode="w") as log_file_pointer:
            subprocess.check_call(
                ["cmake", "--build", self.root_path / "build"],
                stdout=log_file_pointer,
                stderr=log_file_pointer,
            )

        build_data.update_status(new_status=build_data.BuildStatus.COMPILED, path=self.root_path)
