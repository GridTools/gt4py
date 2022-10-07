from __future__ import annotations

import dataclasses
import pathlib
import subprocess
from typing import Optional

from functional.otf import languages, stages
from functional.otf.compilation import build_data, cache, common, compiler
from functional.otf.compilation.build_systems import cmake_lists


@dataclasses.dataclass
class CMakeFactory(
    compiler.BuildSystemProjectGenerator[
        languages.Cpp, languages.LanguageWithHeaderFilesSettings, languages.Python
    ]
):
    cmake_generator_name: str = "Ninja"
    cmake_build_type: str = "Debug"
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
            raise compiler.CompilerError(
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
            fencil_name=name,
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
    root_path: pathlib.Path
    source_files: dict[str, str]
    fencil_name: str
    generator_name: str = "Ninja"
    build_type: str = "Debug"
    extra_cmake_flags: list[str] = dataclasses.field(default_factory=list)

    def build(self):
        self.write_files()
        self.run_config()
        self.run_build()

    def write_files(self):
        for name, content in self.source_files.items():
            (self.root_path / name).write_text(content, encoding="utf-8")

        build_data.write_data(
            build_data.BuildData(
                status=build_data.BuildStatus.STARTED,
                module=pathlib.Path(
                    f"build/bin/{self.fencil_name}.{common.python_module_suffix()}"
                ),
                entry_point_name=self.fencil_name,
            ),
            self.root_path,
        )

    def run_build(self):
        logfile = self.root_path / "log_build.txt"
        with logfile.open(mode="w") as log_file_pointer:
            subprocess.check_call(
                ["cmake", "--build", self.root_path / "build"],
                stdout=log_file_pointer,
                stderr=log_file_pointer,
            )

        build_data.update_status(new_status=build_data.BuildStatus.COMPILED, path=self.root_path)

    def run_config(self):
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
                    f"-DCMAKE_BUILD_TYPE={self.build_type}",
                    *self.extra_cmake_flags,
                ],
                stdout=log_file_pointer,
                stderr=log_file_pointer,
            )

        build_data.update_status(new_status=build_data.BuildStatus.CONFIGURED, path=self.root_path)
