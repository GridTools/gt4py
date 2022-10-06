import dataclasses
import pathlib
import subprocess
from typing import Optional

from functional.otf import languages, stages, step_types
from functional.otf.compile import build_data, common, compiler
from functional.otf.compile.build_systems import cmake_lists
from functional.program_processors.builders import cache


def make_cmake_factory(
    cmake_generator_name: str = "Ninja",
    cmake_build_type: str = "Debug",
    cmake_extra_flags: Optional[list[str]] = None,
) -> step_types.BuildSystemProjectGenerator[
    languages.Cpp, languages.LanguageWithHeaderFilesSettings, languages.Python
]:
    def cmake_factory(
        source: stages.CompilableSource[
            languages.Cpp,
            languages.LanguageWithHeaderFilesSettings,
            languages.Python,
        ],
        cache_strategy: cache.Strategy,
    ) -> stages.BuildSystemProject[
        languages.Cpp, languages.LanguageWithHeaderFilesSettings, languages.Python
    ]:
        if not source.binding_source:
            raise compiler.CompilerError(
                "CMake build system project requires separate bindings code file."
            )
        name = source.program_source.entry_point.name
        header_name = f"{name}.{source.program_source.language_settings.header_extension}"
        bindings_name = f"{name}_bindings.{source.program_source.language_settings.file_extension}"
        return CMake(
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
            generator_name=cmake_generator_name,
            build_type=cmake_build_type,
            extra_cmake_flags=cmake_extra_flags or [],
        )

    return cmake_factory


@dataclasses.dataclass
class CMake(
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
            build_data.OTFBuildData(
                status=build_data.OTFBuildStatus.STARTED,
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

        build_data.update_status(new_status=build_data.OTFBuildStatus.COMPILED, path=self.root_path)

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

        build_data.update_status(
            new_status=build_data.OTFBuildStatus.CONFIGURED, path=self.root_path
        )
