import dataclasses
import pathlib
import subprocess
from typing import Optional

from functional.fencil_processors import pipeline
from functional.fencil_processors.builders import cache
from functional.fencil_processors.builders.cpp import build, cmake_lists
from functional.fencil_processors.source_modules import source_modules


def cmake_builder_generator(
    cmake_generator_name: str = "Ninja",
    cmake_build_type: str = "Debug",
    cmake_extra_flags: Optional[list[str]] = None,
) -> pipeline.JITBuilderGenerator:
    def generate_cmake_builder(
        jit_module: source_modules.JITSourceModule[
            source_modules.Cpp,
            source_modules.LanguageWithHeaderFilesSettings,
            source_modules.Python,
        ],
        cache_strategy: cache.Strategy,
    ) -> pipeline.JITBuilder:
        name = jit_module.source_module.entry_point.name
        header_name = f"{name}.{jit_module.source_module.language_settings.header_extension}"
        bindings_name = (
            f"{name}_bindings.{jit_module.source_module.language_settings.file_extension}"
        )
        return CMakeJITBuilder(
            root_path=build.jit_path_from_jit_module(jit_module, cache_strategy),
            source_files={
                header_name: jit_module.source_module.source_code,
                bindings_name: jit_module.bindings_module.source_code,
                "CMakeLists.txt": cmake_lists.generate_cmakelists_source(
                    name,
                    jit_module.library_deps,
                    [header_name, bindings_name],
                ),
            },
            fencil_name=name,
            generator_name=cmake_generator_name,
            build_type=cmake_build_type,
            extra_cmake_flags=cmake_extra_flags or [],
        )

    return generate_cmake_builder


@dataclasses.dataclass
class CMakeJITBuilder(pipeline.JITBuilder):
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

    def run_build(self):
        logfile = self.root_path / "log_build.txt"
        with logfile.open(mode="w") as log_file_pointer:
            subprocess.check_call(
                ["cmake", "--build", self.root_path / "build"],
                stdout=log_file_pointer,
                stderr=log_file_pointer,
            )
        data = build.data_from_jit_path(self.root_path)
        data["status"] = "built"
        data["extension"] = str(
            self.root_path / "build" / "bin" / f"{self.fencil_name}.{build.python_module_suffix()}"
        )
        build.data_to_jit_path(data, self.root_path)

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

        data = build.data_from_jit_path(self.root_path)
        data["status"] = "configured"
        build.data_to_jit_path(data, self.root_path)
