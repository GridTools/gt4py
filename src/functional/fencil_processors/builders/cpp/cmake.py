import dataclasses
import pathlib
import subprocess
from typing import Optional

from functional.fencil_processors import pipeline
from functional.fencil_processors.builders import build_data, cache
from functional.fencil_processors.builders.cpp import cmake_lists, common
from functional.fencil_processors.source_modules import source_modules


def make_cmake_factory(
    cmake_generator_name: str = "Ninja",
    cmake_build_type: str = "Debug",
    cmake_extra_flags: Optional[list[str]] = None,
) -> pipeline.OTFBuilderGenerator:
    def cmake_factory(
        otf_module: source_modules.OTFSourceModule[
            source_modules.Cpp,
            source_modules.LanguageWithHeaderFilesSettings,
            source_modules.Python,
        ],
        cache_strategy: cache.Strategy,
    ) -> CMake:
        name = otf_module.source_module.entry_point.name
        header_name = f"{name}.{otf_module.source_module.language_settings.header_extension}"
        bindings_name = (
            f"{name}_bindings.{otf_module.source_module.language_settings.file_extension}"
        )
        return CMake(
            root_path=cache.get_cache_folder(otf_module, cache_strategy),
            source_files={
                header_name: otf_module.source_module.source_code,
                bindings_name: otf_module.bindings_module.source_code,
                "CMakeLists.txt": cmake_lists.generate_cmakelists_source(
                    name,
                    otf_module.library_deps,
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
    pipeline.OTFBuilder[
        source_modules.Cpp, source_modules.LanguageWithHeaderFilesSettings, source_modules.Python
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
