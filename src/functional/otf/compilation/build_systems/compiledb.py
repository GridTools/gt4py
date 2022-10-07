from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess
from typing import Optional

from functional.otf import languages, stages
from functional.otf.compilation import build_data, cache, compiler
from functional.otf.compilation.build_systems import cmake, cmake_lists
from functional.otf.source import source


@dataclasses.dataclass
class CompiledbFactory(
    compiler.BuildSystemProjectGenerator[
        languages.Cpp, languages.LanguageWithHeaderFilesSettings, languages.Python
    ]
):
    cmake_build_type: str = "Debug"
    cmake_extra_flags: Optional[list[str]] = None
    renew_compiledb: bool = False

    def __call__(
        self,
        source: stages.CompilableSource[
            languages.Cpp,
            languages.LanguageWithHeaderFilesSettings,
            languages.Python,
        ],
        cache_strategy: cache.Strategy,
    ) -> CompiledbProject:
        if not source.binding_source:
            raise compiler.CompilerError(
                "CMake build system project requires separate bindings code file."
            )
        name = source.program_source.entry_point.name
        header_name = f"{name}.{source.program_source.language_settings.header_extension}"
        bindings_name = f"{name}_bindings.{source.program_source.language_settings.file_extension}"

        cc_cache_module = _cc_cache_module(
            deps=source.library_deps,
            build_type=self.cmake_build_type,
            cmake_flags=self.cmake_extra_flags or [],
        )

        if self.renew_compiledb or not (
            compiledb_template := _cc_get_compiledb(cc_cache_module, cache_strategy)
        ):
            compiledb_template = _cc_generate_compiledb(
                cc_cache_module,
                build_type=self.cmake_build_type,
                cmake_flags=self.cmake_extra_flags or [],
                cache_strategy=cache_strategy,
            )

        return CompiledbProject(
            root_path=cache.get_cache_folder(source, cache_strategy),
            fencil_name=name,
            source_files={
                header_name: source.program_source.source_code,
                bindings_name: source.binding_source.source_code,
            },
            bindings_file_name=bindings_name,
            compile_commands_cache=compiledb_template,
        )


@dataclasses.dataclass()
class CompiledbProject(
    stages.BuildSystemProject[
        languages.Cpp, languages.LanguageWithHeaderFilesSettings, languages.Python
    ]
):
    root_path: pathlib.Path
    source_files: dict[str, str]
    fencil_name: str
    compile_commands_cache: pathlib.Path
    bindings_file_name: str

    def build(self):
        self.write_files()
        if build_data.read_data(self.root_path).status < build_data.BuildStatus.CONFIGURED:
            self.run_config()
        if (
            build_data.BuildStatus.CONFIGURED
            <= build_data.read_data(self.root_path).status
            < build_data.BuildStatus.COMPILED
        ):
            self.run_build()

    def write_files(self):
        for name, content in self.source_files.items():
            (self.root_path / name).write_text(content, encoding="utf-8")

        build_data.write_data(
            data=build_data.BuildData(
                status=build_data.BuildStatus.STARTED,
                module=pathlib.Path(""),
                entry_point_name=self.fencil_name,
            ),
            path=self.root_path,
        )

    def run_build(self):
        logfile = self.root_path / "log_build.txt"
        compile_db = json.loads((self.root_path / "compile_commands.json").read_text())
        assert compile_db
        with logfile.open(mode="w") as log_file_pointer:
            for entry in compile_db:
                log_file_pointer.write(entry["command"] + "\n")
                subprocess.check_call(
                    entry["command"],
                    cwd=self.root_path,
                    shell=True,
                    stdout=log_file_pointer,
                    stderr=log_file_pointer,
                )

        build_data.update_status(new_status=build_data.BuildStatus.COMPILED, path=self.root_path)

    def run_config(self):
        compile_db = json.loads(self.compile_commands_cache.read_text())

        (self.root_path / "build").mkdir(exist_ok=True)
        (self.root_path / "bin").mkdir(exist_ok=True)

        for entry in compile_db:
            for key, value in entry.items():
                entry[key] = (
                    value.replace("$NAME", self.fencil_name)
                    .replace("$BINDINGS_FILE", self.bindings_file_name)
                    .replace("$SRC_PATH", str(self.root_path))
                )

        (self.root_path / "compile_commands.json").write_text(json.dumps(compile_db))

        build_data.write_data(
            build_data.BuildData(
                status=build_data.BuildStatus.CONFIGURED,
                module=pathlib.Path(compile_db[-1]["output"]),
                entry_point_name=self.fencil_name,
            ),
            self.root_path,
        )


def _cc_cache_name(
    deps: tuple[source.LibraryDependency, ...], build_type: str, flags: list[str]
) -> str:
    fencil_name = "compile_commands_cache"
    deps_str = "_".join(f"{dep.name}_{dep.version}" for dep in deps)
    flags_str = "_".join(flags)
    return "_".join([fencil_name, deps_str, build_type, flags_str]).replace(".", "_")


def _cc_cache_module(
    deps: tuple[source.LibraryDependency, ...],
    build_type: str,
    cmake_flags: list[str],
) -> stages.ProgramSource:
    name = _cc_cache_name(deps, build_type, cmake_flags)
    return stages.ProgramSource(
        entry_point=source.Function(name=name, parameters=()),
        source_code="",
        library_deps=deps,
        language=languages.Cpp,
        language_settings=languages.LanguageWithHeaderFilesSettings(
            formatter_key="",
            formatter_style=None,
            file_extension="",
            header_extension="",
        ),
    )


def _cc_get_compiledb(
    source_module: stages.ProgramSource, cache_strategy: cache.Strategy
) -> Optional[pathlib.Path]:
    cache_path = cache.get_cache_folder(
        stages.CompilableSource(source_module, None), cache_strategy
    )
    compile_db_path = cache_path / "compile_commands.json"
    if compile_db_path.exists():
        return compile_db_path
    return None


def _cc_generate_compiledb(
    source_module: stages.ProgramSource,
    build_type: str,
    cmake_flags: list[str],
    cache_strategy: cache.Strategy,
) -> pathlib.Path:
    name = source_module.entry_point.name
    cache_path = cache.get_cache_folder(
        stages.CompilableSource(source_module, None), cache_strategy
    )

    otf_builder = cmake.CMakeProject(
        generator_name="Ninja",
        build_type=build_type,
        extra_cmake_flags=cmake_flags,
        root_path=cache_path,
        source_files={
            f"{name}.hpp": "",
            f"{name}.cpp": "",
            "CMakeLists.txt": cmake_lists.generate_cmakelists_source(
                name, source_module.library_deps, [f"{name}.hpp", f"{name}.cpp"]
            ),
        },
        fencil_name=name,
    )

    otf_builder.write_files()
    otf_builder.run_config()

    log_file = cache_path / "log_compiledb.txt"

    with log_file.open("w") as log_file_pointer:
        commands = json.loads(
            subprocess.check_output(
                ["ninja", "-t", "compdb"],
                cwd=cache_path / "build",
                stderr=log_file_pointer,
            ).decode("utf-8")
        )

    compile_db = [
        cmd for cmd in commands if name in pathlib.Path(cmd["file"]).stem and cmd["command"]
    ]

    assert compile_db

    for entry in compile_db:
        entry["directory"] = "$SRC_PATH"
        entry["command"] = (
            entry["command"]
            .replace(f"CMakeFiles/{name}.dir", "build")
            .replace(str(cache_path), "$SRC_PATH")
            .replace(f"{name}.cpp", "$BINDINGS_FILE")
            .replace(f"{name}", "$NAME")
            .replace("-I$SRC_PATH/build/_deps", f"-I{cache_path}/build/_deps")
        )
        entry["file"] = (
            entry["file"]
            .replace(f"CMakeFiles/{name}.dir", "build")
            .replace(str(cache_path), "$SRC_PATH")
            .replace(f"{name}.cpp", "$BINDINGS_FILE")
        )
        entry["output"] = (
            entry["output"]
            .replace(f"CMakeFiles/{name}.dir", "build")
            .replace(f"{name}.cpp", "$BINDINGS_FILE")
            .replace(f"{name}", "$NAME")
        )

    compile_db_path = cache_path / "compile_commands.json"
    compile_db_path.write_text(json.dumps(compile_db))
    return compile_db_path
