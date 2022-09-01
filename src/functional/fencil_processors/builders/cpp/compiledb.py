import dataclasses
import json
import pathlib
import subprocess
from typing import Optional

from functional.fencil_processors import pipeline
from functional.fencil_processors.builders import cache
from functional.fencil_processors.builders.cpp import build, cmake, cmake_lists
from functional.fencil_processors.source_modules import source_modules


@dataclasses.dataclass
class CompiledbJITBuilder(pipeline.JITBuilder):
    root_path: pathlib.Path
    source_files: dict[str, str]
    fencil_name: str
    compile_commands_cache: pathlib.Path
    bindings_file_name: str

    def build(self):
        self.write_files()
        if build.data_from_jit_path(self.root_path)["status"] not in ["configured", "built"]:
            self.run_config()
        if build.data_from_jit_path(self.root_path)["status"] == "configured":
            self.run_build()

    def write_files(self):
        for name, content in self.source_files.items():
            (self.root_path / name).write_text(content, encoding="utf-8")

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
                last_entry = entry

        build.data_to_jit_path(
            build.data_from_jit_path(self.root_path)
            | {
                "status": "built",
                "extension": str(self.root_path / pathlib.Path(last_entry["output"])),
            },
            self.root_path,
        )

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

        build.data_to_jit_path(
            build.data_from_jit_path(self.root_path) | {"status": "configured"},
            self.root_path,
        )


def compiledb_builder_generator(
    cmake_build_type: str = "Debug",
    cmake_extra_flags: Optional[list[str]] = None,
    renew_compiledb: bool = False,
) -> pipeline.JITBuilderGenerator:
    def generate_compiledb_builder(
        jit_module: source_modules.JITSourceModule[
            source_modules.Cpp,
            source_modules.LanguageWithHeaderFilesSettings,
            source_modules.Python,
        ],
        cache_strategy: cache.Strategy,
    ) -> CompiledbJITBuilder:
        name = jit_module.source_module.entry_point.name
        header_name = f"{name}.{jit_module.source_module.language_settings.header_extension}"
        bindings_name = (
            f"{name}_bindings.{jit_module.source_module.language_settings.file_extension}"
        )

        cc_cache_module = _cc_cache_module(
            deps=jit_module.library_deps,
            build_type=cmake_build_type,
            cmake_flags=cmake_extra_flags or [],
        )

        if renew_compiledb or not (
            compiledb_template := _cc_get_compiledb(cc_cache_module, cache_strategy)
        ):
            compiledb_template = _cc_generate_compiledb(
                cc_cache_module,
                build_type=cmake_build_type,
                cmake_flags=cmake_extra_flags or [],
                cache_strategy=cache_strategy,
            )

        return CompiledbJITBuilder(
            root_path=build.jit_path_from_jit_module(jit_module, cache_strategy),
            fencil_name=name,
            source_files={
                header_name: jit_module.source_module.source_code,
                bindings_name: jit_module.bindings_module.source_code,
            },
            bindings_file_name=bindings_name,
            compile_commands_cache=compiledb_template,
        )

    return generate_compiledb_builder


def _cc_cache_name(
    deps: list[source_modules.LibraryDependency], build_type: str, flags: list[str]
) -> str:
    fencil_name = "compile_commands_cache"
    deps_str = "_".join(f"{dep.name}_{dep.version}" for dep in deps)
    flags_str = "_".join(flags)
    return "_".join([fencil_name, deps_str, build_type, flags_str]).replace(".", "_")


def _cc_cache_module(
    deps: list[source_modules.LibraryDependency],
    build_type: str,
    cmake_flags: list[str],
) -> pathlib.Path:
    name = _cc_cache_name(deps, build_type, cmake_flags)
    return source_modules.SourceModule(
        entry_point=source_modules.Function(name=name, parameters=[]),
        source_code="",
        library_deps=deps,
        language=source_modules.Cpp,
        language_settings=source_modules.LanguageWithHeaderFilesSettings(
            formatter_key="",
            formatter_style=None,
            file_extension="",
            header_extension="",
        ),
    )


def _cc_get_compiledb(
    source_module: source_modules.SourceModule, cache_strategy: cache.Strategy
) -> Optional[pathlib.Path]:
    cache_path = cache.get_cache_folder(source_module, cache_strategy)
    compile_db_path = cache_path / "compile_commands.json"
    if compile_db_path.exists():
        return compile_db_path
    return None


def _cc_generate_compiledb(
    source_module: source_modules.SourceModule,
    build_type: str,
    cmake_flags: list[str],
    cache_strategy: cache.Strategy,
) -> pathlib.Path:
    name = source_module.entry_point.name
    cache_path = cache.get_cache_folder(source_module, cache_strategy)

    jit_builder = cmake.CMakeJITBuilder(
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

    jit_builder.write_files()
    jit_builder.run_config()

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
