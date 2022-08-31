# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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
"""Build system functionality."""


from __future__ import annotations

import dataclasses
import importlib
import json
import pathlib
import subprocess
import textwrap
from datetime import datetime
from typing import Callable, Optional, Sequence

import eve
from eve.codegen import JinjaTemplate as as_jinja
from functional.fencil_processors import pipeline
from functional.fencil_processors.builders import cache, importer
from functional.fencil_processors.source_modules import source_modules


class FindDependency(eve.Node):
    name: str
    version: str


class LinkDependency(eve.Node):
    name: str
    target: str


class CMakeListsFile(eve.Node):
    project_name: str
    find_deps: Sequence[FindDependency]
    link_deps: Sequence[LinkDependency]
    source_names: Sequence[str]
    bin_output_suffix: str


class CMakeListsGenerator(eve.codegen.TemplatedGenerator):
    CMakeListsFile = as_jinja(
        """
        project({{project_name}})
        cmake_minimum_required(VERSION 3.20.0)

        # Languages
        enable_language(CXX)

        # Paths
        list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
        list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

        # Find dependencies
        include(FetchContent)

        {{"\\n".join(find_deps)}}

        # Targets
        add_library({{project_name}} MODULE)

        target_compile_features({{project_name}} PRIVATE cxx_std_17)
        set_target_properties({{project_name}} PROPERTIES PREFIX "" SUFFIX ".{{bin_output_suffix}}")

        target_sources({{project_name}}
            PRIVATE
                {{"\\n".join(source_names)}}
        )

        # Link dependencies
        {{"\\n".join(link_deps)}}
        """
    )

    def visit_FindDependency(self, dep: FindDependency):
        match dep.name:
            case "pybind11":
                import pybind11

                return f"find_package(pybind11 CONFIG REQUIRED PATHS {pybind11.get_cmake_dir()})"
            case "gridtools":
                return textwrap.dedent(
                    """\
                    FetchContent_Declare(GridTools
                        GIT_REPOSITORY https://github.com/GridTools/gridtools.git
                        GIT_TAG        master
                    )
                    FetchContent_MakeAvailable(GridTools)\
                    """
                )
            case _:
                raise ValueError("Library {name} is not supported".format(name=dep.name))

    def visit_LinkDependency(self, dep: LinkDependency):
        match dep.name:
            case "pybind11":
                lib_name = "pybind11::module"
            case "gridtools":
                lib_name = "GridTools::fn_naive"
            case _:
                raise ValueError("Library {name} is not supported".format(name=dep.name))
        return "target_link_libraries({target} PUBLIC {lib})".format(
            target=dep.target, lib=lib_name
        )


def _render_cmakelists(
    project_name: str,
    dependencies: Sequence[source_modules.LibraryDependency],
    source_names: Sequence[str],
) -> str:
    cmakelists_file = CMakeListsFile(
        project_name=project_name,
        find_deps=[FindDependency(name=d.name, version=d.version) for d in dependencies],
        link_deps=[LinkDependency(name=d.name, target=project_name) for d in dependencies],
        source_names=source_names,
        bin_output_suffix=_get_python_module_suffix(),
    )
    return CMakeListsGenerator.apply(cmakelists_file)


def _get_python_module_suffix():
    return importlib.machinery.EXTENSION_SUFFIXES[0][1:]


# @todo adapt CompileCommandProject to design changes
# @todo clean up CompileCommandProject
@dataclasses.dataclass(frozen=True)
class CompileCommandProject(pipeline.BuildableProject):
    """Use CMake to configure a valid compile command and then just compile."""

    source_module: source_modules.SourceModule[
        source_modules.Cpp, source_modules.LanguageWithHeaderFilesSettings
    ]
    bindings_module: source_modules.BindingModule
    cache_strategy: cache.Strategy

    def get_compile_command(
        self, reconfigure: bool = False
    ) -> tuple[list[dict[str, str]], bool, pathlib.Path]:
        sentinel_source_module = dataclasses.replace(
            self.source_module,
            entry_point=source_modules.Function("cc_sentry", parameters=()),
            source_code="",
        )
        sentinel_binding_module = dataclasses.replace(self.bindings_module, source_code="")
        sentinel_project = CMakeProject(
            sentinel_source_module,
            sentinel_binding_module,
            self.cache_strategy,
        )

        if config_did_run := not sentinel_project.is_configured() or reconfigure:
            sentinel_project.configure()

            commands = json.loads(
                subprocess.check_output(
                    ["ninja", "-t", "compdb"],
                    cwd=sentinel_project.build_dir,
                    stderr=subprocess.STDOUT,
                ).decode("utf-8")
            )

            (sentinel_project.build_dir / "compile_commands.json").write_text(
                json.dumps(
                    [
                        cmd
                        for cmd in commands
                        if "cc_sentry" in pathlib.Path(cmd["file"]).stem and cmd["command"]
                    ]
                )
            )

        with (sentinel_project.src_dir / "build" / "compile_commands.json").open() as fp:
            result = json.load(fp)
            return result, config_did_run, sentinel_project.src_dir

    @property
    def name(self) -> str:
        return self.source_module.entry_point.name

    @property
    def src_dir(self) -> pathlib.Path:
        # TODO(ricoh): For any real caching, the source module, bindings module, type of build system
        #   have to be taken into account at least.
        return cache.get_cache_folder(self.source_module, self.cache_strategy)

    @property
    def binary_file(self) -> pathlib.Path:
        return self.src_dir / "bin" / (self.name + "." + _get_python_module_suffix())

    def build(self) -> None:
        header_name = self.name + "." + self.source_module.language_settings.header_extension
        bindings_name = (
            self.name + "_bindings" + "." + self.source_module.language_settings.file_extension
        )
        files = {
            header_name: self.source_module.source_code,
            bindings_name: self.bindings_module.source_code,
        }
        root = self.src_dir

        logfile = root / "log.txt"

        logfile.write_text(f"starting compilation at {datetime.now().isoformat()}")
        for name, content in files.items():
            (root / name).write_text(content, encoding="utf-8")

        compile_commands, _, sentry_root = self.get_compile_command()

        (root / "build").mkdir(exist_ok=True)
        (root / "bin").mkdir(exist_ok=True)

        for compile_command in compile_commands:
            cmd = []
            for item in compile_command["command"].split(" "):
                item = item.replace("CMakeFiles/cc_sentry.dir", "build")
                if str(sentry_root / "build" / "_deps") not in item:
                    item = item.replace(str(sentry_root), str(root))
                if "-I" not in item:
                    item = item.replace("cc_sentry", self.name)
                if item not in [":", "&&"]:
                    cmd.append(item)

            output = root / compile_command["output"].replace(
                "CMakeFiles/cc_sentry.dir", "build"
            ).replace("cc_sentry", self.name)

            with logfile.open(mode="a") as log_fp:
                log_fp.write("\n" + " ".join(cmd))
                subprocess.check_call(
                    " ".join(cmd), cwd=root, shell=True, stdout=log_fp, stderr=log_fp
                )

            if not output.exists():
                raise RuntimeError(f"command produced no output: {output} does not exists.")

    def is_built(self) -> bool:
        return self.binary_file.exists()

    def get_implementation(self) -> Callable:
        if not self.is_built():
            self.build()
        return getattr(importer.import_from_path(self.binary_file), self.name)


# @todo split CMakeProject into cachable jit module (data, caching ops) and jit builder (manage build system)
@dataclasses.dataclass(frozen=True)
class CMakeProject(pipeline.BuildableProject):
    """Represent a CMake project for an externally compiled fencil."""

    source_module: source_modules.SourceModule[
        source_modules.Cpp, source_modules.LanguageWithHeaderFilesSettings
    ]
    bindings_module: source_modules.BindingModule[source_modules.Cpp, source_modules.Python]
    cache_strategy: cache.Strategy
    extra_cmake_flags: list[str] = dataclasses.field(default_factory=list)

    @property
    def name(self) -> str:
        return self.source_module.entry_point.name

    @property
    def sources(self) -> dict[str, str]:
        header_name = f"{self.name}.{self.source_module.language_settings.header_extension}"
        bindings_name = (
            f"{self.name}_bindings.{self.source_module.language_settings.file_extension}"
        )
        return {
            header_name: self.source_module.source_code,
            bindings_name: self.bindings_module.source_code,
        }

    @property
    def files(self) -> dict[str, str]:
        return self.sources | {
            "CMakeLists.txt": self.cmakelists_source,
        }

    @property
    def cmakelists_source(self) -> str:
        return _render_cmakelists(
            self.name,
            [*self.source_module.library_deps, *self.bindings_module.library_deps],
            list(self.sources.keys()),
        )

    @property
    def src_dir(self) -> pathlib.Path:
        # TODO(ricoh): For any real caching, the source module, bindings module, type of build system
        #   have to be taken into account at least.
        return cache.get_cache_folder(self.source_module, self.cache_strategy)

    @property
    def build_dir(self) -> pathlib.Path:
        return self.src_dir / "build"

    @property
    def binary_file(self) -> pathlib.Path:
        return self.build_dir / "bin" / (self.name + "." + _get_python_module_suffix())

    def is_written(self) -> bool:
        return self.src_dir.exists and (self.src_dir / "CMakeLists.txt").exists()

    def write(self) -> None:
        root = self.src_dir
        for name, content in self.files.items():
            (root / name).write_text(content, encoding="utf-8")

    def is_configured(self) -> bool:
        return self.build_dir.exists and (self.build_dir / "CMakeCache.txt").exists()

    def configure(self) -> None:
        if not self.is_written():
            self.write()
        src_dir = self.src_dir
        build_dir = self.build_dir
        build_dir.mkdir(exist_ok=True)
        result = subprocess.run(
            [
                "cmake",
                "-G",
                "Ninja",
                "-S",
                str(src_dir),
                "-B",
                str(build_dir),
                "-DCMAKE_BUILD_TYPE=Debug",
                *self.extra_cmake_flags,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            error = result.stdout.decode()
            raise RuntimeError(error)

    def is_built(self) -> bool:
        return self.binary_file.exists()

    def build(self) -> None:
        if not self.is_configured():
            self.configure()

        result = subprocess.run(
            ["cmake", "--build", self.build_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            error = result.stdout.decode()
            raise RuntimeError(error)

    def get_implementation(self) -> Callable:
        if not self.is_built():
            self.build()
        return getattr(importer.import_from_path(self.binary_file), self.name)


def cache_dir_from_jit_module(
    jit_module: source_modules.JITCompileModule, cache_strategy: cache.Strategy
) -> pathlib.Path:
    # TODO: the output of this should depend also on at least the bindings module
    return cache.get_cache_folder(jit_module.source_module, cache_strategy)


def data_is_in_jit_path(jit_path: pathlib.Path) -> bool:
    return (jit_path / "gt4py.json").exists()


def data_from_jit_path(jit_path: pathlib.Path) -> dict:
    data_file = jit_path / "gt4py.json"
    if not data_file.exists():
        return {"status": "unknown"}
    return json.loads(data_file.read_text())


def data_to_jit_path(data: dict, jit_path: pathlib.Path):
    (jit_path / "gt4py.json").write_text(json.dumps(data))


def compiled_fencil_from_jit_path(jit_path: pathlib.Path) -> Optional[Callable]:
    data = data_from_jit_path(jit_path)
    if data["status"] != "built":  # @todo turn into enum or such
        return None
    return importer.import_from_path(jit_path / data["extension"])


def jit_module_to_compiled_fencil(
    jit_module: source_modules.JITCompileModule,
    jit_builder_generator: pipeline.JITBuilderGenerator,
    cache_strategy: cache.Strategy,
) -> Callable:
    jit_dir = cache_dir_from_jit_module(jit_module, cache_strategy)
    compiled_module = compiled_fencil_from_jit_path(jit_dir)
    if compiled_module:
        return getattr(compiled_module, jit_module.source_module.entry_point.name)
    jit_builder = jit_builder_generator(jit_module, cache_strategy)
    jit_builder.build()
    compiled_module = compiled_fencil_from_jit_path(jit_dir)
    if not compiled_module:
        raise AssertionError(
            "Build completed but no compiled python extension was found"
        )  # @todo: make safer, improve error msg
    return getattr(compiled_module, jit_module.source_module.entry_point.name)


def cmake_builder_generator(
    cmake_generator_name: str = "Ninja",
    cmake_build_type: str = "Debug",
    cmake_extra_flags: Optional[list[str]] = None,
) -> pipeline.JITBuilderGenerator:
    def generate_cmake_builder(
        jit_module: source_modules.JITCompileModule[
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
            root_path=cache_dir_from_jit_module(jit_module, cache_strategy),
            source_files={
                header_name: jit_module.source_module.source_code,
                bindings_name: jit_module.bindings_module.source_code,
                "CMakeLists.txt": _render_cmakelists(
                    name,
                    [
                        *jit_module.source_module.library_deps,
                        *jit_module.bindings_module.library_deps,
                    ],
                    [header_name, bindings_name],
                ),
            },
            fencil_name=name,
            generator_name=cmake_generator_name,
            build_type=cmake_build_type,
            extra_cmake_flags=cmake_extra_flags or [],
        )

    return generate_cmake_builder


def compile_command_builder_generator(
    cmake_build_type: str = "Debug",
    cmake_extra_flags: Optional[list[str]] = None,
    renew_compiledb: bool = False,
) -> pipeline.JITBuilderGenerator:
    def generate_compile_command_builder(
        jit_module: source_modules.JITCompileModule[
            source_modules.Cpp,
            source_modules.LanguageWithHeaderFilesSettings,
            source_modules.Python,
        ],
        cache_strategy: cache.Strategy,
    ) -> CompileCommandsJITBuilder:
        name = jit_module.source_module.entry_point.name
        header_name = f"{name}.{jit_module.source_module.language_settings.header_extension}"
        bindings_name = (
            f"{name}_bindings.{jit_module.source_module.language_settings.file_extension}"
        )

        cc_cache_module = _cc_cache_module(
            deps=_cc_deps_from_jit_module(jit_module),
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

        return CompileCommandsJITBuilder(
            root_path=cache_dir_from_jit_module(jit_module, cache_strategy),
            fencil_name=name,
            source_files={
                header_name: jit_module.source_module.source_code,
                bindings_name: jit_module.bindings_module.source_code,
            },
            bindings_file_name=bindings_name,
            compile_commands_cache=compiledb_template,
        )

    return generate_compile_command_builder


def _cc_deps_from_jit_module(
    jit_module: source_modules.JITCompileModule,
) -> list[source_modules.LibraryDependency]:
    return [
        *jit_module.source_module.library_deps,
        *jit_module.bindings_module.library_deps,
    ]


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

    jit_builder = CMakeJITBuilder(
        generator_name="Ninja",
        build_type=build_type,
        extra_cmake_flags=cmake_flags,
        root_path=cache_path,
        source_files={
            f"{name}.hpp": "",
            f"{name}.cpp": "",
            "CMakeLists.txt": _render_cmakelists(
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
        data = data_from_jit_path(self.root_path)
        data["status"] = "built"
        data["extension"] = str(
            self.root_path / "build" / "bin" / f"{self.fencil_name}.{_get_python_module_suffix()}"
        )
        data_to_jit_path(data, self.root_path)

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

        data = data_from_jit_path(self.root_path)
        data["status"] = "configured"
        data_to_jit_path(data, self.root_path)


@dataclasses.dataclass
class CompileCommandsJITBuilder(pipeline.JITBuilder):
    root_path: pathlib.Path
    source_files: dict[str, str]
    fencil_name: str
    compile_commands_cache: pathlib.Path
    bindings_file_name: str

    def build(self):
        self.write_files()
        if data_from_jit_path(self.root_path)["status"] not in ["configured", "built"]:
            self.run_config()
        if data_from_jit_path(self.root_path)["status"] == "configured":
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

        data_to_jit_path(
            data_from_jit_path(self.root_path)
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

        data_to_jit_path(
            data_from_jit_path(self.root_path) | {"status": "configured"},
            self.root_path,
        )
