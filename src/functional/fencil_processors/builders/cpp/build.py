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


import importlib
import json
import pathlib
import subprocess
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Sequence

import eve
from eve.codegen import JinjaTemplate as as_jinja, TemplatedGenerator
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


class CMakeListsGenerator(TemplatedGenerator):
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


@dataclass(frozen=True)
class CompileCommandProject(pipeline.BuildProject):
    """Use CMake to configure a valid compile command and then just compile."""

    source_module: source_modules.SourceModule[source_modules.LanguageWithHeaders]
    bindings_module: source_modules.BindingModule
    cache_strategy: cache.Strategy

    def get_compile_command(
        self, reconfigure: bool = False
    ) -> tuple[list[dict[str, str]], bool, pathlib.Path]:
        sentinel_source_module = source_modules.SourceModule(
            entry_point=source_modules.Function("cc_sentry", parameters=()),
            source_code="",
            library_deps=self.source_module.library_deps,
            language=self.source_module.language,
        )
        sentinel_binding_module = source_modules.BindingModule(
            source_code="", library_deps=self.bindings_module.library_deps
        )
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
        header_name = self.name + "." + self.source_module.language.include_extension
        bindings_name = (
            self.name + "_bindings" + "." + self.source_module.language.implementation_extension
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

            logfile.write_text("\n" + " ".join(cmd))

            with logfile.open(mode="a") as log_fp:
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


@dataclass(frozen=True)
class CMakeProject(pipeline.BuildProject):
    """Represent a CMake project for an externally compiled fencil."""

    source_module: source_modules.SourceModule[source_modules.LanguageWithHeaders]
    bindings_module: source_modules.BindingModule
    cache_strategy: cache.Strategy
    extra_cmake_flags: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.source_module.entry_point.name

    @property
    def sources(self) -> dict[str, str]:
        header_name = self.name + "." + self.source_module.language.include_extension
        bindings_name = (
            self.name + "_bindings" + "." + self.source_module.language.implementation_extension
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
                src_dir,
                "-B",
                build_dir,
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
