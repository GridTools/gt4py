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
from dataclasses import dataclass
from typing import Dict, Final, Optional, Sequence

from eve import Node
from eve.codegen import JinjaTemplate as as_jinja, TemplatedGenerator
from functional.fencil_processors import source_modules
from functional.fencil_processors.builders import cache
from functional.fencil_processors.builders.importer import import_from_path
from functional.fencil_processors.pipeline import BuildProject, SupportedLanguage


_BUILD_SUBDIR: Final = "build"


class FindDependency(Node):
    name: str
    version: str


class LinkDependency(Node):
    name: str
    target: str


class CMakeListsFile(Node):
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


class CMakeProject:
    """Work with CMake projects on the file system."""

    folder: Optional[pathlib.Path] = None
    name: str
    extension: str
    cmakelists: str
    sources: Dict[str, str]

    def __init__(
        self,
        name: str,
        dependencies: Sequence[source_modules.LibraryDependency],
        sources: Dict[str, str],
    ):
        self.name = name
        self.extension = _get_python_module_suffix()
        self.cmakelists = _render_cmakelists(name, dependencies, list(sources.keys()))
        self.sources = sources

    @staticmethod
    def get_binary(
        root_folder: pathlib.Path, name: str, extension: str = _get_python_module_suffix()
    ):
        return root_folder / _BUILD_SUBDIR / "bin" / (name + "." + extension)

    def get_current_binary(self) -> pathlib.Path:
        if not self.folder:
            raise RuntimeError("First you have to write the project to a folder.")

        return self.__class__.get_binary(self.folder, self.name, self.extension)

    def write(self, folder: pathlib.Path):
        (folder / "CMakeLists.txt").write_text(self.cmakelists, encoding="utf-8")
        for file_name, file_content in self.sources.items():
            (folder / file_name).write_text(file_content, encoding="utf-8")
        self.folder = folder

    def configure(self):
        if not self.folder:
            raise RuntimeError("First you have to write the project to a folder.")

        (self.folder / _BUILD_SUBDIR).mkdir(exist_ok=True)
        result = subprocess.run(
            [
                "cmake",
                "-G",
                "Ninja",
                "-S",
                self.folder,
                "-B",
                self.folder / _BUILD_SUBDIR,
                "-DCMAKE_BUILD_TYPE=Debug",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            error = result.stdout.decode()
            raise RuntimeError(error)

    def build(self):
        if not self.folder:
            raise RuntimeError("First you have to write the project to a folder.")

        result = subprocess.run(
            ["cmake", "--build", self.folder / _BUILD_SUBDIR],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            error = result.stdout.decode()
            raise RuntimeError(error)


# Could be frozen dataclass: state is encoded purely on FS and can always be overridden?
# If so, could be automatically hashable???
@dataclass(frozen=True)
class CMakeBuildProject(BuildProject):
    source_module: source_modules.SourceModule
    bindings_module: source_modules.BindingModule
    language: SupportedLanguage
    cache_strategy: cache.Strategy

    @property
    def name(self) -> str:
        return self.source_module.entry_point.name + "_" + hash(self)

    @property
    def dependencies(self) -> list[source_modules.LibraryDependency]:
        return [*self.source_module_library_deps, *self.bindings_module.library_deps]

    @property
    def sources(self) -> dict[str, str]:
        header_name = self.name + self.language.include_extension
        bindings_name = self.name + "_bindings" + self.language.implementation_extension
        return {
            header_name: self.source_module.source_code,
            bindings_name: self.bindings_module.source_code,
            "CMakeLists.txt": self.cmakelists_source,
        }

    @property
    def cmakelists_source(self) -> str:
        return _render_cmakelists(self.name, self.dependencies, list(self.sources.keys()))

    @property
    def src_dir(self) -> pathlib.Path:
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
        for name, content in self.sources.items():
            (root / name).write_text(content, encoding="utf-8")

        with (root / "gt4py_info.json").open() as info_fp:
            json.dump(
                self.__dict__
                | {"BuildProject": self.__class__.__module__ + "." + self.__class__.__name__},
                info_fp,
            )

    def is_configured(self) -> bool:
        return self.build_dir.exists and (self.build_dir / "CMakeCache.txt").exists()

    def configure(self):
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
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            error = result.stdout.decode()
            raise RuntimeError(error)

    def is_built(self):
        return self.binary_file.exists()

    def build(self):
        if not self.is_configured():
            self.configure()

        result = subprocess.run(
            ["cmake", "--build", self.folder / _BUILD_SUBDIR],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            error = result.stdout.decode()
            raise RuntimeError(error)

    def get_implementation(self):
        if not self.is_built():
            self.build(source_module.entry_point getattr(import_from_path(self.binary_file), self.name)
