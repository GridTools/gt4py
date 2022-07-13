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


import importlib
import pathlib
import subprocess
import textwrap
from typing import Dict, Final, Optional, Sequence

from eve import Node
from eve.codegen import JinjaTemplate, TemplatedGenerator
from functional.fencil_processors import defs


_build_subdir: Final = "build"


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
    CMakeListsFile = JinjaTemplate(
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
                        GIT_REPOSITORY https://github.com/havogt/gridtools.git
                        GIT_TAG        python_neighbor_table
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
    project_name: str, dependencies: Sequence[defs.LibraryDependency], source_names: Sequence[str]
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
    folder: Optional[pathlib.Path] = None
    name: str
    extension: str
    cmakelists: str
    sources: Dict[str, str]

    def __init__(
        self, name: str, dependencies: Sequence[defs.LibraryDependency], sources: Dict[str, str]
    ):
        self.name = name
        self.extension = _get_python_module_suffix()
        self.cmakelists = _render_cmakelists(name, dependencies, list(sources.keys()))
        self.sources = sources

    @staticmethod
    def get_binary(
        root_folder: pathlib.Path, name: str, extension: str = _get_python_module_suffix()
    ):
        return root_folder / _build_subdir / "bin" / (name + "." + extension)

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

        (self.folder / _build_subdir).mkdir(exist_ok=True)
        result = subprocess.run(
            [
                "cmake",
                "-G",
                "Ninja",
                "-S",
                self.folder,
                "-B",
                self.folder / _build_subdir,
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
            ["cmake", "--build", self.folder / _build_subdir],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            error = result.stdout.decode()
            raise RuntimeError(error)
