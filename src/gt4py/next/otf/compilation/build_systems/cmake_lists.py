# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

from typing import Sequence

import gt4py.eve as eve
from gt4py.eve.codegen import JinjaTemplate as as_jinja
from gt4py.next.otf.binding import interface
from gt4py.next.otf.compilation import common


class FindDependency(eve.Node):
    name: str
    version: str


class LinkDependency(eve.Node):
    name: str
    target: str


class Language(eve.Node):
    name: str


class CMakeListsFile(eve.Node):
    project_name: str
    find_deps: Sequence[FindDependency]
    link_deps: Sequence[LinkDependency]
    source_names: Sequence[str]
    bin_output_suffix: str
    languages: Sequence[Language]


class CMakeListsGenerator(eve.codegen.TemplatedGenerator):
    CMakeListsFile = as_jinja(
        """
        cmake_minimum_required(VERSION 3.20.0)

        project({{project_name}})

        # Languages
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 60)
        endif()
        {{"\\n".join(languages)}}

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
        # TODO(ricoh): do not add more libraries here
        #   and do not use this design in a new build system.
        #   Instead, design this to be extensible (refer to ADR-0016).
        match dep.name:
            case "nanobind":
                import nanobind

                py = "find_package(Python COMPONENTS Interpreter Development REQUIRED)"
                nb = f"find_package(nanobind CONFIG REQUIRED PATHS {nanobind.cmake_dir()} NO_DEFAULT_PATHS)"
                return py + "\n" + nb
            case "gridtools_cpu" | "gridtools_gpu":
                import gridtools_cpp

                return f"find_package(GridTools REQUIRED PATHS {gridtools_cpp.get_cmake_dir()} NO_DEFAULT_PATH)"
            case _:
                raise ValueError(f"Library '{dep.name}' is not supported")

    def visit_LinkDependency(self, dep: LinkDependency):
        # TODO(ricoh): do not add more libraries here
        #   and do not use this design in a new build system.
        #   Instead, design this to be extensible (refer to ADR-0016).
        match dep.name:
            case "nanobind":
                lib_name = "nanobind-static"
            case "gridtools_cpu":
                lib_name = "GridTools::fn_naive"
            case "gridtools_gpu":
                lib_name = "GridTools::fn_gpu"
            case _:
                raise ValueError(f"Library '{dep.name}' is not supported")

        cfg = ""
        if dep.name == "nanobind":
            cfg = "\n".join(
                [
                    "nanobind_build_library(nanobind-static)",
                    f"nanobind_compile_options({dep.target})",
                    f"nanobind_link_options({dep.target})",
                ]
            )
        lnk = f"target_link_libraries({dep.target} PUBLIC {lib_name})"
        return cfg + "\n" + lnk

    Language = as_jinja("enable_language({{name}})")


def generate_cmakelists_source(
    project_name: str,
    dependencies: tuple[interface.LibraryDependency, ...],
    source_names: Sequence[str],
    languages: Sequence[Language] = (Language(name="CXX"),),
) -> str:
    """
    Generate CMakeLists file contents.

    Assumes the name of the gt4py program to be the same as the project name.
    """
    cmakelists_file = CMakeListsFile(
        project_name=project_name,
        find_deps=[FindDependency(name=d.name, version=d.version) for d in dependencies],
        link_deps=[LinkDependency(name=d.name, target=project_name) for d in dependencies],
        source_names=source_names,
        bin_output_suffix=common.python_module_suffix(),
        languages=languages,
    )
    return CMakeListsGenerator.apply(cmakelists_file)
