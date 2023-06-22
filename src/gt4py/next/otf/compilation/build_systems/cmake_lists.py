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

import textwrap
from typing import Sequence

import gt4py.eve as eve
from gt4py.eve.codegen import JinjaTemplate as as_jinja
from gt4py.next.otf import languages
from gt4py.next.otf.binding import interface
from gt4py.next.otf.compilation import common


class FindDependency(eve.Node):
    name: str
    version: str
    libraries: list[str]


class LinkDependency(eve.Node):
    name: str
    target: str
    library: str


class ExtraLanguage(eve.Node):
    name: str


class CMakeListsFile(eve.Node):
    project_name: str
    find_deps: Sequence[FindDependency]
    link_deps: Sequence[LinkDependency]
    source_names: Sequence[str]
    bin_output_suffix: str
    extra_languages: Sequence[ExtraLanguage]


class CMakeListsGenerator(eve.codegen.TemplatedGenerator):
    # TODO(ricoh): enable_language(CUDA) has to be switchable
    CMakeListsFile = as_jinja(
        textwrap.dedent(
            """
        project({{project_name}})
        cmake_minimum_required(VERSION 3.20.0)

        # Languages
        enable_language(CXX)
        {{"\\n".join(extra_languages)}}

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
    )

    LinkDependency = as_jinja("target_link_libraries({{target}} PUBLIC {{library}})")

    ExtraLanguage = as_jinja("enable_language({{name}})")

    def visit_FindDependency(self, dep: FindDependency):
        match dep.name:
            case "pybind11":
                import pybind11

                return f"find_package(pybind11 CONFIG REQUIRED PATHS {pybind11.get_cmake_dir()} NO_DEFAULT_PATH)"
            case "gridtools":
                import gridtools_cpp

                return f"find_package(GridTools REQUIRED PATHS {gridtools_cpp.get_cmake_dir()} NO_DEFAULT_PATH)"
            case _:
                raise ValueError("Library {name} is not supported".format(name=dep.name))


def generate_cmakelists_source(
    project_name: str,
    dependencies: tuple[interface.LibraryDependency, ...],
    source_names: Sequence[str],
    language_settings: languages.LanguageWithHeaderFilesSettings,
) -> str:
    """
    Generate CMakeLists file contents.

    Assumes the name of the gt4py program to be the same as the project name.
    """
    findlibs: dict[tuple[str, str], FindDependency] = {}
    for dep in dependencies:
        findlib = findlibs.setdefault(
            (dep.name, dep.version),
            FindDependency(name=dep.name, version=dep.version, libraries=[]),
        )
        findlib.libraries.append(dep.library)

    cmakelists_file = CMakeListsFile(
        project_name=project_name,
        find_deps=list(findlibs.values()),
        link_deps=[
            LinkDependency(name=d.name, target=project_name, library=d.library)
            for d in dependencies
        ],
        source_names=source_names,
        bin_output_suffix=common.python_module_suffix(),
        extra_languages=[ExtraLanguage(name=n) for n in language_settings.extra_languages],
    )
    return CMakeListsGenerator.apply(cmakelists_file)
