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
import shutil

import jinja2
import numpy as np
import pytest

import functional.common as common
import functional.type_system.type_specifications as ts
from functional.otf import languages, stages
from functional.otf.binding import cpp_interface, interface, pybind
from functional.otf.compilation import cache


IDim = common.Dimension("IDim")
JDim = common.Dimension("JDim")


def make_program_source(name: str) -> stages.ProgramSource:
    entry_point = interface.Function(
        name,
        parameters=[
            interface.Parameter(
                "buf",
                ts.FieldType(dims=[IDim, JDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32)),
            ),
            interface.Parameter("sc", ts.ScalarType(kind=ts.ScalarKind.FLOAT32)),
        ],
    )
    func = cpp_interface.render_function_declaration(
        entry_point,
        """\
        const auto xdim = gridtools::at_key<generated::IDim_t>(sid_get_upper_bounds(buf));
        const auto ydim = gridtools::at_key<generated::JDim_t>(sid_get_upper_bounds(buf));
        return xdim * ydim * sc;\
        """,
    )
    src = jinja2.Template(
        """\
        #include <gridtools/fn/cartesian.hpp>
        #include <gridtools/fn/unstructured.hpp>
        namespace generated {
        struct IDim_t {} constexpr inline IDim;
        struct JDim_t {} constexpr inline JDim;
        }
        {{func}}\
        """
    ).render(func=func)

    return stages.ProgramSource(
        entry_point=entry_point,
        source_code=src,
        library_deps=[
            interface.LibraryDependency("gridtools", "master"),
        ],
        language=languages.Cpp,
        language_settings=cpp_interface.CPP_DEFAULT,
    )


@pytest.fixture
def program_source_with_name():
    yield make_program_source


@pytest.fixture
def program_source_example():
    return make_program_source("stencil")


@pytest.fixture
def compilable_source_example(program_source_example):
    return stages.CompilableSource(
        program_source=program_source_example,
        binding_source=pybind.create_bindings(program_source_example),
    )


@pytest.fixture
def clean_example_session_cache(compilable_source_example):
    cache_dir = cache.get_cache_folder(compilable_source_example, cache.Strategy.SESSION)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    yield
