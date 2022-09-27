# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

from functional.fencil_processors.builders import cache
from functional.fencil_processors.builders.cpp import bindings
from functional.fencil_processors.source_modules import cpp_gen, source_modules


def make_source_module(name: str):
    entry_point = source_modules.Function(
        name,
        parameters=[
            source_modules.BufferParameter("buf", ("I", "J"), np.dtype(np.float32)),
            source_modules.ScalarParameter("sc", np.dtype(np.float32)),
        ],
    )
    func = cpp_gen.render_function_declaration(
        entry_point,
        """\
        const auto xdim = gridtools::at_key<generated::I_t>(sid_get_upper_bounds(buf));
        const auto ydim = gridtools::at_key<generated::J_t>(sid_get_upper_bounds(buf));
        return xdim * ydim * sc;\
        """,
    )
    src = jinja2.Template(
        """\
        #include <gridtools/fn/cartesian.hpp>
        #include <gridtools/fn/unstructured.hpp>
        namespace generated {
        struct I_t {} constexpr inline I;
        struct J_t {} constexpr inline J;
        }
        {{func}}\
        """
    ).render(func=func)

    return source_modules.SourceModule(
        entry_point=entry_point,
        source_code=src,
        library_deps=[
            source_modules.LibraryDependency("gridtools", "master"),
        ],
        language=source_modules.Cpp,
        language_settings=cpp_gen.CPP_DEFAULT,
    )


@pytest.fixture
def source_module_with_name(source_module_example):
    yield make_source_module


@pytest.fixture
def source_module_example():
    return make_source_module("stencil")


@pytest.fixture
def otf_module_example(source_module_example):
    return source_modules.OTFSourceModule(
        source_module=source_module_example,
        bindings_module=bindings.create_bindings(source_module_example),
    )


@pytest.fixture
def clean_example_session_cache(otf_module_example):
    cache_dir = cache.get_cache_folder(otf_module_example, cache.Strategy.SESSION)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    yield
