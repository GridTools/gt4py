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


import math

import jinja2
import numpy as np
import pytest

from functional.fencil_processors.builders import cache
from functional.fencil_processors.builders.cpp import bindings, build
from functional.fencil_processors.source_modules import cpp_gen, source_modules


@pytest.fixture
def source_module_example():
    entry_point = source_modules.Function(
        "stencil",
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


def test_gtfn_cpp_with_cmake(source_module_example):
    wrapper = build.CMakeProject(
        source_module=source_module_example,
        bindings_module=bindings.create_bindings(source_module_example),
        cache_strategy=cache.Strategy.SESSION,
    ).get_implementation()
    buf = np.zeros(shape=(6, 5), dtype=np.float32)
    sc = np.float32(3.1415926)
    res = wrapper(buf, sc)
    assert math.isclose(res, 6 * 5 * 3.1415926, rel_tol=1e-4)
