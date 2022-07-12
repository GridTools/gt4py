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


import ctypes
import math

import jinja2
import numpy as np
import pytest

from functional.fencil_processors import cpp, defs
from functional.fencil_processors.callables.cpp import callable


@pytest.fixture
def source_module():
    entry_point = defs.Function(
        "stencil",
        parameters=[
            defs.BufferParameter("buf", ["I", "J"], ctypes.c_float),
            defs.ScalarParameter("sc", ctypes.c_float),
        ],
    )
    func = cpp.render_function_declaration(
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

    return defs.SourceCodeModule(
        entry_point=entry_point,
        source_code=src,
        library_deps=[
            defs.LibraryDependency("gridtools", "master"),
        ],
        language=cpp.language_id,
    )


def test_callable(source_module):
    wrapper = callable.create_callable(source_module)
    buf = np.zeros(shape=(6, 5), dtype=np.float32)
    sc = np.float32(3.1415926)
    res = wrapper(buf, sc)
    assert math.isclose(res, 6 * 5 * 3.1415926, rel_tol=1e-4)
