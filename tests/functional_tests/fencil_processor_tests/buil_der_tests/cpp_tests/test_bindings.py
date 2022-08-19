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

import pytest

from eve.codegen import format_source
from functional.fencil_processors.builders.cpp import bindings
from functional.fencil_processors.source_modules import source_modules


@pytest.fixture
def example_function():
    return defs.Function(
        name="example",
        parameters=[
            defs.BufferParameter("buf", ["I", "J"], ctypes.c_float),
            defs.ScalarParameter("sc", ctypes.c_float),
        ],
    )


def test_bindings(example_function):
    module = bindings.create_bindings(example_function, "example.hpp")
    expected_src = format_source(
        "cpp",
        """\
        #include "example.hpp"
        
        #include <gridtools/common/defs.hpp>
        #include <gridtools/fn/backend/naive.hpp>
        #include <gridtools/fn/cartesian.hpp>
        #include <gridtools/fn/unstructured.hpp>
        #include <gridtools/sid/rename_dimensions.hpp>
        #include <gridtools/storage/adapter/python_sid_adapter.hpp>
        #include <pybind11/pybind11.h>
        #include <pybind11/stl.h>
        
        decltype(auto) example_wrapper(pybind11::buffer buf, float sc) {
          return example(
              gridtools::sid::rename_numbered_dimensions<generated::I_t,
                                                         generated::J_t>(
                  gridtools::as_sid<float, 2, gridtools::integral_constant<int, 0>,
                                    999'999'999>(buf)),
              sc);
        }
        
        PYBIND11_MODULE(example, module) {
          module.doc() = "";
          module.def("example", &example_wrapper, "");
        }\
        """,
        style="LLVM",
    )
    assert module.library_deps[0].name == "pybind11"
    assert module.source_code == expected_src
