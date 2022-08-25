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

import numpy as np
import pytest

from functional.fencil_processors.builders.cpp import bindings
from functional.fencil_processors.source_modules import cpp_gen, source_modules


@pytest.fixture
def example_source_module():
    return source_modules.SourceModule(
        entry_point=source_modules.Function(
            name="example",
            parameters=[
                source_modules.BufferParameter("buf", ("I", "J"), np.dtype(ctypes.c_float)),
                source_modules.ScalarParameter("sc", np.dtype(ctypes.c_float)),
            ],
        ),
        source_code="",
        library_deps=[],
        language=source_modules.Cpp,
        language_settings=cpp_gen.CPP_DEFAULT,
    )


def test_bindings(example_source_module):
    module = bindings.create_bindings(example_source_module)
    expected_src = source_modules.format_source(
        example_source_module.language_settings,
        """\
        #include "example.cpp.inc"
        
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
    )
    assert module.library_deps[0].name == "pybind11"
    assert module.source_code == expected_src
