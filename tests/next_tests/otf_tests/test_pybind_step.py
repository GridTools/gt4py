# GT4Py - GridTools Framework
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

from gt4py.next.otf.binding import interface, pybind


def test_bindings(program_source_example):
    module = pybind.create_bindings(program_source_example)
    expected_src = interface.format_source(
        program_source_example.language_settings,
        """\
        #include "stencil.cpp.inc"

        #include <gridtools/common/defs.hpp>
        #include <gridtools/fn/backend/naive.hpp>
        #include <gridtools/fn/cartesian.hpp>
        #include <gridtools/fn/unstructured.hpp>
        #include <gridtools/sid/rename_dimensions.hpp>
        #include <gridtools/storage/adapter/python_sid_adapter.hpp>
        #include <pybind11/pybind11.h>
        #include <pybind11/stl.h>

        decltype(auto) stencil_wrapper(pybind11::buffer buf, float sc) {
          return stencil(
              gridtools::sid::rename_numbered_dimensions<generated::I_t,
                                                         generated::J_t>(
                  gridtools::as_sid<float, 2, gridtools::integral_constant<int, 0>,
                                    999'999'999>(buf)),
              sc);
        }

        PYBIND11_MODULE(stencil, module) {
          module.doc() = "";
          module.def("stencil", &stencil_wrapper, "");
        }\
        """,
    )
    assert module.library_deps[0].name == "pybind11"
    assert module.source_code == expected_src
