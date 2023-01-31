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

import math

import numpy as np

from gt4py.next.otf import workflow
from gt4py.next.otf.binding import pybind
from gt4py.next.otf.compilation import cache, compiler
from gt4py.next.otf.compilation.build_systems import cmake, compiledb


def test_gtfn_cpp_with_cmake(program_source_with_name):
    example_program_source = program_source_with_name("gtfn_cpp_with_cmake")
    build_the_program = workflow.Step(pybind.bind_source).chain(
        compiler.Compiler(
            cache_strategy=cache.Strategy.SESSION, builder_factory=cmake.CMakeFactory()
        ),
    )
    compiled_program = build_the_program(example_program_source)
    buf = np.zeros(shape=(6, 5), dtype=np.float32)
    sc = np.float32(3.1415926)
    res = compiled_program(buf, sc)
    assert math.isclose(res, 6 * 5 * 3.1415926, rel_tol=1e-4)


def test_gtfn_cpp_with_compiledb(program_source_with_name):
    example_program_source = program_source_with_name("gtfn_cpp_with_compiledb")
    build_the_program = workflow.Step(pybind.bind_source).chain(
        compiler.Compiler(
            cache_strategy=cache.Strategy.SESSION,
            builder_factory=compiledb.CompiledbFactory(),
        ),
    )
    compiled_program = build_the_program(example_program_source)
    buf = np.zeros(shape=(6, 5), dtype=np.float32)
    sc = np.float32(3.1415926)
    res = compiled_program(buf, sc)
    assert math.isclose(res, 6 * 5 * 3.1415926, rel_tol=1e-4)
