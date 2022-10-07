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
import math

import numpy as np

from functional.otf import workflow
from functional.otf.binding import pybind
from functional.otf.compilation import compiler
from functional.otf.compilation.build_systems import cmake, compiledb
from functional.program_processors.builders import cache


def test_gtfn_cpp_with_cmake(program_source_with_name):
    source_module_example = program_source_with_name("gtfn_cpp_with_cmake")
    build_the_program = workflow.Workflow(
        pybind.program_source_to_compilable_source,
        compiler.Compiler(
            cache_strategy=cache.Strategy.SESSION, builder_factory=cmake.CMakeFactory()
        ),
    )
    otf_fencil = build_the_program(source_module_example)
    buf = np.zeros(shape=(6, 5), dtype=np.float32)
    sc = np.float32(3.1415926)
    res = otf_fencil(buf, sc)
    assert math.isclose(res, 6 * 5 * 3.1415926, rel_tol=1e-4)


def test_gtfn_cpp_with_compiledb(program_source_with_name):
    source_module_example = program_source_with_name("gtfn_cpp_with_compiledb")
    build_the_program = workflow.Workflow(
        pybind.program_source_to_compilable_source,
        compiler.Compiler(
            cache_strategy=cache.Strategy.SESSION,
            builder_factory=compiledb.CompiledbFactory(),
        ),
    )
    otf_fencil = build_the_program(source_module_example)
    buf = np.zeros(shape=(6, 5), dtype=np.float32)
    sc = np.float32(3.1415926)
    res = otf_fencil(buf, sc)
    assert math.isclose(res, 6 * 5 * 3.1415926, rel_tol=1e-4)
