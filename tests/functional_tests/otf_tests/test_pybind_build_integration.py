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

from functional.fencil_processors import pipeline
from functional.fencil_processors.builders import cache, otf_compiler
from functional.fencil_processors.builders.cpp import bindings, cmake, compiledb


def test_gtfn_cpp_with_cmake(source_module_with_name):
    source_module_example = source_module_with_name("gtfn_cpp_with_cmake")
    workflow = pipeline.OTFWorkflow(
        bindings.source_module_to_otf_module,
        otf_compiler.OnTheFlyCompiler(
            cache_strategy=cache.Strategy.SESSION, builder_factory=cmake.make_cmake_factory()
        ),
    )
    otf_fencil = workflow(source_module_example)
    buf = np.zeros(shape=(6, 5), dtype=np.float32)
    sc = np.float32(3.1415926)
    res = otf_fencil(buf, sc)
    assert math.isclose(res, 6 * 5 * 3.1415926, rel_tol=1e-4)


def test_gtfn_cpp_with_compiledb(source_module_with_name):
    source_module_example = source_module_with_name("gtfn_cpp_with_compiledb")
    workflow = pipeline.OTFWorkflow(
        bindings.source_module_to_otf_module,
        otf_compiler.OnTheFlyCompiler(
            cache_strategy=cache.Strategy.SESSION,
            builder_factory=compiledb.make_compiledb_factory(),
        ),
    )
    otf_fencil = workflow(source_module_example)
    buf = np.zeros(shape=(6, 5), dtype=np.float32)
    sc = np.float32(3.1415926)
    res = otf_fencil(buf, sc)
    assert math.isclose(res, 6 * 5 * 3.1415926, rel_tol=1e-4)
