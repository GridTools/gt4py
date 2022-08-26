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


import dataclasses
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


def test_gt_cpp_with_cmake(source_module_example):
    wrapper = build.CMakeProject(
        source_module=source_module_example,
        bindings_module=bindings.create_bindings(source_module_example),
        cache_strategy=cache.Strategy.SESSION,
    ).get_implementation()
    buf = np.zeros(shape=(6, 5), dtype=np.float32)
    sc = np.float32(3.1415926)
    res = wrapper(buf, sc)
    assert math.isclose(res, 6 * 5 * 3.1415926, rel_tol=1e-4)


def test_gt_cpp_with_compile_command(source_module_example):
    wrapper = (
        foo := build.CompileCommandProject(
            source_module=source_module_example,
            bindings_module=bindings.create_bindings(source_module_example),
            cache_strategy=cache.Strategy.SESSION,
        )
    ).get_implementation()
    log = (foo.src_dir / "log.txt").read_text()
    assert len(log.splitlines()) >= 2
    buf = np.zeros(shape=(6, 5), dtype=np.float32)
    sc = np.float32(3.1415926)
    res = wrapper(buf, sc)
    assert math.isclose(res, 6 * 5 * 3.1415926, rel_tol=1e-4)


def test_compile_command_only_configures_once(source_module_example):
    first_cc = build.CompileCommandProject(
        source_module=source_module_example,
        bindings_module=bindings.create_bindings(source_module_example),
        cache_strategy=cache.Strategy.SESSION,
    )

    _, config_did_run, _ = first_cc.get_compile_command(reconfigure=True)

    assert config_did_run is True

    changed_source_module = dataclasses.replace(
        source_module_example,
        entry_point=source_modules.Function(
            "test_compile_command_only_configure_once_fencil",
            parameters=source_module_example.entry_point.parameters,
        ),
    )

    second_cc = build.CompileCommandProject(
        source_module=changed_source_module,
        bindings_module=bindings.create_bindings(changed_source_module),
        cache_strategy=cache.Strategy.SESSION,
    )

    assert second_cc.get_compile_command()[1] is False
