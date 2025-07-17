# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import shutil

import jinja2
import pytest

import gt4py.next as gtx
import gt4py.next.type_system.type_specifications as ts
from gt4py.next import config
from gt4py.next.otf import languages, stages
from gt4py.next.otf.binding import cpp_interface, interface, nanobind
from gt4py.next.otf.compilation import cache


def make_program_source(name: str) -> stages.ProgramSource:
    entry_point = interface.Function(
        name,
        parameters=[
            interface.Parameter(
                name="buf",
                type_=ts.FieldType(
                    dims=[gtx.Dimension("I"), gtx.Dimension("J")],
                    dtype=ts.ScalarType(ts.ScalarKind.FLOAT32),
                ),
            ),
            interface.Parameter(
                name="tup",
                type_=ts.TupleType(
                    types=[
                        ts.FieldType(
                            dims=[gtx.Dimension("I"), gtx.Dimension("J")],
                            dtype=ts.ScalarType(ts.ScalarKind.FLOAT32),
                        ),
                        ts.FieldType(
                            dims=[gtx.Dimension("I"), gtx.Dimension("J")],
                            dtype=ts.ScalarType(ts.ScalarKind.FLOAT32),
                        ),
                    ]
                ),
            ),
            interface.Parameter(name="sc", type_=ts.ScalarType(ts.ScalarKind.FLOAT32)),
        ],
        returns=True,
    )
    func = cpp_interface.render_function_declaration(
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

    return stages.ProgramSource(
        entry_point=entry_point,
        source_code=src,
        library_deps=[interface.LibraryDependency("gridtools_cpu", "master")],
        language=languages.CPP,
        language_settings=cpp_interface.CPP_DEFAULT,
        implicit_domain=False,
    )


@pytest.fixture
def program_source_with_name():
    yield make_program_source


@pytest.fixture
def program_source_example():
    return make_program_source("stencil")


@pytest.fixture
def compilable_source_example(program_source_example):
    return stages.CompilableSource(
        program_source=program_source_example,
        binding_source=nanobind.create_bindings(program_source_example),
    )


@pytest.fixture
def clean_example_session_cache(compilable_source_example):
    cache_dir = cache.get_cache_folder(compilable_source_example, config.BuildCacheLifetime.SESSION)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    yield
