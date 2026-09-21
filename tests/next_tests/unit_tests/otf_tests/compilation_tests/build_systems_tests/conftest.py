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
from gt4py.next import common, config
from gt4py.next.otf import artifacts
from gt4py.next.otf.binding import cpp_interface, interface, nanobind
from gt4py.next.otf.compilation import cache


class I(gtx.DimensionIndex): ...


class J(gtx.DimensionIndex): ...


def make_program_source(name: str) -> artifacts.ProgramSource:
    entry_point = interface.Function(
        name,
        parameters=(
            interface.Parameter(
                name="buf",
                type_=ts.FieldType(
                    dims=[I, J],
                    dtype=ts.ScalarType(ts.ScalarKind.FLOAT32),
                ),
            ),
            interface.Parameter(
                name="tup",
                type_=ts.TupleType(
                    types=[
                        ts.FieldType(
                            dims=[I, J],
                            dtype=ts.ScalarType(ts.ScalarKind.FLOAT32),
                        ),
                        ts.FieldType(
                            dims=[I, J],
                            dtype=ts.ScalarType(ts.ScalarKind.FLOAT32),
                        ),
                    ]
                ),
            ),
            interface.Parameter(name="sc", type_=ts.ScalarType(ts.ScalarKind.FLOAT32)),
        ),
        returns=True,
    )
    # NOTE: the tag types are named after the *mangled* dimension tags, which is what the
    # generated bindings reference; a dimension's tag is its qualified Python name (ADR 0028).
    i_t, j_t = (common.codegen_name(d.tag) for d in (I, J))
    func = cpp_interface.render_function_declaration(
        entry_point,
        f"""\
        const auto xdim = gridtools::at_key<generated::{i_t}_t>(sid_get_upper_bounds(buf));
        const auto ydim = gridtools::at_key<generated::{j_t}_t>(sid_get_upper_bounds(buf));
        return xdim * ydim * sc;\
        """,
    )
    src = jinja2.Template(
        """\
        #include <gridtools/fn/cartesian.hpp>
        #include <gridtools/fn/unstructured.hpp>
        namespace generated {
        struct {{i_t}}_t {} constexpr inline {{i_t}};
        struct {{j_t}}_t {} constexpr inline {{j_t}};
        }
        {{func}}\
        """
    ).render(func=func, i_t=i_t, j_t=j_t)

    return artifacts.ProgramSource(
        entry_point=entry_point,
        source_code=src,
        library_deps=(interface.LibraryDependency("gridtools_cpu", "master"),),
        code_spec=artifacts.CPPCodeSpec(),
    )


@pytest.fixture
def program_source_with_name():
    yield make_program_source


@pytest.fixture
def program_source_example():
    return make_program_source("stencil")


@pytest.fixture
def extension_source_example(program_source_example):
    return artifacts.ExtensionSource(
        program_source=program_source_example,
        binding_source=nanobind.create_bindings(
            program_source_example, config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE
        ),
    )
