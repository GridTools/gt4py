# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import gt4py.next as gtx
import gt4py.next.type_system.type_specifications as ts
from gt4py.next import common, config
from gt4py.next.otf import artifacts
from gt4py.next.otf.binding import interface, nanobind

from next_tests.unit_tests.otf_tests.compilation_tests.build_systems_tests.conftest import (
    program_source_example,
)


def test_bindings(program_source_example):
    module = nanobind.create_bindings(
        program_source_example, config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE
    )
    assert module.library_deps[0].name == "nanobind"


_DYN = "gridtools::nanobind::dynamic_size"


@pytest.mark.parametrize(
    "grid_type, dims, has_unit_stride, expected_stride_spec",
    [
        # unstructured: unit stride on the horizontal dimension only, not on the vertical one
        (
            common.GridType.UNSTRUCTURED,
            [gtx.Dimension("Vertex"), gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)],
            True,
            f"1, {_DYN}",
        ),
        (
            common.GridType.UNSTRUCTURED,
            [gtx.Dimension("Vertex"), gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)],
            False,
            f"{_DYN}, {_DYN}",
        ),
        # cartesian: never unit stride, although both dimensions are horizontal
        (
            common.GridType.CARTESIAN,
            [gtx.Dimension("I"), gtx.Dimension("J")],
            True,
            f"{_DYN}, {_DYN}",
        ),
    ],
)
def test_bindings_static_strides(grid_type, dims, has_unit_stride, expected_stride_spec):
    program_source = artifacts.ProgramSource(
        entry_point=interface.Function(
            "stencil",
            parameters=(
                interface.Parameter(
                    name="buf",
                    type_=ts.FieldType(dims=dims, dtype=ts.ScalarType(ts.ScalarKind.FLOAT64)),
                ),
            ),
        ),
        source_code="",
        library_deps=(),
        code_spec=artifacts.CPPCodeSpec(),
        grid_type=grid_type,
    )

    bindings = nanobind.create_bindings(program_source, has_unit_stride)

    assert f"gridtools::nanobind::stride_spec<{expected_stride_spec}>{{}}" in bindings.source_code
