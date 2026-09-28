# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from gt4py.cartesian.backend.module_generator import (
    BaseModuleGenerator,
    ModuleData,
    make_args_data_from_gtir,
)
from gt4py.cartesian.definitions import AccessKind, Boundary, FieldInfo, ParameterInfo
from gt4py.cartesian.gtscript import PARALLEL, Field, computation, interval
from gt4py.cartesian.stencil_builder import StencilBuilder


class SampleModuleGenerator(BaseModuleGenerator):
    def __init__(self, builder: StencilBuilder) -> None:
        super().__init__(builder)

    def generate_implementation(self) -> str:
        return "pass"


def sample_stencil(in_field: Field[float]):  # type: ignore
    with computation(PARALLEL), interval(...):  # type: ignore
        in_field += 1  # type: ignore


@pytest.fixture
def sample_builder():
    yield StencilBuilder(sample_stencil)


@pytest.fixture
def sample_args_data():
    dtype = np.dtype(np.float64)
    yield ModuleData(
        field_info={
            "in_field": FieldInfo(
                access=AccessKind.READ_WRITE,
                boundary=Boundary.zeros(ndims=3),
                axes=("I", "J", "K"),
                data_dims=tuple([]),
                dtype=dtype,
            )
        },
        parameter_info={"param": ParameterInfo(access=AccessKind.READ, dtype=dtype)},
    )


def test_initialized_builder(sample_builder, sample_args_data):
    generator = SampleModuleGenerator(builder=sample_builder)
    assert generator.builder

    source = generator(args_data=sample_args_data)
    assert source


def sample_stencil_with_args(
    used_io_field: Field[float],  # type: ignore
    used_in_field: Field[float],  # type: ignore
    unused_field: Field[int],  # type: ignore
    used_scalar: float,  # type: ignore
    unused_scalar: bool,  # type: ignore
):
    with computation(PARALLEL), interval(...):  # type: ignore
        used_io_field = used_in_field[1, 0, 0] + used_scalar  # type: ignore


def test_module_data():
    builder = StencilBuilder(sample_stencil_with_args)
    module_data = make_args_data_from_gtir(builder.gtir_pipeline)

    assert module_data.field_info["used_io_field"].access == AccessKind.WRITE
    assert module_data.field_info["used_in_field"].access == AccessKind.READ
    assert module_data.field_info["unused_field"].access == AccessKind.NONE

    assert module_data.parameter_info["used_scalar"].access == AccessKind.READ
    assert module_data.parameter_info["unused_scalar"].access == AccessKind.NONE
