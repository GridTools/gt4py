# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest

from gt4py.backend.module_generator import (
    BaseModuleGenerator,
    ModuleData,
    make_args_data_from_gtir,
    make_args_data_from_iir,
)
from gt4py.gtscript import PARALLEL, Field, computation, interval
from gt4py.stencil_builder import StencilBuilder


class SampleModuleGenerator(BaseModuleGenerator):
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
    yield ModuleData(field_info={"in_field": None}, parameter_info={"inf_field": None})


def test_uninitialized_builder(sample_builder, sample_args_data):
    generator = SampleModuleGenerator()

    # if builder not passed in constructor, trying to access it is guaranteed to raise
    with pytest.raises(RuntimeError):
        assert generator.builder

    source = generator(args_data=sample_args_data, builder=sample_builder)
    assert source
    assert generator.builder


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
    # flake8: noqa
    with computation(PARALLEL), interval(...):  # type: ignore
        used_io_field = used_in_field[1, 0, 0] + used_scalar  # type: ignore


def test_module_data_equivalence():
    builder = StencilBuilder(sample_stencil_with_args)

    legacy_module_data = make_args_data_from_iir(builder.implementation_ir)
    gtc_module_data = make_args_data_from_gtir(builder.gtir_pipeline)

    assert legacy_module_data == gtc_module_data
