# -*- coding: utf-8 -*-
import pytest

from gt4py.backend.base import BaseModuleGenerator
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
    yield {"field_info": {"in_field": ""}, "parameter_info": {"inf_field": ""}}


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
