# flake8: noqa: F841
from typing import Callable, List, Tuple, TypedDict

import pytest

from gt4py.backend import from_name
from gt4py.gtscript import PARALLEL, Field, computation, interval, stencil
from gt4py.stencil_builder import StencilBuilder
from gtc.passes import gtir_definitive_assignment_analysis as daa


class TestData(TypedDict):
    valid: bool


# A list of dictionaries containing a stencil definition and the expected test case outputs
test_data: List[Tuple[Callable, TestData]] = []


def register_test_case(*, valid):
    def _wrapper(definition):
        global test_data
        test_data.append((definition, {"valid": valid}))
        return definition

    return _wrapper


# test cases
@register_test_case(valid=False)
def daa_0(in_field: Field[float], mask: Field[bool], out_field: Field[float]):
    with computation(PARALLEL):
        with interval(...):
            if mask:
                tmp = in_field
            out_field = tmp


@register_test_case(valid=False)
def daa_1(in_field: Field[float], mask: bool, out_field: Field[float]):
    with computation(PARALLEL):
        with interval(...):
            if mask:
                tmp = in_field
            out_field = tmp


@register_test_case(valid=True)
def daa_2(in_field: Field[float], mask: Field[bool], out_field: Field[float]):
    with computation(PARALLEL):
        with interval(...):
            if mask:
                tmp = in_field
                out_field = tmp


@register_test_case(valid=True)
def daa_3(in_field: Field[float], mask: Field[bool], out_field: Field[float]):
    with computation(PARALLEL):
        with interval(...):
            if mask:
                tmp = in_field
            else:
                tmp = in_field + 1
            out_field = tmp


@register_test_case(valid=False)
def daa_4(in_field: Field[float], mask: bool, out_field: Field[float]):
    with computation(PARALLEL):
        with interval(...):
            if mask:
                tmp = in_field
            if not mask:
                tmp = in_field + 1
            out_field = tmp


@register_test_case(valid=True)
def daa_5(in_field: Field[float], mask: Field[bool], out_field: Field[float]):
    with computation(PARALLEL):
        with interval(...):
            if mask:
                if not mask:
                    tmp = in_field
                else:
                    tmp = in_field + 1
            else:
                tmp = in_field + 2
            out_field = tmp


@register_test_case(valid=True)
def daa_6(in_field: Field[float], mask: Field[bool], out_field: Field[float]):
    with computation(PARALLEL):
        with interval(...):
            if mask:
                tmp = in_field
            tmp = in_field + 1
            out_field = tmp


@register_test_case(valid=True)
def daa_7(in_field: Field[float], mask: Field[bool], out_field: Field[float]):
    with computation(PARALLEL):
        with interval(...):
            if in_field > 0:
                tmp = in_field
            else:
                tmp = in_field + 1
            out_field = tmp


@pytest.mark.parametrize("definition,valid", [(s, d["valid"]) for s, d in test_data])
def test_daa(definition, valid):
    builder = StencilBuilder(definition, backend=from_name("debug"))
    gtir_stencil_expr = builder.gtir_pipeline.full()
    invalid_accesses = daa.analyse(gtir_stencil_expr)
    if valid:
        assert len(invalid_accesses) == 0
    else:
        assert len(invalid_accesses) == 1 and invalid_accesses[0].name == "tmp"


@pytest.mark.parametrize("definition", [s for s, d in test_data if not d["valid"]])
def test_daa_warn(definition):
    backend = "gtc:gt:cpu_ifirst"
    with pytest.warns(UserWarning, match="`tmp` may be uninitialized."):
        stencil(backend, definition)
