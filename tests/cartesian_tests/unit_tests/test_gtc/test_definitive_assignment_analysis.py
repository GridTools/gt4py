# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

import typing
from typing import Callable, List, Tuple, TypedDict

import pytest

from gt4py.cartesian.backend import from_name
from gt4py.cartesian.gtc.passes import gtir_definitive_assignment_analysis as daa
from gt4py.cartesian.gtscript import PARALLEL, Field, computation, interval, stencil
from gt4py.cartesian.stencil_builder import StencilBuilder


# A list of dictionaries containing a stencil definition and the expected test case outputs
test_data: List[Tuple[Callable, bool]] = []


def register_test_case(*, valid):
    def _wrapper(definition):
        global test_data
        test_data.append((definition, valid))
        return definition

    return _wrapper


@pytest.fixture
def clear_gtir_caches():
    from gt4py.cartesian.backend import module_generator
    from gt4py.cartesian.gtc.passes.gtir_pipeline import GtirPipeline

    GtirPipeline._cache.clear()
    module_generator._args_data_cache.clear()
    yield


# test cases
@typing.no_type_check
@register_test_case(valid=False)
def daa_0(in_field: Field[float], mask: Field[bool], out_field: Field[float]):
    """Invalid stencil with `tmp` undefined in one branch if a conditional on a boolean mask field"""
    with computation(PARALLEL):
        with interval(...):
            if mask:
                tmp = in_field
            out_field = tmp


@typing.no_type_check
@register_test_case(valid=False)
def daa_1(in_field: Field[float], mask: bool, out_field: Field[float]):
    """Invalid stencil with `tmp` undefined in one branch if a conditional on a boolean mask parameter"""
    with computation(PARALLEL):
        with interval(...):
            if mask:
                tmp = in_field
            out_field = tmp


@typing.no_type_check
@register_test_case(valid=True)
def daa_2(in_field: Field[float], mask: Field[bool], out_field: Field[float]):
    """Valid stencil with `tmp` defined in one branch and only used inside of that branch."""
    with computation(PARALLEL):
        with interval(...):
            if mask:
                tmp = in_field
                out_field = tmp


@typing.no_type_check
@register_test_case(valid=True)
def daa_3(in_field: Field[float], mask: Field[bool], out_field: Field[float]):
    """Valid stencil with `tmp` defined in both branches of a conditional on a boolean field and used outside."""
    with computation(PARALLEL):
        with interval(...):
            if mask:
                tmp = in_field
            else:
                tmp = in_field + 1
            out_field = tmp


@typing.no_type_check
@register_test_case(valid=False)
def daa_4(in_field: Field[float], mask: bool, out_field: Field[float]):
    """Valid stencil with `tmp` defined in both branches of a conditional on a boolean parameter and used outside."""
    with computation(PARALLEL):
        with interval(...):
            if mask:
                tmp = in_field
            if not mask:
                tmp = in_field + 1
            out_field = tmp


@typing.no_type_check
@register_test_case(valid=True)
def daa_5(in_field: Field[float], mask: Field[bool], out_field: Field[float]):
    """Valid stencil with `tmp` defined in every branch of a nested if-statement and used outside."""
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


@typing.no_type_check
@register_test_case(valid=True)
def daa_6(in_field: Field[float], mask: Field[bool], out_field: Field[float]):
    """Valid stencil with `tmp` defined in one branch only, but unconditionally overwritten outside before use."""
    with computation(PARALLEL):
        with interval(...):
            if mask:
                tmp = in_field
            tmp = in_field + 1
            out_field = tmp


@typing.no_type_check
@register_test_case(valid=True)
def daa_7(in_field: Field[float], out_field: Field[float]):
    """Valid stencil with `tmp` defined in both branches with a condition on a float field"""
    with computation(PARALLEL):
        with interval(...):
            if in_field > 0:
                tmp = in_field
            else:
                tmp = in_field + 1
            out_field = tmp


@typing.no_type_check
@register_test_case(valid=True)
def daa_8(in_field: Field[float], out_field: Field[float]):
    """Valid stencil with `tmp` defined in all three branches of if-elif-else statement"""
    with computation(PARALLEL):
        with interval(...):
            if in_field > 0:
                tmp = in_field
            elif in_field == 0:
                tmp = in_field + 1
            else:
                tmp = in_field + 2
            out_field = tmp


@typing.no_type_check
@register_test_case(valid=True)
def daa_9(in_field: Field[float], cond_field: Field[float], mask: bool, out_field: Field[float]):
    """Valid stencil with `tmp` defined in all branches of a nested if-elif-else statement"""
    with computation(PARALLEL):
        with interval(...):
            if in_field > 0:
                tmp = in_field
            elif in_field == 0:
                if in_field + 1 == 0:
                    tmp = in_field + 1
                else:
                    tmp = in_field + 2
            else:
                tmp = in_field + 3
            out_field = tmp


@typing.no_type_check
@register_test_case(valid=False)
def daa_10(in_field: Field[float], cond_field: Field[float], mask: bool, out_field: Field[float]):
    """Invalid stencil with `tmp` defined in all, but one branches of a nested if-elif-else statement"""
    with computation(PARALLEL):
        with interval(...):
            if in_field > 0:
                tmp = in_field
            elif in_field == 0:
                if in_field + 1 == 0:
                    tmp = in_field + 1
            else:
                tmp = in_field + 3
            out_field = tmp


@pytest.mark.parametrize("definition,valid", [(stencil, valid) for stencil, valid in test_data])
def test_daa(definition, valid, clear_gtir_caches):
    builder = StencilBuilder(definition, backend=from_name("debug"))
    gtir_stencil_expr = builder.gtir_pipeline.full()
    invalid_accesses = daa.analyze(gtir_stencil_expr)
    if valid:
        assert len(invalid_accesses) == 0
    else:
        assert len(invalid_accesses) == 1 and invalid_accesses[0].name == "tmp"


@pytest.mark.parametrize("definition", [stencil for stencil, valid in test_data if not valid])
def test_daa_warn(definition, clear_gtir_caches):
    backend = "gt:cpu_ifirst"
    with pytest.warns(UserWarning, match="`tmp` may be uninitialized."):
        stencil(backend, definition)
