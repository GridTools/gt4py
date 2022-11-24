# -*- coding: utf-8 -*-
#
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

from functional.ffront.decorator import field_operator
from functional.ffront.fbuiltins import (
    Field,
    cbrt,
    ceil,
    cos,
    cosh,
    exp,
    floor,
    isfinite,
    isinf,
    isnan,
    log,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    trunc,
)

from .test_ffront_utils import *


# Math builtins


def test_arithmetic(fieldview_backend):
    @field_operator(backend=fieldview_backend)
    def arithmetic(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        return (inp1 + inp2 / 3.0 - inp2) * 2.0

    arithmetic(a_float, b_float, out=out_float, offset_provider={})
    expected = (a_float.array() + b_float.array() / 3.0 - b_float.array()) * 2.0
    assert np.allclose(expected, out_float)


def test_power(fieldview_backend):
    @field_operator(backend=fieldview_backend)
    def pow(inp1: Field[[IDim], float64]) -> Field[[IDim], float64]:
        return inp1**2

    pow(a_float, out=out_float, offset_provider={})
    assert np.allclose(a_float.array() ** 2, out_float)


def test_floordiv(fieldview_backend):
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.skip("FloorDiv not yet supported.")

    @field_operator(backend=fieldview_backend)
    def floorDiv(inp1: Field[[IDim], int64]) -> Field[[IDim], int64]:
        return inp1 // 2

    floorDiv(a_int, out=out_int, offset_provider={})
    assert np.allclose(a_int.array() // 2, out_int)


def test_mod(fieldview_backend):
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.skip("Modulo not yet supported.")

    @field_operator(backend=fieldview_backend)
    def mod_fieldop(inp1: Field[[IDim], int64]) -> Field[[IDim], int64]:
        return inp1 % 2

    mod_fieldop(a_int, out=out_int, offset_provider={})
    assert np.allclose(a_int.array() % 2, out_int)


def test_bit_xor(fieldview_backend):
    @field_operator(backend=fieldview_backend)
    def binary_xor(inp1: Field[[IDim], bool], inp2: Field[[IDim], bool]) -> Field[[IDim], bool]:
        return inp1 ^ inp2

    binary_xor(a_bool, b_bool, out=out_bool, offset_provider={})
    assert np.allclose(a_bool.array() ^ b_bool.array(), out_bool)


def test_bit_and(fieldview_backend):
    @field_operator(backend=fieldview_backend)
    def bit_and(inp1: Field[[IDim], bool], inp2: Field[[IDim], bool]) -> Field[[IDim], bool]:
        return inp1 & inp2 & True

    bit_and(a_bool, b_bool, out=out_bool, offset_provider={})
    assert np.allclose(a_bool.array() & b_bool.array(), out_bool)


def test_bit_or(fieldview_backend):
    @field_operator(backend=fieldview_backend)
    def bit_or(inp1: Field[[IDim], bool], inp2: Field[[IDim], bool]) -> Field[[IDim], bool]:
        return inp1 | inp2 | True

    bit_or(a_bool, b_bool, out=out_bool, offset_provider={})
    assert np.allclose(a_bool.array() | b_bool.array(), out_bool)


# Unary builtins


def test_unary_neg(fieldview_backend):
    @field_operator(backend=fieldview_backend)
    def uneg(inp: Field[[IDim], int64]) -> Field[[IDim], int64]:
        return -inp

    uneg(a_int, out=out_int, offset_provider={})
    assert np.allclose(-a_int.array(), out_int)


def test_unary_invert(fieldview_backend):
    @field_operator(backend=fieldview_backend)
    def tilde_fieldop(inp1: Field[[IDim], bool]) -> Field[[IDim], bool]:
        return ~inp1

    tilde_fieldop(a_bool, out=out_bool, offset_provider={})
    assert np.allclose(~a_bool.array(), out_bool)


def test_unary_not(fieldview_backend):
    @field_operator(backend=fieldview_backend)
    def not_fieldop(inp1: Field[[IDim], bool]) -> Field[[IDim], bool]:
        return not inp1

    not_fieldop(a_bool, out=out_bool, offset_provider={})
    assert np.allclose(~a_bool.array(), out_bool)


# Trig builtins


def test_basic_trig(fieldview_backend):
    @field_operator(backend=fieldview_backend)
    def basic_trig_fieldop(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        return sin(cos(inp1)) - sinh(cosh(inp2)) + tan(inp1) - tanh(inp2)

    basic_trig_fieldop(a_float, b_float, out=out_float, offset_provider={})

    expected = (
        np.sin(np.cos(a_float)) - np.sinh(np.cosh(b_float)) + np.tan(a_float) - np.tanh(b_float)
    )
    assert np.allclose(expected, out_float)


def test_exp_log(fieldview_backend):
    a_float = np_as_located_field(IDim)(np.ones((size)))

    @field_operator(backend=fieldview_backend)
    def exp_log_fieldop(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        return log(inp1) - exp(inp2)

    exp_log_fieldop(a_float, b_float, out=out_float, offset_provider={})

    expected = np.log(a_float) - np.exp(b_float)
    assert np.allclose(expected, out_float)


def test_roots(fieldview_backend):
    a_float = np_as_located_field(IDim)(np.ones((size)))

    @field_operator(backend=fieldview_backend)
    def roots_fieldop(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        return sqrt(inp1) - cbrt(inp2)

    roots_fieldop(a_float, b_float, out=out_float, offset_provider={})

    expected = np.sqrt(a_float) - np.cbrt(b_float)
    assert np.allclose(expected, out_float)


def test_is_values(fieldview_backend):
    out_bool_1 = out_bool
    out_bool_2 = out_bool

    @field_operator(backend=fieldview_backend)
    def is_isinf_fieldop(inp1: Field[[IDim], float64]) -> Field[[IDim], bool]:
        return isinf(inp1)

    is_isinf_fieldop(a_float, out=out_bool, offset_provider={})
    expected = np.isinf(a_float)
    assert np.allclose(expected, out_bool)

    @field_operator(backend=fieldview_backend)
    def is_isnan_fieldop(inp1: Field[[IDim], float64]) -> Field[[IDim], bool]:
        return isnan(inp1)

    is_isnan_fieldop(a_float, out=out_bool_1, offset_provider={})
    expected = np.isnan(a_float)
    assert np.allclose(expected, out_bool_1)

    @field_operator(backend=fieldview_backend)
    def is_isfinite_fieldop(inp1: Field[[IDim], float64]) -> Field[[IDim], bool]:
        return isfinite(inp1)

    is_isfinite_fieldop(a_float, out=out_bool_2, offset_provider={})
    expected = np.isfinite(a_float)
    assert np.allclose(expected, out_bool_2)


def test_rounding_funs(fieldview_backend):
    @field_operator(backend=fieldview_backend)
    def rounding_funs_fieldop(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        return floor(inp1) - ceil(inp2) + trunc(inp1)

    rounding_funs_fieldop(a_float, b_float, out=out_float, offset_provider={})

    expected = np.floor(a_float) - np.ceil(b_float) + np.trunc(a_float)
    assert np.allclose(expected, out_float)
