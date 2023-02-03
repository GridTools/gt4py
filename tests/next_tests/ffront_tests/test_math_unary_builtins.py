# -*- coding: utf-8 -*-
# GT4Py - GridTools Framework
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

#
from gt4py.next.ffront.decorator import field_operator
from gt4py.next.ffront.fbuiltins import (
    Field,
    cbrt,
    ceil,
    cos,
    cosh,
    exp,
    float64,
    floor,
    int64,
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

from .ffront_test_utils import *


# Math builtins


def test_arithmetic(fieldview_backend):
    a_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    b_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))

    @field_operator(backend=fieldview_backend)
    def arithmetic(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        return (inp1 + inp2 / 3.0 - inp2) * 2.0

    arithmetic(a_I_float, b_I_float, out=out_I_float, offset_provider={})
    expected = (a_I_float.array() + b_I_float.array() / 3.0 - b_I_float.array()) * 2.0
    assert np.allclose(expected, out_I_float)


def test_power(fieldview_backend):
    a_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))

    @field_operator(backend=fieldview_backend)
    def pow(inp1: Field[[IDim], float64]) -> Field[[IDim], float64]:
        return inp1**2

    pow(a_I_float, out=out_I_float, offset_provider={})
    assert np.allclose(a_I_float.array() ** 2, out_I_float)


def test_floordiv(fieldview_backend):
    a_I_int = np_as_located_field(IDim)(np.random.randn(size).astype("int64"))
    out_I_int = np_as_located_field(IDim)(np.zeros((size,), dtype=int64))

    if fieldview_backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.skip("FloorDiv not yet supported.")

    @field_operator(backend=fieldview_backend)
    def floorDiv(inp1: Field[[IDim], int64]) -> Field[[IDim], int64]:
        return inp1 // 2

    floorDiv(a_I_int, out=out_I_int, offset_provider={})
    assert np.allclose(a_I_int.array() // 2, out_I_int)


def test_mod(fieldview_backend):
    a_I_int = np_as_located_field(IDim)(np.random.randn(size).astype("int64"))
    out_I_int = np_as_located_field(IDim)(np.zeros((size,), dtype=int64))

    if fieldview_backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.skip("Modulo not yet supported.")

    @field_operator(backend=fieldview_backend)
    def mod_fieldop(inp1: Field[[IDim], int64]) -> Field[[IDim], int64]:
        return inp1 % 2

    mod_fieldop(a_I_int, out=out_I_int, offset_provider={})
    assert np.allclose(a_I_int.array() % 2, out_I_int)


def test_bit_xor(fieldview_backend):
    a_I_bool = np_as_located_field(IDim)(np.random.randn(size).astype(bool))
    b_I_bool = np_as_located_field(IDim)(np.random.randn(size).astype(bool))
    out_I_bool = np_as_located_field(IDim)(np.zeros((size,), dtype=bool))

    @field_operator(backend=fieldview_backend)
    def binary_xor(inp1: Field[[IDim], bool], inp2: Field[[IDim], bool]) -> Field[[IDim], bool]:
        return inp1 ^ inp2

    binary_xor(a_I_bool, b_I_bool, out=out_I_bool, offset_provider={})
    assert np.allclose(a_I_bool.array() ^ b_I_bool.array(), out_I_bool)


def test_bit_and(fieldview_backend):
    a_I_bool = np_as_located_field(IDim)(np.random.randn(size).astype(bool))
    b_I_bool = np_as_located_field(IDim)(np.random.randn(size).astype(bool))
    out_I_bool = np_as_located_field(IDim)(np.zeros((size,), dtype=bool))

    @field_operator(backend=fieldview_backend)
    def bit_and(inp1: Field[[IDim], bool], inp2: Field[[IDim], bool]) -> Field[[IDim], bool]:
        return inp1 & inp2 & True

    bit_and(a_I_bool, b_I_bool, out=out_I_bool, offset_provider={})
    assert np.allclose(a_I_bool.array() & b_I_bool.array(), out_I_bool)


def test_bit_or(fieldview_backend):
    a_I_bool = np_as_located_field(IDim)(np.random.randn(size).astype(bool))
    b_I_bool = np_as_located_field(IDim)(np.random.randn(size).astype(bool))
    out_I_bool = np_as_located_field(IDim)(np.zeros((size,), dtype=bool))

    @field_operator(backend=fieldview_backend)
    def bit_or(inp1: Field[[IDim], bool], inp2: Field[[IDim], bool]) -> Field[[IDim], bool]:
        return inp1 | inp2 | True

    bit_or(a_I_bool, b_I_bool, out=out_I_bool, offset_provider={})
    assert np.allclose(a_I_bool.array() | b_I_bool.array(), out_I_bool)


# Unary builtins


def test_unary_neg(fieldview_backend):
    a_I_int = np_as_located_field(IDim)(np.random.randn(size).astype("int64"))
    out_I_int = np_as_located_field(IDim)(np.zeros((size,), dtype=int64))

    @field_operator(backend=fieldview_backend)
    def uneg(inp: Field[[IDim], int64]) -> Field[[IDim], int64]:
        return -inp

    uneg(a_I_int, out=out_I_int, offset_provider={})
    assert np.allclose(-a_I_int.array(), out_I_int)


def test_unary_invert(fieldview_backend):
    a_I_bool = np_as_located_field(IDim)(np.random.randn(size).astype(bool))
    out_I_bool = np_as_located_field(IDim)(np.zeros((size,), dtype=bool))

    @field_operator(backend=fieldview_backend)
    def tilde_fieldop(inp1: Field[[IDim], bool]) -> Field[[IDim], bool]:
        return ~inp1

    tilde_fieldop(a_I_bool, out=out_I_bool, offset_provider={})
    assert np.allclose(~a_I_bool.array(), out_I_bool)


def test_unary_not(fieldview_backend):
    a_I_bool = np_as_located_field(IDim)(np.random.randn(size).astype(bool))
    out_I_bool = np_as_located_field(IDim)(np.zeros((size,), dtype=bool))

    @field_operator(backend=fieldview_backend)
    def not_fieldop(inp1: Field[[IDim], bool]) -> Field[[IDim], bool]:
        return not inp1

    not_fieldop(a_I_bool, out=out_I_bool, offset_provider={})
    assert np.allclose(~a_I_bool.array(), out_I_bool)


# Trig builtins


def test_basic_trig(fieldview_backend):
    a_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    b_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))

    @field_operator(backend=fieldview_backend)
    def basic_trig_fieldop(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        return sin(cos(inp1)) - sinh(cosh(inp2)) + tan(inp1) - tanh(inp2)

    basic_trig_fieldop(a_I_float, b_I_float, out=out_I_float, offset_provider={})

    expected = (
        np.sin(np.cos(a_I_float))
        - np.sinh(np.cosh(b_I_float))
        + np.tan(a_I_float)
        - np.tanh(b_I_float)
    )
    assert np.allclose(expected, out_I_float)


def test_exp_log(fieldview_backend):
    a_float = np_as_located_field(IDim)(np.ones((size)))
    b_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))

    @field_operator(backend=fieldview_backend)
    def exp_log_fieldop(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        return log(inp1) - exp(inp2)

    exp_log_fieldop(a_float, b_I_float, out=out_I_float, offset_provider={})

    expected = np.log(a_float) - np.exp(b_I_float)
    assert np.allclose(expected, out_I_float)


def test_roots(fieldview_backend):
    a_float = np_as_located_field(IDim)(np.ones((size)))
    b_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))

    @field_operator(backend=fieldview_backend)
    def roots_fieldop(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        return sqrt(inp1) - cbrt(inp2)

    roots_fieldop(a_float, b_I_float, out=out_I_float, offset_provider={})

    expected = np.sqrt(a_float) - np.cbrt(b_I_float)
    assert np.allclose(expected, out_I_float)


def test_is_values(fieldview_backend):
    out_bool_1 = np_as_located_field(IDim)(np.zeros((size,), dtype=bool))
    out_bool_2 = np_as_located_field(IDim)(np.zeros((size,), dtype=bool))
    a_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_bool = np_as_located_field(IDim)(np.zeros((size,), dtype=bool))

    @field_operator(backend=fieldview_backend)
    def is_isinf_fieldop(inp1: Field[[IDim], float64]) -> Field[[IDim], bool]:
        return isinf(inp1)

    is_isinf_fieldop(a_I_float, out=out_I_bool, offset_provider={})
    expected = np.isinf(a_I_float)
    assert np.allclose(expected, out_I_bool)

    @field_operator(backend=fieldview_backend)
    def is_isnan_fieldop(inp1: Field[[IDim], float64]) -> Field[[IDim], bool]:
        return isnan(inp1)

    is_isnan_fieldop(a_I_float, out=out_bool_1, offset_provider={})
    expected = np.isnan(a_I_float)
    assert np.allclose(expected, out_bool_1)

    @field_operator(backend=fieldview_backend)
    def is_isfinite_fieldop(inp1: Field[[IDim], float64]) -> Field[[IDim], bool]:
        return isfinite(inp1)

    is_isfinite_fieldop(a_I_float, out=out_bool_2, offset_provider={})
    expected = np.isfinite(a_I_float)
    assert np.allclose(expected, out_bool_2)


def test_rounding_funs(fieldview_backend):
    a_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    b_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))

    @field_operator(backend=fieldview_backend)
    def rounding_funs_fieldop(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        return floor(inp1) - ceil(inp2) + trunc(inp1)

    rounding_funs_fieldop(a_I_float, b_I_float, out=out_I_float, offset_provider={})

    expected = np.floor(a_I_float) - np.ceil(b_I_float) + np.trunc(a_I_float)
    assert np.allclose(expected, out_I_float)
