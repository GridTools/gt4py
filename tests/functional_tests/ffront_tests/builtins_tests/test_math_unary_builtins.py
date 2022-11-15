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
from typing import TypeVar

import numpy as np
import pytest

from functional.ffront.decorator import field_operator
from functional.ffront.fbuiltins import Dimension, Field, float64, int64
from functional.iterator.embedded import np_as_located_field
from functional.program_processors.runners import gtfn_cpu, roundtrip


@pytest.fixture(params=[roundtrip.executor, gtfn_cpu.run_gtfn])
def fieldview_backend(request):
    yield request.param


def debug_itir(tree):
    """Compare tree snippets while debugging."""
    from devtools import debug

    from eve.codegen import format_python_source
    from functional.program_processors import EmbeddedDSL

    debug(format_python_source(EmbeddedDSL.apply(tree)))


DimsType = TypeVar("DimsType")
DType = TypeVar("DType")

IDim = Dimension("IDim")
JDim = Dimension("JDim")

size = 10
a_bool = np_as_located_field(IDim)(np.random.randn(size).astype("bool"))
b_bool = np_as_located_field(IDim)(np.random.randn(size).astype("bool"))
out_bool = np_as_located_field(IDim)(np.zeros((size), dtype=bool))
a_int = np_as_located_field(IDim)(np.random.randn(size).astype("int64"))
out_int = np_as_located_field(IDim)(np.zeros((size), dtype=int64))
a_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
b_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
out_float = np_as_located_field(IDim)(np.zeros((size), dtype=float64))


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


# Trig functions
