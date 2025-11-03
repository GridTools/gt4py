# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import (
    broadcast,
    cbrt,
    ceil,
    cos,
    cosh,
    exp,
    float64,
    floor,
    int32,
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

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IDim, cartesian_case, unstructured_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)


# Math builtins


def test_arithmetic(cartesian_case):
    @gtx.field_operator
    def arithmetic(inp1: cases.IFloatField, inp2: cases.IFloatField) -> gtx.Field[[IDim], float64]:
        return (inp1 + inp2 / 3.0 - inp2) * 2.0

    cases.verify_with_default_data(
        cartesian_case, arithmetic, ref=lambda inp1, inp2: (inp1 + inp2 / 3.0 - inp2) * 2.0
    )


def test_power(cartesian_case):
    @gtx.field_operator
    def pow(inp1: cases.IField) -> cases.IField:
        return inp1**2

    cases.verify_with_default_data(cartesian_case, pow, ref=lambda inp1: inp1**2)


@pytest.mark.uses_floordiv
def test_floordiv(cartesian_case):
    @gtx.field_operator
    def floorDiv(inp1: cases.IField) -> cases.IField:
        return inp1 // 2

    cases.verify_with_default_data(cartesian_case, floorDiv, ref=lambda inp1: inp1 // 2)


@pytest.mark.uses_negative_modulo
def test_mod(cartesian_case):
    @gtx.field_operator
    def mod_fieldop(inp1: cases.IField) -> cases.IField:
        return inp1 % 2

    inp1 = cartesian_case.as_field(
        [IDim], np.asarray(range(cartesian_case.default_sizes[IDim]), dtype=int32) - 5
    )
    out = cases.allocate(cartesian_case, mod_fieldop, cases.RETURN)()

    cases.verify(cartesian_case, mod_fieldop, inp1, out=out, ref=inp1 % 2)


def test_bit_xor(cartesian_case):
    @gtx.field_operator
    def binary_xor(inp1: cases.IBoolField, inp2: cases.IBoolField) -> cases.IBoolField:
        return inp1 ^ inp2

    size = cartesian_case.default_sizes[IDim]
    inp1 = cartesian_case.as_field([IDim], np.random.choice(a=[False, True], size=(size)))
    inp2 = cartesian_case.as_field([IDim], np.random.choice(a=[False, True], size=(size)))
    out = cases.allocate(cartesian_case, binary_xor, cases.RETURN)()
    cases.verify(cartesian_case, binary_xor, inp1, inp2, out=out, ref=inp1 ^ inp2)


def test_bit_and(cartesian_case):
    @gtx.field_operator
    def bit_and(inp1: cases.IBoolField, inp2: cases.IBoolField) -> cases.IBoolField:
        return inp1 & inp2

    size = cartesian_case.default_sizes[IDim]
    inp1 = cartesian_case.as_field([IDim], np.random.choice(a=[False, True], size=(size)))
    inp2 = cartesian_case.as_field([IDim], np.random.choice(a=[False, True], size=(size)))
    out = cases.allocate(cartesian_case, bit_and, cases.RETURN)()
    cases.verify(cartesian_case, bit_and, inp1, inp2, out=out, ref=inp1 & inp2)


def test_bit_or(cartesian_case):
    @gtx.field_operator
    def bit_or(inp1: cases.IBoolField, inp2: cases.IBoolField) -> cases.IBoolField:
        return inp1 | inp2

    size = cartesian_case.default_sizes[IDim]
    inp1 = cartesian_case.as_field([IDim], np.random.choice(a=[False, True], size=(size)))
    inp2 = cartesian_case.as_field([IDim], np.random.choice(a=[False, True], size=(size)))
    out = cases.allocate(cartesian_case, bit_or, cases.RETURN)()
    cases.verify(cartesian_case, bit_or, inp1, inp2, out=out, ref=inp1 | inp2)


# Unary builtins


def test_unary_neg(cartesian_case):
    @gtx.field_operator
    def uneg(inp: cases.IField) -> cases.IField:
        return -inp

    cases.verify_with_default_data(cartesian_case, uneg, ref=lambda inp1: -inp1)


def test_unary_pos(cartesian_case):
    @gtx.field_operator
    def upos(inp: cases.IField) -> cases.IField:
        return +inp

    cases.verify_with_default_data(cartesian_case, upos, ref=lambda inp1: inp1)


def test_unary_neg_float_conversion(cartesian_case):
    @gtx.field_operator
    def uneg_float() -> cases.IFloatField:
        inp_f = broadcast(float(-1), (IDim,))
        return inp_f

    size = cartesian_case.default_sizes[IDim]
    ref = cartesian_case.as_field([IDim], np.full(size, -1.0, dtype=float))
    out = cases.allocate(cartesian_case, uneg_float, cases.RETURN)()
    cases.verify(cartesian_case, uneg_float, out=out, ref=ref)


def test_unary_neg_bool_conversion(cartesian_case):
    @gtx.field_operator
    def uneg_bool() -> cases.IBoolField:
        inp_f = broadcast(bool(-1), (IDim,))
        return inp_f

    size = cartesian_case.default_sizes[IDim]
    ref = cartesian_case.as_field([IDim], np.full(size, True, dtype=bool))
    out = cases.allocate(cartesian_case, uneg_bool, cases.RETURN)()
    cases.verify(cartesian_case, uneg_bool, out=out, ref=ref)


def test_unary_invert(cartesian_case):
    @gtx.field_operator
    def tilde_fieldop(inp1: cases.IBoolField) -> cases.IBoolField:
        return ~inp1

    size = cartesian_case.default_sizes[IDim]
    inp1 = cartesian_case.as_field([IDim], np.random.choice(a=[False, True], size=(size)))
    out = cases.allocate(cartesian_case, tilde_fieldop, cases.RETURN)()
    cases.verify(cartesian_case, tilde_fieldop, inp1, out=out, ref=~inp1)


def test_unary_not(cartesian_case):
    pytest.xfail(
        "We accidentally supported 'not' on fields. This is wrong, we should raise an error."
    )
    with pytest.raises:  # TODO 'not' on a field should be illegal

        @gtx.field_operator
        def not_fieldop(inp1: cases.IBoolField) -> cases.IBoolField:
            return not inp1


# Trig builtins


def test_basic_trig(cartesian_case):
    @gtx.field_operator
    def basic_trig_fieldop(inp1: cases.IFloatField, inp2: cases.IFloatField) -> cases.IFloatField:
        return sin(cos(inp1)) - sinh(cosh(inp2)) + tan(inp1) - tanh(inp2)

    cases.verify_with_default_data(
        cartesian_case,
        basic_trig_fieldop,
        ref=lambda inp1, inp2: np.sin(np.cos(inp1))
        - np.sinh(np.cosh(inp2))
        + np.tan(inp1)
        - np.tanh(inp2),
    )


def test_exp_log(cartesian_case):
    @gtx.field_operator
    def exp_log_fieldop(inp1: cases.IFloatField, inp2: cases.IFloatField) -> cases.IFloatField:
        return log(inp1) - exp(inp2)

    cases.verify_with_default_data(
        cartesian_case, exp_log_fieldop, ref=lambda inp1, inp2: np.log(inp1) - np.exp(inp2)
    )


def test_roots(cartesian_case):
    @gtx.field_operator
    def roots_fieldop(inp1: cases.IFloatField, inp2: cases.IFloatField) -> cases.IFloatField:
        return sqrt(inp1) - cbrt(inp2)

    cases.verify_with_default_data(
        cartesian_case, roots_fieldop, ref=lambda inp1, inp2: np.sqrt(inp1) - np.cbrt(inp2)
    )


def test_is_values(cartesian_case):
    @gtx.field_operator
    def is_isinf_fieldop(inp1: cases.IFloatField) -> cases.IBoolField:
        return isinf(inp1)

    @gtx.field_operator
    def is_isnan_fieldop(inp1: cases.IFloatField) -> cases.IBoolField:
        return isnan(inp1)

    @gtx.field_operator
    def is_isfinite_fieldop(inp1: cases.IFloatField) -> cases.IBoolField:
        return isfinite(inp1)

    cases.verify_with_default_data(
        cartesian_case, is_isinf_fieldop, ref=lambda inp1: np.isinf(inp1)
    )

    cases.verify_with_default_data(
        cartesian_case, is_isnan_fieldop, ref=lambda inp1: np.isnan(inp1)
    )

    cases.verify_with_default_data(
        cartesian_case, is_isfinite_fieldop, ref=lambda inp1: np.isfinite(inp1)
    )


def test_rounding_funs(cartesian_case):
    @gtx.field_operator
    def rounding_funs_fieldop(
        inp1: cases.IFloatField, inp2: cases.IFloatField
    ) -> cases.IFloatField:
        return floor(inp1) - ceil(inp2) + trunc(inp1)

    cases.verify_with_default_data(
        cartesian_case,
        rounding_funs_fieldop,
        ref=lambda inp1, inp2: np.floor(inp1) - np.ceil(inp2) + np.trunc(inp1),
    )
