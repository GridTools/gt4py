# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import types

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import errors, where

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import cartesian_case
from next_tests.integration_tests.cases_utils import exec_alloc_descriptor


def test_constant_closure_vars_with_frozen_namespace(cartesian_case):
    from gt4py.eve.utils import FrozenNamespace

    constants = FrozenNamespace(PI=np.float64(3.142), E=np.float64(2.718))

    @gtx.field_operator
    def consume_constants(input: cases.IFloatField) -> cases.IFloatField:
        return constants.PI * constants.E * input

    cases.verify_with_default_data(
        cartesian_case, consume_constants, ref=lambda input: constants.PI * constants.E * input
    )


def test_constant_closure_vars_with_enums(cartesian_case):
    import enum

    class Constants(np.float64, enum.Enum):
        PI = 3.142
        E = 2.718

    @gtx.field_operator
    def consume_constants(input: cases.IFloatField) -> cases.IFloatField:
        return Constants.PI * Constants.E * input

    cases.verify_with_default_data(
        cartesian_case, consume_constants, ref=lambda input: Constants.PI * Constants.E * input
    )


def test_module_prefixed_builtin(cartesian_case):
    @gtx.field_operator
    def select(a: cases.IFloatField, b: cases.IFloatField) -> cases.IFloatField:
        return gtx.where(a > b, a, b)

    cases.verify_with_default_data(cartesian_case, select, ref=lambda a, b: np.where(a > b, a, b))


def test_aliased_builtin(cartesian_case):
    my_where = where

    @gtx.field_operator
    def select(a: cases.IFloatField, b: cases.IFloatField) -> cases.IFloatField:
        return my_where(a > b, a, b)

    cases.verify_with_default_data(cartesian_case, select, ref=lambda a, b: np.where(a > b, a, b))


def test_module_prefixed_type_constructor(cartesian_case):
    @gtx.field_operator
    def shifted(a: cases.IFloatField) -> cases.IFloatField:
        return a + gtx.float64(1)

    cases.verify_with_default_data(cartesian_case, shifted, ref=lambda a: a + 1.0)


def test_module_attribute_scalar_constant(cartesian_case):
    @gtx.field_operator
    def offset_by_pi(a: cases.IFloatField) -> cases.IFloatField:
        return a + np.pi

    cases.verify_with_default_data(cartesian_case, offset_by_pi, ref=lambda a: a + np.pi)


@gtx.field_operator
def _increment(a: cases.IFloatField) -> cases.IFloatField:
    return a + 1.0


@gtx.field_operator
def _double(a: cases.IFloatField) -> cases.IFloatField:
    return a + a


#: stand-in for a real helper module ('import helpers')
helpers_module = types.ModuleType("helpers_module")
helpers_module.increment = _increment
#: stand-in for a second module defining an operator with the same definition name
other_helpers_module = types.ModuleType("other_helpers_module")
other_helpers_module.increment = _double

increment_alias = _increment


def test_module_prefixed_field_operator(cartesian_case):
    @gtx.field_operator
    def wrapper(a: cases.IFloatField) -> cases.IFloatField:
        return helpers_module.increment(a)

    cases.verify_with_default_data(cartesian_case, wrapper, ref=lambda a: a + 1.0)


def test_aliased_field_operator(cartesian_case):
    @gtx.field_operator
    def wrapper(a: cases.IFloatField) -> cases.IFloatField:
        return increment_alias(a)

    cases.verify_with_default_data(cartesian_case, wrapper, ref=lambda a: a + 1.0)


def test_same_field_operator_under_two_names(cartesian_case):
    also_increment = increment_alias

    @gtx.field_operator
    def wrapper(a: cases.IFloatField) -> cases.IFloatField:
        return increment_alias(a) + also_increment(a)

    cases.verify_with_default_data(cartesian_case, wrapper, ref=lambda a: 2 * (a + 1.0))


def test_same_named_field_operators_from_different_modules(cartesian_case):
    @gtx.field_operator
    def use_first(a: cases.IFloatField) -> cases.IFloatField:
        return helpers_module.increment(a)

    @gtx.field_operator
    def use_second(a: cases.IFloatField) -> cases.IFloatField:
        return other_helpers_module.increment(a)

    @gtx.program
    def prog(a: cases.IFloatField, out1: cases.IFloatField, out2: cases.IFloatField):
        use_first(a, out=out1)
        use_second(a, out=out2)

    a = cases.allocate(cartesian_case, prog, "a")()
    out1 = cases.allocate(cartesian_case, prog, "out1")()
    out2 = cases.allocate(cartesian_case, prog, "out2")()
    cases.verify(
        cartesian_case,
        prog,
        a,
        out1,
        out2,
        inout=(out1, out2),
        ref=(a.asnumpy() + 1.0, a.asnumpy() + a.asnumpy()),
    )


def test_aliased_field_operator_in_program(cartesian_case):
    @gtx.program
    def prog(a: cases.IFloatField, out: cases.IFloatField):
        increment_alias(a, out=out)

    a = cases.allocate(cartesian_case, prog, "a")()
    out = cases.allocate(cartesian_case, prog, "out")()
    cases.verify(cartesian_case, prog, a, out, inout=out, ref=a.asnumpy() + 1.0)


def test_module_prefixed_builtin_shadowed_by_local_error():
    with pytest.raises(errors.DSLError, match="shadowed"):

        @gtx.field_operator
        def select(a: cases.IFloatField, b: cases.IFloatField) -> cases.IFloatField:
            where = a > b
            return gtx.where(where, a, b)
