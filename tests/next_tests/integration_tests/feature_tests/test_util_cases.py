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

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import errors

import next_tests.exclusion_matrices as definitions
from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (  # noqa: F401 # fixtures
    cartesian_case,
    fieldview_backend,
)


@gtx.field_operator
def addition(a: cases.IJKField, b: cases.IJKField) -> cases.IJKField:
    return a + b


@gtx.field_operator
def mixed_args(
    a: cases.IJKField, b: np.float64, c: cases.IJKField
) -> tuple[cases.IJKField, tuple[cases.IJKField, cases.IJKField]]:
    return (a, (c, a))


def test_allocate_default_unique(cartesian_case):  # noqa: F811 # fixtures
    a = cases.allocate(cartesian_case, mixed_args, "a")()

    assert np.min(a.asnumpy()) == 0
    assert np.max(a.asnumpy()) == np.prod(tuple(cartesian_case.default_sizes.values())) - 1

    b = cases.allocate(cartesian_case, mixed_args, "b")()

    assert b == np.max(a.asnumpy()) + 1

    c = cases.allocate(cartesian_case, mixed_args, "c")()

    assert np.min(c.asnumpy()) == b + 1
    assert np.max(c.asnumpy()) == np.prod(tuple(cartesian_case.default_sizes.values())) * 2


def test_allocate_return_default_zeros(cartesian_case):  # noqa: F811 # fixtures
    a, (b, c) = cases.allocate(cartesian_case, mixed_args, cases.RETURN)()

    assert np.all(a.asnumpy() == 0)
    assert np.all(b.asnumpy() == 0)
    assert np.all(c.asnumpy() == 0)


def test_allocate_const(cartesian_case):  # noqa: F811 # fixtures
    a = cases.allocate(cartesian_case, mixed_args, "a").strategy(cases.ConstInitializer(42))()
    assert np.all(a.asnumpy() == 42)

    b = cases.allocate(cartesian_case, mixed_args, "b").strategy(cases.ConstInitializer(42))()
    assert b == 42.0


@pytest.mark.parametrize("fieldview_backend", [~definitions.ProgramBackendId.ROUNDTRIP])
def test_verify_fails_with_wrong_reference(cartesian_case):  # noqa: F811 # fixtures
    a = cases.allocate(cartesian_case, addition, "a")()
    b = cases.allocate(cartesian_case, addition, "b")()
    out = cases.allocate(cartesian_case, addition, cases.RETURN)()
    wrong_ref = a

    with pytest.raises(AssertionError):
        cases.verify(cartesian_case, addition, a, b, out=out, ref=wrong_ref)


@pytest.mark.parametrize("fieldview_backend", [~definitions.ProgramBackendId.ROUNDTRIP])
def test_verify_fails_with_wrong_type(cartesian_case):  # noqa: F811 # fixtures
    a = cases.allocate(cartesian_case, addition, "a").dtype(np.float32)()
    b = cases.allocate(cartesian_case, addition, "b")()
    out = cases.allocate(cartesian_case, addition, cases.RETURN)()

    with pytest.raises(errors.DSLError):
        cases.verify(cartesian_case, addition, a, b, out=out, ref=a + b)


@pytest.mark.parametrize("fieldview_backend", [~definitions.ProgramBackendId.ROUNDTRIP])
def test_verify_with_default_data_fails_with_wrong_reference(
    cartesian_case,  # noqa: F811 # fixtures
):
    def wrong_ref(a, b):
        return a - b

    with pytest.raises(AssertionError):
        cases.verify_with_default_data(cartesian_case, addition, ref=wrong_ref)
