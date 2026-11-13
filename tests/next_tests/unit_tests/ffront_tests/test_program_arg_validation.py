# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import warnings

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import astype, common, errors, float64
from gt4py.next.ffront import decorator
from gt4py.next.type_system import type_translation


I = gtx.Dimension("I")
J = gtx.Dimension("J")
K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)


def _field(dtype, domain):
    domain = common.domain(domain)
    return common._field(np.zeros(domain.shape, dtype=dtype), domain=domain)


_INT32_MAX, _INT64_MAX = int(np.iinfo(np.int32).max), int(np.iinfo(np.int64).max)

_VALUES = [
    True,
    np.bool_(False),
    1.5,
    np.float32(1.5),
    np.float64(1.5),
    np.int32(1),
    np.int64(1),
    0,
    _INT32_MAX,
    _INT32_MAX + 1,
    -_INT32_MAX - 1,
    -_INT32_MAX - 2,
    _INT64_MAX,
    _field(np.float64, {I: 3}),
    _field(np.float64, {I: (2, 7)}),
    _field(np.float32, {I: 3}),
    _field(np.int32, {I: 3}),
    _field(np.bool_, {I: 3}),
    _field(np.float64, {I: 3, K: 2}),
    _field(np.float64, {J: 3}),
    _field(np.float64, {}),
    (1.5, _field(np.float64, {I: 3})),
    (1.5, (_field(np.float64, {I: 3}), 2)),
    (np.float32(1.5), (_field(np.float64, {I: 3}), 2)),
    (1.5, (_field(np.float64, {I: 3}), _INT32_MAX + 1)),
    (),
]


@pytest.mark.parametrize(
    "a, b", itertools.combinations_with_replacement(_VALUES, 2), ids=lambda v: type(v).__name__
)
def test_arg_type_key_determines_type(a, b):
    key_a, key_b = decorator._arg_type_key(a), decorator._arg_type_key(b)
    assert key_a is not None and key_b is not None
    if key_a == key_b:
        assert type_translation.from_value(a) == type_translation.from_value(b)


@pytest.mark.parametrize(
    "value",
    [
        _INT64_MAX + 1,
        common._connectivity(
            np.zeros((3, 2), dtype=np.int32),
            codomain=I,
            domain={I: 3, gtx.Dimension("I2I", kind=gtx.DimensionKind.LOCAL): 2},
        ),
        (1.5, _INT64_MAX + 1),
        [1.5],
    ],
    ids=lambda v: type(v).__name__,
)
def test_arg_type_key_not_cached(value):
    assert decorator._arg_type_key(value) is None


@gtx.field_operator
def _copy(a: gtx.Field[[I], gtx.float64], s: gtx.int32) -> gtx.Field[[I], gtx.float64]:
    return a + astype(s, float64)


@gtx.program(backend=None)
def _copy_program(a: gtx.Field[[I], gtx.float64], s: gtx.int32, out: gtx.Field[[I], gtx.float64]):
    _copy(a, s, out=out)


@pytest.mark.parametrize(
    "a, s",
    [
        (_field(np.int32, {I: 3}), 1),
        (_field(np.float64, {J: 3}), 1),
        (_field(np.float64, {I: 3}), _INT32_MAX + 1),
        (_field(np.float64, {I: 3}), 1.0),
        (_field(np.float64, {I: 3}), np.int64(1)),
    ],
)
def test_validation_after_valid_call_still_rejects(a, s):
    program = _copy_program.with_backend(None)
    out = _field(np.float64, {I: 3})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(2):
            program(_field(np.float64, {I: 3}), 1, out, offset_provider={})
            with pytest.raises(errors.DSLError, match="Invalid argument types"):
                program(a, s, out, offset_provider={})
