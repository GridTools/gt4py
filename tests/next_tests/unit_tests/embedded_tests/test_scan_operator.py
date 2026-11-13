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
from gt4py.next import common
from gt4py.next.embedded import exceptions as embedded_exceptions


I = gtx.Dimension("I")
K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)


@gtx.scan_operator(axis=K, forward=True, init=(0.0, 0.0))
def _forward(carry: tuple[float, float], a: float, b: float, s: float) -> tuple[float, float]:
    return carry[0] + a * b + s, carry[1] - a


@gtx.scan_operator(axis=K, forward=False, init=0.0)
def _backward(carry: float, a: float) -> float:
    return 2.0 * carry + a


def _field(array, domain):
    return common._field(np.asarray(array, dtype=np.float64), domain=common.domain(domain))


def test_scan_reads_broadcast_and_shifted_fields():
    a = np.arange(12.0).reshape(3, 4)
    b = np.array([10.0, 20.0, 30.0, 40.0])
    a_field = _field(a, {I: (0, 3), K: (0, 4)})
    b_field = _field(b, {I: (-1, 3)})
    out = tuple(_field(np.zeros((2, 4)), {I: (1, 3), K: (0, 4)}) for _ in range(2))

    _forward(a_field, b_field, 0.5, out=out, offset_provider={})

    prod = a[1:3, :] * b[2:4, None] + 0.5
    np.testing.assert_array_equal(out[0].asnumpy(), np.cumsum(prod, axis=1))
    np.testing.assert_array_equal(out[1].asnumpy(), -np.cumsum(a[1:3, :], axis=1))


def test_scan_reads_size_one_axis():
    a = np.array([[1.0], [2.0]])
    a_field = common._field(a, domain=common.domain({I: (0, 2), K: (0, 3)}))
    out = _field(np.zeros((2, 3)), {I: (0, 2), K: (0, 3)})

    _backward(a_field, out=out, offset_provider={})

    np.testing.assert_array_equal(out.asnumpy(), [[7.0, 3.0, 1.0], [14.0, 6.0, 2.0]])


def test_scan_out_of_bounds_column():
    a_field = _field(np.ones((2, 3)), {I: (0, 2), K: (0, 3)})
    out = _field(np.zeros((2, 4)), {I: (0, 2), K: (0, 4)})

    with pytest.raises(embedded_exceptions.IndexOutOfBounds):
        _backward(a_field, out=out, offset_provider={})
