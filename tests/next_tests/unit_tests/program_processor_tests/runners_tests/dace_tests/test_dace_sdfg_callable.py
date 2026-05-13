# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test the SDFG callable interface in dace backend."""

import pytest

dace = pytest.importorskip("dace")
from gt4py.next.program_processors.runners.dace import sdfg_callable


def test_dace_shape_mismatch():
    sdfg = dace.SDFG("sdfg_array_shape_mismatch")
    _, array_desc = sdfg.add_array("test_array", (10, 10), dace.float64, strides=(10, 1))

    class MockNDArray:
        shape = (10, 20)
        strides = (80, 8)  # Incorrect stride for the second dimension
        itemsize = 8

    ndarray = MockNDArray()

    with pytest.raises(RuntimeError, match="Array shape mismatch: expected 10, got 20."):
        sdfg_callable.get_array_shape_symbols(array_desc, ndarray)


def test_dace_stride_mismatch():
    sdfg = dace.SDFG("sdfg_array_stride_mismatch")
    _, array_desc = sdfg.add_array("test_array", (10, 10), dace.float64, strides=(10, 1))

    class MockNDArray:
        shape = (10, 10)
        strides = (80, 16)  # Incorrect stride for the second dimension
        itemsize = 8

    ndarray = MockNDArray()

    with pytest.raises(RuntimeError, match="Array stride mismatch: expected 1, got 2."):
        sdfg_callable.get_array_stride_symbols(array_desc, ndarray)
