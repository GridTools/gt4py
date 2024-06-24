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

import pytest

from gt4py._core import definitions as core_defs
from gt4py.storage.allocators import (
    NumPyMemoryResourceHandler,
    CuPyMemoryResourceHandler,
)


HANDLER_IMPLS = list(
    filter(lambda x: x is not None, (NumPyMemoryResourceHandler, CuPyMemoryResourceHandler))
)


class MemoryResourceHandlerTest:
    @pytest.fixture(scope="class", params=HANDLER_IMPLS)
    def handler_impl(self, request):
        return request.param

    def test_address_of(self, handler_impl):
        xp = handler_impl.array_ns
        buffer = xp.zeros((10,), dtype=xp.float32)
        address = handler_impl.address_of(buffer)
        assert isinstance(address, int)
        assert address >= 0
        assert handler_impl.address_of(xp.asarray(buffer)) == address

    def test_malloc(self, handler_impl):
        xp = handler_impl.array_ns
        byte_size = 100
        device_id = 0
        buffer = handler_impl.malloc(byte_size, device_id)
        assert isinstance(buffer, xp.ndarray)
        assert buffer.shape == (byte_size,)
        assert buffer.dtype == xp.uint8

    def test_tensorize(self, handler_impl):
        xp = handler_impl.array_ns
        buffer = xp.zeros((10,), dtype=xp.float32)
        dtype = xp.float32
        shape = (2, 5)
        allocated_shape = (2, 5)
        strides = (20, 4)
        byte_offset = 0
        tensor = handler_impl.tensorize(buffer, dtype, shape, allocated_shape, strides, byte_offset)
        assert isinstance(tensor, xp.ndarray)
        assert tensor.shape == shape
        assert tensor.dtype == dtype


def test_cpu_device_is_zero():
    with pytest.raises(ValueError, match="device Id"):
        NumPyMemoryResourceHandler.malloc(32, 1)
