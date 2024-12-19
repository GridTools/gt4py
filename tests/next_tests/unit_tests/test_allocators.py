# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from typing import Optional

import pytest

import gt4py._core.definitions as core_defs
import gt4py.next._allocators as next_allocators
import gt4py.next.common as common
import gt4py.storage.allocators as core_allocators


class DummyAllocator(next_allocators.FieldBufferAllocatorProtocol):
    __gt_device_type__ = core_defs.DeviceType.CPU

    def __gt_allocate__(
        self,
        domain: common.Domain,
        dtype: core_defs.DType[core_defs.ScalarT],
        device_id: int = 0,
        aligned_index: Optional[Sequence[common.NamedIndex]] = None,
    ) -> core_defs.NDArrayObject:
        pass


class DummyAllocatorFactory(next_allocators.FieldBufferAllocatorFactoryProtocol):
    __gt_allocator__ = DummyAllocator()


def test_is_field_allocator():
    # Test with a field allocator
    allocator = DummyAllocator()
    assert next_allocators.is_field_allocator(allocator)

    # Test with an invalid object
    invalid_obj = "not an allocator"
    assert not next_allocators.is_field_allocator(invalid_obj)


def test_is_field_allocator_for():
    # Test with a valid field allocator for the specified device
    assert next_allocators.is_field_allocator_for(DummyAllocator(), core_defs.DeviceType.CPU)

    # Test with a valid field allocator for a different device
    assert not next_allocators.is_field_allocator_for(DummyAllocator(), core_defs.DeviceType.CUDA)

    # Test with an invalid field allocator
    assert not next_allocators.is_field_allocator_for("not an allocator", core_defs.DeviceType.CPU)


def test_is_field_allocator_factory():
    # Test with a field allocator factory
    allocator_factory = DummyAllocatorFactory()
    assert next_allocators.is_field_allocator_factory(allocator_factory)

    # Test with an invalid object
    invalid_obj = "not an allocator"
    assert not next_allocators.is_field_allocator_factory(invalid_obj)


def test_is_field_allocator_factory_for():
    # Test with a field allocator factory that matches the device type
    allocator_factory = DummyAllocatorFactory()
    assert next_allocators.is_field_allocator_factory_for(
        allocator_factory, core_defs.DeviceType.CPU
    )

    # Test with a field allocator factory that doesn't match the device type
    allocator_factory = DummyAllocatorFactory()
    assert not next_allocators.is_field_allocator_factory_for(
        allocator_factory, core_defs.DeviceType.CUDA
    )

    # Test with an object that is not a field allocator factory
    invalid_obj = "not an allocator factory"
    assert not next_allocators.is_field_allocator_factory_for(invalid_obj, core_defs.DeviceType.CPU)


def test_get_allocator():
    # Test with a field allocator
    allocator = DummyAllocator()
    assert next_allocators.get_allocator(allocator) == allocator

    # Test with a field allocator factory
    allocator_factory = DummyAllocatorFactory()
    assert next_allocators.get_allocator(allocator_factory) == allocator_factory.__gt_allocator__

    # Test with a default allocator
    default_allocator = DummyAllocator()
    assert next_allocators.get_allocator(None, default=default_allocator) == default_allocator

    # Test with an invalid object and no default allocator
    invalid_obj = "not an allocator"
    assert next_allocators.get_allocator(invalid_obj) is None

    with pytest.raises(
        TypeError,
        match=f"Object '{invalid_obj}' is neither a field allocator nor a field allocator factory",
    ):
        next_allocators.get_allocator(invalid_obj, strict=True)


def test_horizontal_first_layout_mapper():
    from gt4py.next._allocators import horizontal_first_layout_mapper

    # Test with only horizontal dimensions
    dims = [
        common.Dimension("D0", common.DimensionKind.HORIZONTAL),
        common.Dimension("D1", common.DimensionKind.HORIZONTAL),
        common.Dimension("D2", common.DimensionKind.HORIZONTAL),
    ]
    expected_layout_map = core_allocators.BufferLayoutMap((2, 1, 0))
    assert horizontal_first_layout_mapper(dims) == expected_layout_map

    # Test with no horizontal dimensions
    dims = [
        common.Dimension("D0", common.DimensionKind.VERTICAL),
        common.Dimension("D1", common.DimensionKind.LOCAL),
        common.Dimension("D2", common.DimensionKind.VERTICAL),
    ]
    expected_layout_map = core_allocators.BufferLayoutMap((2, 0, 1))
    assert horizontal_first_layout_mapper(dims) == expected_layout_map

    # Test with a mix of dimensions
    dims = [
        common.Dimension("D2", common.DimensionKind.LOCAL),
        common.Dimension("D0", common.DimensionKind.HORIZONTAL),
        common.Dimension("D1", common.DimensionKind.VERTICAL),
    ]
    expected_layout_map = core_allocators.BufferLayoutMap((0, 2, 1))
    assert horizontal_first_layout_mapper(dims) == expected_layout_map


class TestInvalidFieldBufferAllocator:
    def test_allocate(self):
        allocator = next_allocators.InvalidFieldBufferAllocator(
            core_defs.DeviceType.CPU, ValueError("test error")
        )
        I = common.Dimension("I")
        J = common.Dimension("J")
        domain = common.domain(((I, (2, 4)), (J, (3, 5))))
        dtype = float
        with pytest.raises(ValueError, match="test error"):
            allocator.__gt_allocate__(domain, dtype)


def test_allocate():
    from gt4py.next._allocators import StandardCPUFieldBufferAllocator, make_concrete_allocator

    I = common.Dimension("I")
    J = common.Dimension("J")
    domain = common.domain(((I, (0, 2)), (J, (0, 3))))
    dtype = core_defs.dtype(float)

    # Test with a explicit field allocator
    allocator = StandardCPUFieldBufferAllocator()
    tensor_buffer = make_concrete_allocator(domain, dtype, allocator=allocator)()
    assert tensor_buffer.shape == domain.shape
    assert tensor_buffer.dtype == dtype

    # Test with a device
    device = core_defs.Device(core_defs.DeviceType.CPU, 0)
    tensor_buffer = make_concrete_allocator(domain, dtype, device=device)()
    assert tensor_buffer.shape == domain.shape
    assert tensor_buffer.dtype == dtype

    # Test with both allocator and device
    with pytest.raises(ValueError, match="are incompatible"):
        make_concrete_allocator(
            domain,
            dtype,
            allocator=allocator,
            device=core_defs.Device(core_defs.DeviceType.CUDA, 0),
        )

    # Test with no device or allocator
    with pytest.raises(ValueError, match="No 'device' or 'allocator' specified"):
        make_concrete_allocator(domain, dtype)
