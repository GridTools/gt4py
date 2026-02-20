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
import gt4py.next.custom_layout_allocators as next_allocators
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
    ) -> core_allocators.TensorBuffer[core_defs.DeviceTypeT, core_defs.ScalarT]:
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


@pytest.mark.parametrize(
    "test_function",
    [
        next_allocators.is_field_allocator_for,
        next_allocators.is_field_allocation_tool_for,  # since an allocator is an allocation_tool
    ],
)
def test_is_field_allocator_for(test_function):
    # Test with a valid field allocator for the specified device
    assert test_function(DummyAllocator(), core_defs.DeviceType.CPU)

    # Test with a valid field allocator for a different device
    assert not test_function(DummyAllocator(), core_defs.DeviceType.CUDA)

    # Test with an invalid field allocator
    assert not test_function("not an allocator", core_defs.DeviceType.CPU)


def test_is_field_allocator_factory():
    # Test with a field allocator factory
    allocator_factory = DummyAllocatorFactory()
    assert next_allocators.is_field_allocator_factory(allocator_factory)

    # Test with an invalid object
    invalid_obj = "not an allocator"
    assert not next_allocators.is_field_allocator_factory(invalid_obj)


@pytest.mark.parametrize(
    "test_function",
    [
        next_allocators.is_field_allocator_factory_for,
        next_allocators.is_field_allocation_tool_for,  # since a factory is an allocation_tool
    ],
)
def test_is_field_allocator_factory_for(test_function):
    # Test with a field allocator factory that matches the device type
    allocator_factory = DummyAllocatorFactory()
    assert test_function(allocator_factory, core_defs.DeviceType.CPU)

    # Test with a field allocator factory that doesn't match the device type
    allocator_factory = DummyAllocatorFactory()
    assert not test_function(allocator_factory, core_defs.DeviceType.CUDA)

    # Test with an object that is not a field allocator factory
    invalid_obj = "not an allocator factory"
    assert not test_function(invalid_obj, core_defs.DeviceType.CPU)


def test_horizontal_first_layout_mapper():
    from gt4py.next.custom_layout_allocators import horizontal_first_layout_mapper

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


Cell = common.Dimension("Cell", common.DimensionKind.HORIZONTAL)
Edge = common.Dimension("Edge", common.DimensionKind.HORIZONTAL)
K = common.Dimension("K", common.DimensionKind.VERTICAL)


class TestBaseFieldBufferAllocatorAlignedIndex:
    """Tests for aligned_index handling in BaseFieldBufferAllocator.__gt_allocate__,
    particularly with non-zero-origin domains."""

    @staticmethod
    def _make_allocator() -> next_allocators.StandardCPUFieldBufferAllocator:
        return next_allocators.StandardCPUFieldBufferAllocator()

    def test_no_aligned_index(self):
        allocator = self._make_allocator()
        domain = common.Domain(
            dims=(Cell, K), ranges=(common.UnitRange(0, 10), common.UnitRange(0, 5))
        )
        result = allocator.__gt_allocate__(domain, core_defs.dtype(float))
        assert result.shape == (10, 5)

    def test_aligned_index_zero_origin(self):
        """With a zero-origin domain, aligned_index values are passed through as-is."""
        allocator = self._make_allocator()
        domain = common.Domain(
            dims=(Cell, K), ranges=(common.UnitRange(0, 10), common.UnitRange(0, 5))
        )
        aligned = [common.NamedIndex(Cell, 3)]
        result = allocator.__gt_allocate__(domain, core_defs.dtype(float), aligned_index=aligned)
        assert result.shape == (10, 5)
        assert result.aligned_index == (3, 0)

    def test_aligned_index_nonzero_origin_cell(self):
        """Domain starting at Cell=100: aligned_index Cell=103 should become relative index 3."""
        allocator = self._make_allocator()
        domain = common.Domain(
            dims=(Cell, K), ranges=(common.UnitRange(100, 110), common.UnitRange(0, 5))
        )
        aligned = [common.NamedIndex(Cell, 103)]
        result = allocator.__gt_allocate__(domain, core_defs.dtype(float), aligned_index=aligned)
        assert result.shape == (10, 5)
        assert result.aligned_index == (3, 0)

    def test_aligned_index_nonzero_origin_edge(self):
        """Domain starting at Edge=200: aligned_index Edge=207 should become relative index 7."""
        allocator = self._make_allocator()
        domain = common.Domain(
            dims=(Edge, K), ranges=(common.UnitRange(200, 212), common.UnitRange(0, 5))
        )
        aligned = [common.NamedIndex(Edge, 207)]
        result = allocator.__gt_allocate__(domain, core_defs.dtype(float), aligned_index=aligned)
        assert result.shape == (12, 5)
        assert result.aligned_index == (7, 0)

    def test_aligned_index_nonzero_origin_all_dims(self):
        """Both dims have non-zero origins."""
        allocator = self._make_allocator()
        domain = common.Domain(
            dims=(Cell, K), ranges=(common.UnitRange(50, 60), common.UnitRange(10, 15))
        )
        aligned = [common.NamedIndex(Cell, 53), common.NamedIndex(K, 12)]
        result = allocator.__gt_allocate__(domain, core_defs.dtype(float), aligned_index=aligned)
        assert result.shape == (10, 5)
        assert result.aligned_index == (3, 2)

    def test_aligned_index_at_domain_start(self):
        """Aligned index equals domain start — relative index should be 0."""
        allocator = self._make_allocator()
        domain = common.Domain(
            dims=(Cell, K), ranges=(common.UnitRange(100, 110), common.UnitRange(20, 25))
        )
        aligned = [common.NamedIndex(Cell, 100), common.NamedIndex(K, 20)]
        result = allocator.__gt_allocate__(domain, core_defs.dtype(float), aligned_index=aligned)
        assert result.shape == (10, 5)
        assert result.aligned_index == (0, 0)

    def test_aligned_index_shared_between_cell_and_edge_fields(self):
        """Same aligned_index with both Cell and Edge can be used to allocate
        both a Cell-field and an Edge-field; the irrelevant dimension is ignored."""
        allocator = self._make_allocator()
        aligned = [common.NamedIndex(Cell, 103), common.NamedIndex(Edge, 207)]

        cell_domain = common.Domain(
            dims=(Cell, K), ranges=(common.UnitRange(100, 110), common.UnitRange(0, 5))
        )
        cell_result = allocator.__gt_allocate__(
            cell_domain, core_defs.dtype(float), aligned_index=aligned
        )
        assert cell_result.shape == (10, 5)
        assert cell_result.aligned_index == (3, 0)

        edge_domain = common.Domain(
            dims=(Edge, K), ranges=(common.UnitRange(200, 212), common.UnitRange(0, 5))
        )
        edge_result = allocator.__gt_allocate__(
            edge_domain, core_defs.dtype(float), aligned_index=aligned
        )
        assert edge_result.shape == (12, 5)
        assert edge_result.aligned_index == (7, 0)

    def test_aligned_index_outside_domain_raises(self):
        """Aligned index outside the domain range should raise ValueError."""
        allocator = self._make_allocator()
        domain = common.Domain(
            dims=(Cell, K), ranges=(common.UnitRange(100, 110), common.UnitRange(0, 5))
        )

        # Before domain start
        with pytest.raises(ValueError, match="outside the domain range"):
            allocator.__gt_allocate__(
                domain,
                core_defs.dtype(float),
                aligned_index=[common.NamedIndex(Cell, 50)],
            )

        # After domain end
        with pytest.raises(ValueError, match="outside the domain range"):
            allocator.__gt_allocate__(
                domain,
                core_defs.dtype(float),
                aligned_index=[common.NamedIndex(Cell, 120)],
            )

        # Exactly at domain end (exclusive upper bound)
        with pytest.raises(ValueError, match="outside the domain range"):
            allocator.__gt_allocate__(
                domain,
                core_defs.dtype(float),
                aligned_index=[common.NamedIndex(Cell, 110)],
            )
