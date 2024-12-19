# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from gt4py import next as gtx
from gt4py._core import definitions as core_defs
from gt4py.next import allocators as next_allocators, common


I = gtx.Dimension("I")
J = gtx.Dimension("J")
K = gtx.Dimension("K")

sizes = {I: 10, J: 10, K: 10}


# TODO: parametrize with gpu backend and compare with cupy array
@pytest.mark.parametrize(
    "allocator, device",
    [
        [next_allocators.StandardCPUFieldBufferAllocator(), None],
        [None, core_defs.Device(core_defs.DeviceType.CPU, 0)],
    ],
)
def test_empty(allocator, device):
    ref = np.empty([sizes[I], sizes[J]]).astype(gtx.float32)
    a = gtx.empty(
        domain={I: range(sizes[I]), J: range(sizes[J])},
        dtype=core_defs.dtype(np.float32),
        allocator=allocator,
        device=device,
    )
    assert a.shape == ref.shape


# TODO: parametrize with gpu backend and compare with cupy array
@pytest.mark.parametrize(
    "allocator, device",
    [
        [next_allocators.StandardCPUFieldBufferAllocator(), None],
        [None, core_defs.Device(core_defs.DeviceType.CPU, 0)],
    ],
)
def test_zeros(allocator, device):
    a = gtx.zeros(
        common.Domain(
            dims=(I, J), ranges=(common.UnitRange(0, sizes[I]), common.UnitRange(0, sizes[J]))
        ),
        dtype=core_defs.dtype(np.float32),
        allocator=allocator,
        device=device,
    )
    ref = np.zeros((sizes[I], sizes[J])).astype(gtx.float32)

    assert np.array_equal(a.ndarray, ref)


# TODO: parametrize with gpu backend and compare with cupy array
@pytest.mark.parametrize(
    "allocator, device",
    [
        [next_allocators.StandardCPUFieldBufferAllocator(), None],
        [None, core_defs.Device(core_defs.DeviceType.CPU, 0)],
    ],
)
def test_ones(allocator, device):
    a = gtx.ones(
        common.Domain(dims=(I, J), ranges=(common.UnitRange(0, 10), common.UnitRange(0, 10))),
        dtype=core_defs.dtype(np.float32),
        allocator=allocator,
        device=device,
    )
    ref = np.ones((sizes[I], sizes[J])).astype(gtx.float32)

    assert np.array_equal(a.ndarray, ref)


# TODO: parametrize with gpu backend and compare with cupy array
@pytest.mark.parametrize(
    "allocator, device",
    [
        [next_allocators.StandardCPUFieldBufferAllocator(), None],
        [None, core_defs.Device(core_defs.DeviceType.CPU, 0)],
    ],
)
def test_full(allocator, device):
    a = gtx.full(
        domain={I: range(sizes[I] - 2), J: (sizes[J] - 2)},
        fill_value=42.0,
        dtype=core_defs.dtype(np.float32),
        allocator=allocator,
        device=device,
    )
    ref = np.full((sizes[I] - 2, sizes[J] - 2), 42.0).astype(gtx.float32)

    assert np.array_equal(a.ndarray, ref)


def test_as_field():
    ref = np.random.rand(sizes[I]).astype(gtx.float32)
    a = gtx.as_field([I], ref)
    assert np.array_equal(a.ndarray, ref)


def test_as_field_domain():
    ref = np.random.rand(sizes[I] - 1, sizes[J] - 1).astype(gtx.float32)
    domain = common.Domain(
        dims=(I, J), ranges=(common.UnitRange(0, sizes[I] - 1), common.UnitRange(0, sizes[J] - 1))
    )
    a = gtx.as_field(domain, ref)
    assert np.array_equal(a.ndarray, ref)


def test_as_field_origin():
    data = np.random.rand(sizes[I], sizes[J]).astype(gtx.float32)
    a = gtx.as_field([I, J], data, origin={I: 1, J: 2})
    domain_range = [(val.start, val.stop) for val in a.domain.ranges]
    assert np.allclose(domain_range, [(-1, 9), (-2, 8)])


# check that `as_field()` domain is correct depending on data origin and domain itself
def test_field_wrong_dims():
    with pytest.raises(ValueError, match=(r"Cannot construct 'Field' from array of shape")):
        gtx.as_field([I, J], np.random.rand(sizes[I]).astype(gtx.float32))


def test_field_wrong_domain():
    with pytest.raises(ValueError, match=(r"Cannot construct 'Field' from array of shape")):
        domain = common.Domain(
            dims=(I, J),
            ranges=(common.UnitRange(0, sizes[I] - 1), common.UnitRange(0, sizes[J] - 1)),
        )
        gtx.as_field(domain, np.random.rand(sizes[I], sizes[J]).astype(gtx.float32))


def test_field_wrong_origin():
    with pytest.raises(ValueError, match=(r"Origin keys {'J'} not in domain")):
        gtx.as_field([I], np.random.rand(sizes[I]).astype(gtx.float32), origin={"J": 0})

    with pytest.raises(ValueError, match=(r"Cannot specify origin for domain I")):
        gtx.as_field("I", np.random.rand(sizes[J]).astype(gtx.float32), origin={"J": 0})


@pytest.mark.xfail(reason="aligned_index not supported yet")
def test_aligned_index():
    gtx.as_field([I], np.random.rand(sizes[I]).astype(gtx.float32), aligned_index=[I, 0])


@pytest.mark.parametrize(
    "data, skip_value",
    [([0, 1, 2], None), ([0, 1, common._DEFAULT_SKIP_VALUE], common._DEFAULT_SKIP_VALUE)],
)
def test_as_connectivity(nd_array_implementation, data, skip_value):
    testee = gtx.as_connectivity([I], J, nd_array_implementation.array(data))
    assert testee.skip_value is skip_value
