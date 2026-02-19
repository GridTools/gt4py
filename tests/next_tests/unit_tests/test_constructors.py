# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from typing import Any
import pytest
from types import ModuleType
import dataclasses

from gt4py import next as gtx
from gt4py._core import definitions as core_defs
from gt4py.next import allocators as next_allocators, common


I = gtx.Dimension("I")
J = gtx.Dimension("J")
K = gtx.Dimension("K")

sizes = {I: 10, J: 10, K: 10}

cp = core_defs.cp


@dataclasses.dataclass
class ConstructureTestSetup:
    allocator: Any  #: gtx.FieldAllocator
    device: core_defs.Device
    expected_xp: ModuleType


def _constructor_test_cases():
    return [
        ConstructureTestSetup(
            allocator=next_allocators.StandardCPUFieldBufferAllocator(),
            device=None,
            expected_xp=np,
        ),
        ConstructureTestSetup(
            allocator=None,
            device=core_defs.Device(core_defs.DeviceType.CPU, 0),
            expected_xp=np,
        ),
        ConstructureTestSetup(
            allocator=np,
            device=None,
            expected_xp=np,
        ),
        pytest.param(
            ConstructureTestSetup(
                allocator=next_allocators.StandardGPUFieldBufferAllocator(),
                device=None,
                expected_xp=cp,
            ),
            marks=pytest.mark.requires_gpu,
        ),
        pytest.param(
            ConstructureTestSetup(
                allocator=None,
                device=core_defs.Device(core_defs.CUPY_DEVICE_TYPE, 0),
                expected_xp=cp,
            ),
            marks=pytest.mark.requires_gpu,
        ),
    ]


@pytest.fixture(
    params=_constructor_test_cases(),
    ids=lambda x: f"{type(x.allocator).__name__ if x.allocator is not None else None}-device={x.device.device_type if x.device is not None and x.device.device_type is not None else None}-{x.expected_xp.__name__ if x.expected_xp is not None else None}",
)
def constructor_test_cases(request):
    yield request.param


def test_empty(constructor_test_cases):
    allocator = constructor_test_cases.allocator
    device = constructor_test_cases.device
    expected_xp = constructor_test_cases.expected_xp

    ref = expected_xp.empty([sizes[I], sizes[J]]).astype(gtx.float32)
    a = gtx.empty(
        domain={I: range(sizes[I]), J: range(sizes[J])},
        dtype=core_defs.dtype(np.float32),
        allocator=allocator,
        device=device,
    )
    assert isinstance(a.ndarray, expected_xp.ndarray)
    assert a.shape == ref.shape


def test_zeros(constructor_test_cases):
    allocator = constructor_test_cases.allocator
    device = constructor_test_cases.device
    expected_xp = constructor_test_cases.expected_xp

    a = gtx.zeros(
        common.Domain(
            dims=(I, J), ranges=(common.UnitRange(0, sizes[I]), common.UnitRange(0, sizes[J]))
        ),
        dtype=core_defs.dtype(np.float32),
        allocator=allocator,
        device=device,
    )
    ref = expected_xp.zeros((sizes[I], sizes[J])).astype(gtx.float32)

    assert isinstance(a.ndarray, expected_xp.ndarray)
    assert expected_xp.array_equal(a.ndarray, ref)


def test_ones(constructor_test_cases):
    allocator = constructor_test_cases.allocator
    device = constructor_test_cases.device
    expected_xp = constructor_test_cases.expected_xp

    a = gtx.ones(
        common.Domain(dims=(I, J), ranges=(common.UnitRange(0, 10), common.UnitRange(0, 10))),
        dtype=core_defs.dtype(np.float32),
        allocator=allocator,
        device=device,
    )
    ref = expected_xp.ones((sizes[I], sizes[J])).astype(gtx.float32)

    assert isinstance(a.ndarray, expected_xp.ndarray)
    assert expected_xp.array_equal(a.ndarray, ref)


def test_full(constructor_test_cases):
    allocator = constructor_test_cases.allocator
    device = constructor_test_cases.device
    expected_xp = constructor_test_cases.expected_xp

    a = gtx.full(
        domain={I: range(sizes[I] - 2), J: (sizes[J] - 2)},
        fill_value=42.0,
        dtype=core_defs.dtype(np.float32),
        allocator=allocator,
        device=device,
    )
    ref = expected_xp.full((sizes[I] - 2, sizes[J] - 2), 42.0).astype(gtx.float32)

    assert isinstance(a.ndarray, expected_xp.ndarray)
    assert expected_xp.array_equal(a.ndarray, ref)


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

    with pytest.raises(ValueError, match=(r"Cannot specify origin for .* domain I")):
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
