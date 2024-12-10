# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from types import ModuleType
from typing import Any

import numpy as np


try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None

import pytest

from gt4py import next as gtx
from gt4py._core import definitions as core_defs
from gt4py.next import allocators as next_allocators, common


I = gtx.Dimension("I")
J = gtx.Dimension("J")
K = gtx.Dimension("K")

sizes = {I: 10, J: 10, K: 10}


def _pretty_print(val):
    if val is None:
        return "None"
    if isinstance(val, ModuleType):
        return val.__name__
    return val.__class__.__name__


def _pretty_print_allocator_device_namespace(val: tuple[Any, Any, Any]):
    return f"allocator={_pretty_print(val[0])}-device={_pretty_print(val[1])}-ref_namespace={_pretty_print(val[2])}"


def allocator_device_refnamespace_params():
    for v in [
        [next_allocators.StandardCPUFieldBufferAllocator(), None, np],
        [None, core_defs.Device(core_defs.DeviceType.CPU, 0), np],
        [np, None, np],
    ]:
        yield pytest.param(
            v,
            id=_pretty_print_allocator_device_namespace(v),
        )
    for v in [
        [next_allocators.StandardGPUFieldBufferAllocator(), None, cp],
        [None, core_defs.Device(core_defs.DeviceType.CUDA, 0), cp],  # TODO CUDA or HIP...
    ]:
        yield pytest.param(
            v, id=_pretty_print_allocator_device_namespace(v), marks=pytest.mark.requires_gpu
        )
    for v in [[jnp, None, jnp]]:
        yield pytest.param(
            v, id=_pretty_print_allocator_device_namespace(v), marks=pytest.mark.requires_jax
        )


@pytest.fixture(params=allocator_device_refnamespace_params())
def allocator_device_refnamespace(request):
    return request.param


def test_empty(allocator_device_refnamespace):
    allocator, device, xp = allocator_device_refnamespace
    ref = xp.empty([sizes[I], sizes[J]]).astype(gtx.float32)
    a = gtx.empty(
        domain={I: range(sizes[I]), J: range(sizes[J])},
        dtype=core_defs.dtype(np.float32),
        allocator=allocator,
        device=device,
    )
    assert a.shape == ref.shape


def test_zeros(allocator_device_refnamespace):
    allocator, device, xp = allocator_device_refnamespace
    a = gtx.zeros(
        common.Domain(
            dims=(I, J), ranges=(common.UnitRange(0, sizes[I]), common.UnitRange(0, sizes[J]))
        ),
        dtype=core_defs.dtype(np.float32),
        allocator=allocator,
        device=device,
    )
    ref = xp.zeros((sizes[I], sizes[J])).astype(gtx.float32)

    assert xp.array_equal(a.ndarray, ref)


def test_ones(allocator_device_refnamespace):
    allocator, device, xp = allocator_device_refnamespace
    a = gtx.ones(
        common.Domain(dims=(I, J), ranges=(common.UnitRange(0, 10), common.UnitRange(0, 10))),
        dtype=core_defs.dtype(np.float32),
        allocator=allocator,
        device=device,
    )
    ref = xp.ones((sizes[I], sizes[J])).astype(gtx.float32)

    assert xp.array_equal(a.ndarray, ref)


def test_full(allocator_device_refnamespace):
    allocator, device, xp = allocator_device_refnamespace
    a = gtx.full(
        domain={I: range(sizes[I] - 2), J: (sizes[J] - 2)},
        fill_value=42.0,
        dtype=core_defs.dtype(np.float32),
        allocator=allocator,
        device=device,
    )
    ref = xp.full((sizes[I] - 2, sizes[J] - 2), 42.0).astype(gtx.float32)

    assert xp.array_equal(a.ndarray, ref)


def test_deepcopy():
    testee = gtx.as_field([I, J], np.random.rand(sizes[I], sizes[J]))
    result = copy.deepcopy(testee)
    assert testee.ndarray.strides == result.ndarray.strides
    assert (
        result.ndarray.strides != result.ndarray.copy().strides
    )  # sanity check for this test, make sure our allocator don't have C-contiguous strides
    assert np.array_equal(testee.ndarray, result.ndarray)


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
