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

from gt4py import next as gtx
from gt4py._core import definitions as core_defs
from gt4py.next import allocators as next_allocators, common, float32
from gt4py.next.program_processors.runners import roundtrip

from next_tests.integration_tests import cases


I = gtx.Dimension("I")
J = gtx.Dimension("J")

sizes = {"I": 10, "J": 10, "K": 10}


def test_as_field():
    a = np.random.rand(sizes["I"]).astype(gtx.float32)
    ref = gtx.as_field([I], a)
    assert np.allclose(ref.ndarray, a)


def test_as_field_domain():
    a = np.random.rand(sizes["I"] - 1, sizes["J"] - 1).astype(gtx.float32)
    domain = common.Domain(
        dims=(I, J),
        ranges=(common.UnitRange(0, sizes["I"] - 1), common.UnitRange(0, sizes["J"] - 1)),
    )
    ref = gtx.as_field(domain, a)
    assert np.allclose(ref.ndarray, a)


def test_as_field_origin():
    a = np.random.rand(sizes["I"], sizes["J"]).astype(gtx.float32)
    ref = gtx.as_field([I, J], a, origin={I: 1, J: 2})
    domain_range = [(val.start, val.stop) for val in ref.domain.ranges]
    assert np.allclose(domain_range, [(-1, 9), (-2, 8)])


# for as_field, check that the domain is correct depending on data origin and domain itself


def test_field_wrong_dims():
    with pytest.raises(
        ValueError,
        match=(r"Cannot construct `Field` from array of shape"),
    ):
        gtx.as_field([I, J], np.random.rand(sizes["I"]).astype(gtx.float32))


def test_field_wrong_domain():
    with pytest.raises(
        ValueError,
        match=(r"Cannot construct `Field` from array of shape"),
    ):
        domain = common.Domain(
            dims=(I, J),
            ranges=(common.UnitRange(0, sizes["I"] - 1), common.UnitRange(0, sizes["J"] - 1)),
        )
        gtx.as_field(domain, np.random.rand(sizes["I"], sizes["J"]).astype(gtx.float32))


def test_field_wrong_origin():
    with pytest.raises(
        ValueError,
        match=(r"Origin keys {'J'} not in domain"),
    ):
        gtx.as_field([I], np.random.rand(sizes["I"]).astype(gtx.float32), origin={"J": 0})

    with pytest.raises(
        ValueError,
        match=(r"Cannot specify origin for domain I"),
    ):
        gtx.as_field("I", np.random.rand(sizes["J"]).astype(gtx.float32), origin={"J": 0})


def test_aligned_index():
    with pytest.raises(
        AssertionError,
    ):
        gtx.as_field([I], np.random.rand(sizes["I"]).astype(gtx.float32), aligned_index=[I, 0])


@pytest.mark.parametrize(
    "allocator, device",
    [[roundtrip.backend, None], [None, core_defs.Device(core_defs.DeviceType.CPU, 0)]],
)
def test_empty(allocator, device):
    a = np.empty([sizes["I"], sizes["J"]]).astype(gtx.float32)
    ref = gtx.constructors.empty(
        domain={I: range(sizes["I"]), J: range(sizes["J"])},
        dtype=core_defs.dtype(np.float32),
        allocator=allocator,
        device=device,
    )
    assert ref.shape, a.shape
    assert np.allclose(ref.ndarray, a)


@pytest.mark.parametrize(
    "allocator, device",
    [[roundtrip.backend, None], [None, core_defs.Device(core_defs.DeviceType.CPU, 0)]],
)
def test_zeros(allocator, device):
    ref = gtx.constructors.zeros(
        common.Domain(
            dims=(I, J), ranges=(common.UnitRange(0, sizes["I"]), common.UnitRange(0, sizes["J"]))
        ),
        dtype=core_defs.dtype(np.float32),
        allocator=allocator,
        device=device,
    )
    a = np.zeros((sizes["I"], sizes["J"])).astype(gtx.float32)

    assert np.allclose(ref.ndarray, a)


# parametrize with gpu backend and compare with cupy array


@pytest.mark.parametrize(
    "allocator, device",
    [[roundtrip.backend, None], [None, core_defs.Device(core_defs.DeviceType.CPU, 0)]],
)
def test_ones(allocator, device):
    ref = gtx.constructors.ones(
        common.Domain(dims=(I, J), ranges=(common.UnitRange(0, 10), common.UnitRange(0, 10))),
        dtype=core_defs.dtype(np.float32),
        allocator=allocator,
        device=device,
    )
    a = np.ones((sizes["I"], sizes["J"])).astype(gtx.float32)

    assert np.allclose(ref.ndarray, a)


@pytest.mark.parametrize(
    "allocator, device",
    [[roundtrip.backend, None], [None, core_defs.Device(core_defs.DeviceType.CPU, 0)]],
)
def test_full(allocator, device):
    ref = gtx.constructors.full(
        domain={I: range(sizes["I"] - 2), J: (sizes["J"] - 2)},
        fill_value=42.0,
        dtype=core_defs.dtype(np.float32),
        allocator=allocator,
        device=device,
    )
    a = np.full((sizes["I"] - 2, sizes["J"] - 2), 42.0).astype(gtx.float32)

    assert np.allclose(ref.ndarray, a)


def test_as_field_with(cartesian_case):
    @gtx.field_operator
    def as_field_with_fo(a: gtx.Field[[I, J], gtx.float32]) -> gtx.Field[[I, J], gtx.float32]:
        return a

    @gtx.field_operator
    def field_fo(a: gtx.Field[[I, J], gtx.float32]) -> gtx.Field[[I, J], gtx.float32]:
        return a

    @gtx.program(backend=roundtrip.backend)
    def as_field_with_prog(
        a: gtx.Field[[I, J], gtx.float32],
        out: gtx.Field[[I, J], gtx.float32],
    ):
        field_fo(a, out=out)

    a = gtx.as_field([I, J], np.random.rand(sizes["I"], sizes["J"]).astype(gtx.float32))
    ref = gtx.constructors.as_field_with(
        common.Domain(dims=(I, J), ranges=(common.UnitRange(0, 10), common.UnitRange(0, 10))),
    )

    out = gtx.as_field([I, J], np.zeros((sizes["I"], sizes["J"])).astype(gtx.float32))
    cases.verify(
        cartesian_case,
        as_field_with_fo,
        a,
        out=out,
        ref=ref(a, allocator=as_field_with_prog),
    )


# @gtx.program(backend=roundtrip.backend)
# def prog(
#     a: gtx.Field[[I, J], gtx.float32],
#     b: gtx.Field[[I, J], gtx.float32],
#     out: gtx.Field[[I, J], gtx.float32],
# ):
#     add(a, b, out=out)
#
#
# a = gtx.constructors.ones(
#     common.Domain(dims=(I, J), ranges=(common.UnitRange(0, 10), common.UnitRange(0, 10))),
#     dtype=core_defs.dtype(np.float32),
#     allocator=prog,
# )
#
#
# arr = np.full((10, 10), 42.0)
# b = gtx.constructors.as_field(
#     common.Domain(dims=(I, J), ranges=(common.UnitRange(0, 10), common.UnitRange(0, 10))),
#     arr,
#     dtype=core_defs.dtype(np.float32),
#     allocator=prog,
# )
#
# out = gtx.constructors.empty(
#     common.Domain(dims=(I, J), ranges=(common.UnitRange(0, 10), common.UnitRange(0, 10))),
#     dtype=core_defs.dtype(np.float32),
#     allocator=prog,
# )
#
# prog(a, b, out, offset_provider={})
#
# print(out.ndarray)
