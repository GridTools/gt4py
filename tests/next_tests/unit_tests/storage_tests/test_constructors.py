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
from gt4py.next.ffront.fbuiltins import astype, broadcast
from gt4py.next.program_processors.runners import roundtrip

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import cartesian_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    fieldview_backend,
    reduction_setup,
)


I = gtx.Dimension("I")
J = gtx.Dimension("J")

default_sizes = {"I": 10, "J": 10, "K": 10}


def test_add(cartesian_case):
    @gtx.field_operator
    def add(
        a: gtx.Field[[I, J], gtx.float32], b: gtx.Field[[I, J], gtx.float32]
    ) -> gtx.Field[[I, J], gtx.float32]:
        return a + b

    a = gtx.as_field(
        [I, J], np.random.rand(default_sizes["I"], default_sizes["J"]).astype(gtx.float32)
    )
    b = gtx.as_field(
        [I, J], np.random.rand(default_sizes["I"], default_sizes["J"]).astype(gtx.float32)
    )
    out = gtx.as_field(
        [I, J], np.zeros((default_sizes["I"], default_sizes["J"])).astype(gtx.float32)
    )
    cases.verify(
        cartesian_case,
        add,
        a,
        b,
        out=out,
        ref=(a.ndarray + b.ndarray),
    )


def test_field_wrong_dims(cartesian_case):
    with pytest.raises(
        ValueError,
        match=(r"Cannot construct `Field` from array of shape"),
    ):
        gtx.as_field([I, J], np.random.rand(default_sizes["I"]).astype(gtx.float32))


def test_field_wrong_origin(cartesian_case):
    with pytest.raises(
        ValueError,
        match=(r"Origin keys {'J'} not in domain"),
    ):
        gtx.as_field([I], np.random.rand(default_sizes["I"]).astype(gtx.float32), origin={"J": 0})

    with pytest.raises(
        ValueError,
        match=(r"Cannot specify origin for domain I"),
    ):
        gtx.as_field("I", np.random.rand(default_sizes["J"]).astype(gtx.float32), origin={"J": 0})


def test_empty(cartesian_case):
    @gtx.field_operator
    def empty_fo(a: gtx.Field[[I, J], gtx.float32]) -> gtx.Field[[I, J], gtx.float32]:
        return a

    @gtx.field_operator
    def field_fo(a: gtx.Field[[I, J], gtx.float32]) -> gtx.Field[[I, J], gtx.float32]:
        return a

    @gtx.program(backend=roundtrip.backend)
    def empty_prog(
        a: gtx.Field[[I, J], gtx.float32],
        out: gtx.Field[[I, J], gtx.float32],
    ):
        field_fo(a, out=out)

    a = gtx.as_field([I, J], np.empty([default_sizes["I"], default_sizes["J"]]).astype(gtx.float32))
    ref = gtx.constructors.empty(
        common.Domain(dims=(I, J), ranges=(common.UnitRange(0, 10), common.UnitRange(0, 10))),
        dtype=core_defs.dtype(np.float32),
        allocator=empty_prog,
    )

    cases.verify(
        cartesian_case,
        empty_fo,
        a,
        out=a,
        ref=ref,
    )


def test_zeros(cartesian_case):
    @gtx.field_operator
    def zeros_fo() -> gtx.Field[[I, J], gtx.float32]:
        return astype(broadcast(0.0, (I, J)), float32)

    @gtx.field_operator
    def field_fo(a: gtx.Field[[I, J], gtx.float32]) -> gtx.Field[[I, J], gtx.float32]:
        return a

    @gtx.program(backend=roundtrip.backend)
    def zeros_prog(
        a: gtx.Field[[I, J], gtx.float32],
        out: gtx.Field[[I, J], gtx.float32],
    ):
        field_fo(a, out=out)

    ref = gtx.constructors.zeros(
        common.Domain(dims=(I, J), ranges=(common.UnitRange(0, 10), common.UnitRange(0, 10))),
        dtype=core_defs.dtype(np.float32),
        allocator=zeros_prog,
    )
    out = gtx.as_field(
        [I, J], np.zeros((default_sizes["I"], default_sizes["J"])).astype(gtx.float32)
    )
    cases.verify(
        cartesian_case,
        zeros_fo,
        out=out,
        ref=ref,
    )


def test_ones(cartesian_case):
    @gtx.field_operator
    def ones_fo() -> gtx.Field[[I, J], gtx.float32]:
        return astype(broadcast(1.0, (I, J)), float32)

    @gtx.field_operator
    def field_fo(a: gtx.Field[[I, J], gtx.float32]) -> gtx.Field[[I, J], gtx.float32]:
        return a

    @gtx.program(backend=roundtrip.backend)
    def ones_prog(
        a: gtx.Field[[I, J], gtx.float32],
        out: gtx.Field[[I, J], gtx.float32],
    ):
        field_fo(a, out=out)

    ref = gtx.constructors.ones(
        common.Domain(dims=(I, J), ranges=(common.UnitRange(0, 10), common.UnitRange(0, 10))),
        dtype=core_defs.dtype(np.float32),
        allocator=ones_prog,
    )
    out = gtx.as_field(
        [I, J], np.zeros((default_sizes["I"], default_sizes["J"])).astype(gtx.float32)
    )
    cases.verify(
        cartesian_case,
        ones_fo,
        out=out,
        ref=ref,
    )


def test_full(cartesian_case):
    @gtx.field_operator
    def full_fo() -> gtx.Field[[I, J], gtx.float32]:
        return astype(broadcast(42.0, (I, J)), float32)

    @gtx.field_operator
    def field_fo(a: gtx.Field[[I, J], gtx.float32]) -> gtx.Field[[I, J], gtx.float32]:
        return a

    @gtx.program(backend=roundtrip.backend)
    def full_prog(
        a: gtx.Field[[I, J], gtx.float32],
        out: gtx.Field[[I, J], gtx.float32],
    ):
        field_fo(a, out=out)

    ref = gtx.constructors.full(
        common.Domain(dims=(I, J), ranges=(common.UnitRange(0, 10), common.UnitRange(0, 10))),
        fill_value=42.0,
        dtype=core_defs.dtype(np.float32),
        allocator=full_prog,
    )
    out = gtx.as_field(
        [I, J], np.zeros((default_sizes["I"], default_sizes["J"])).astype(gtx.float32)
    )
    cases.verify(
        cartesian_case,
        full_fo,
        out=out,
        ref=ref,
    )


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

    a = gtx.as_field(
        [I, J], np.random.rand(default_sizes["I"], default_sizes["J"]).astype(gtx.float32)
    )
    ref = gtx.constructors.as_field_with(
        common.Domain(dims=(I, J), ranges=(common.UnitRange(0, 10), common.UnitRange(0, 10))),
    )

    out = gtx.as_field(
        [I, J], np.zeros((default_sizes["I"], default_sizes["J"])).astype(gtx.float32)
    )
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
