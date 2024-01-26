# -*- coding: utf-8 -*-
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

import gt4py.next as gtx
from gt4py.next import int32

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import cartesian_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
    reduction_setup,
)


def test_with_bound_args(cartesian_case):
    @gtx.field_operator
    def fieldop_bound_args(a: cases.IField, scalar: int32, condition: bool) -> cases.IField:
        if not condition:
            scalar = 0
        return a + scalar

    @gtx.program
    def program_bound_args(a: cases.IField, scalar: int32, condition: bool, out: cases.IField):
        fieldop_bound_args(a, scalar, condition, out=out)

    a = cases.allocate(cartesian_case, program_bound_args, "a")()
    scalar = int32(1)
    ref = a + scalar
    out = cases.allocate(cartesian_case, program_bound_args, "out")()

    prog_bounds = program_bound_args.with_bound_args(scalar=scalar, condition=True)
    cases.verify(cartesian_case, prog_bounds, a, out, inout=out, ref=ref)


def test_with_bound_args_order_args(cartesian_case):
    @gtx.field_operator
    def fieldop_args(a: cases.IField, condition: bool, scalar: int32) -> cases.IField:
        scalar = 0 if not condition else scalar
        return a + scalar

    @gtx.program(backend=cartesian_case.executor)
    def program_args(a: cases.IField, condition: bool, scalar: int32, out: cases.IField):
        fieldop_args(a, condition, scalar, out=out)

    a = cases.allocate(cartesian_case, program_args, "a")()
    out = cases.allocate(cartesian_case, program_args, "out")()

    prog_bounds = program_args.with_bound_args(condition=True)
    prog_bounds(a=a, scalar=int32(1), out=out, offset_provider={})
    np.allclose(out.asnumpy(), a.asnumpy() + int32(1))
