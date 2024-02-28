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
from gt4py.next import errors

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IField, cartesian_case  # noqa: F401 [unused-import]
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (  # noqa: F401 [unused-import]
    KDim,
    exec_alloc_descriptor,
)


def test_default_backend_is_respected_field_operator(cartesian_case):
    """Test that manually calling the field operator without setting the backend raises an error."""

    # Important not to set the backend here!
    @gtx.field_operator
    def copy(a: IField) -> IField:
        return a

    a = cases.allocate(cartesian_case, copy, "a")()

    with pytest.raises(ValueError, match="No backend selected!"):
        # Calling this should fail if the default backend is respected
        # due to `exec_alloc_descriptor` fixture (dependency of `cartesian_case`)
        # setting the default backend to something invalid.
        _ = copy(a, out=a, offset_provider={})


def test_default_backend_is_respected_scan_operator(cartesian_case):
    """Test that manually calling the scan operator without setting the backend raises an error."""

    # Important not to set the backend here!
    @gtx.scan_operator(axis=KDim, init=0.0, forward=True)
    def sum(state: float, a: float) -> float:
        return state + a

    a = gtx.ones({KDim: 10}, allocator=cartesian_case.allocator)

    with pytest.raises(ValueError, match="No backend selected!"):
        # see comment in field_operator test
        _ = sum(a, out=a, offset_provider={})


def test_default_backend_is_respected_program(cartesian_case):
    """Test that manually calling the program without setting the backend raises an error."""

    @gtx.field_operator
    def copy(a: IField) -> IField:
        return a

    # Important not to set the backend here!
    @gtx.program
    def copy_program(a: IField, b: IField) -> IField:
        copy(a, out=b)

    a = cases.allocate(cartesian_case, copy_program, "a")()
    b = cases.allocate(cartesian_case, copy_program, "b")()

    with pytest.raises(ValueError, match="No backend selected!"):
        # see comment in field_operator test
        _ = copy_program(a, b, offset_provider={})


def test_missing_arg_field_operator(cartesian_case):
    """Test that calling a field_operator without required args raises an error."""

    @gtx.field_operator(backend=cartesian_case.executor)
    def copy(a: IField) -> IField:
        return a

    a = cases.allocate(cartesian_case, copy, "a")()

    with pytest.raises(errors.MissingArgumentError, match="'out'"):
        _ = copy(a, offset_provider={})

    with pytest.raises(errors.MissingArgumentError, match="'offset_provider'"):
        _ = copy(a, out=a)


def test_missing_arg_scan_operator(cartesian_case):
    """Test that calling a scan_operator without required args raises an error."""

    @gtx.scan_operator(backend=cartesian_case.executor, axis=KDim, init=0.0, forward=True)
    def sum(state: float, a: float) -> float:
        return state + a

    a = cases.allocate(cartesian_case, sum, "a")()

    with pytest.raises(errors.MissingArgumentError, match="'out'"):
        _ = sum(a, offset_provider={})

    with pytest.raises(errors.MissingArgumentError, match="'offset_provider'"):
        _ = sum(a, out=a)


def test_missing_arg_program(cartesian_case):
    """Test that calling a program without required args raises an error."""

    @gtx.field_operator
    def copy(a: IField) -> IField:
        return a

    a = cases.allocate(cartesian_case, copy, "a")()
    b = cases.allocate(cartesian_case, copy, cases.RETURN)()

    with pytest.raises(errors.DSLError, match="Invalid call"):

        @gtx.program(backend=cartesian_case.executor)
        def copy_program(a: IField, b: IField) -> IField:
            copy(a)

        _ = copy_program(a, offset_provider={})

    with pytest.raises(TypeError, match="'offset_provider'"):

        @gtx.program(backend=cartesian_case.executor)
        def copy_program(a: IField, b: IField) -> IField:
            copy(a, out=b)

        _ = copy_program(a)
