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

from functools import reduce

import numpy as np
import pytest

from gt4py.next import Field, errors, field_operator, float64, index_field, np_as_located_field
from gt4py.next.program_processors.runners import dace_iterator, gtfn_cpu

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    E2V,
    V2E,
    E2VDim,
    Edge,
    IDim,
    Ioff,
    JDim,
    Joff,
    KDim,
    Koff,
    V2EDim,
    Vertex,
    cartesian_case,
    unstructured_case,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    Cell,
    fieldview_backend,
    size,
)


@pytest.mark.parametrize("condition", [True, False])
def test_simple_if(condition, cartesian_case):
    if cartesian_case.backend in [
        gtfn_cpu.run_gtfn,
        gtfn_cpu.run_gtfn_imperative,
        dace_iterator.run_dace_iterator,
    ]:
        pytest.xfail("If-stmts are not supported yet.")

    @field_operator
    def simple_if(a: cases.IField, b: cases.IField, condition: bool) -> cases.IField:
        if condition:
            result = a
        else:
            result = b
        return result

    a = cases.allocate(cartesian_case, simple_if, "a")()
    b = cases.allocate(cartesian_case, simple_if, "b")()
    out = cases.allocate(cartesian_case, simple_if, cases.RETURN)()

    cases.verify(cartesian_case, simple_if, a, b, condition, out=out, ref=a if condition else b)


@pytest.mark.parametrize("condition1, condition2", [[True, False], [True, False]])
def test_simple_if_conditional(condition1, condition2, cartesian_case):
    if cartesian_case.backend in [
        gtfn_cpu.run_gtfn,
        gtfn_cpu.run_gtfn_imperative,
        dace_iterator.run_dace_iterator,
    ]:
        pytest.xfail("If-stmts are not supported yet.")

    @field_operator
    def simple_if(
        a: cases.IField,
        b: cases.IField,
        condition1: bool,
        condition2: bool,
    ) -> cases.IField:
        if condition1:
            result1 = a
            result2 = a + 1
        else:
            result1 = b
            result2 = b + 1
        return result1 if condition2 else result2

    a = cases.allocate(cartesian_case, simple_if, "a")()
    b = cases.allocate(cartesian_case, simple_if, "b")()
    out = cases.allocate(cartesian_case, simple_if, cases.RETURN)()

    cases.verify(
        cartesian_case,
        simple_if,
        a,
        b,
        condition1,
        condition2,
        out=out,
        ref=(a if condition1 else b).array() + (0 if condition2 else 1),
    )


@pytest.mark.parametrize("condition", [True, False])
def test_local_if(cartesian_case, condition):
    if cartesian_case.backend in [
        gtfn_cpu.run_gtfn,
        gtfn_cpu.run_gtfn_imperative,
        dace_iterator.run_dace_iterator,
    ]:
        pytest.xfail("If-stmts are not supported yet.")

    @field_operator
    def local_if(a: cases.IField, b: cases.IField, condition: bool) -> cases.IField:
        if condition:
            tmp = a
            result = tmp
        else:
            result = b
        return result

    a = cases.allocate(cartesian_case, local_if, "a")()
    b = cases.allocate(cartesian_case, local_if, "b")()
    out = cases.allocate(cartesian_case, local_if, cases.RETURN)()

    cases.verify(cartesian_case, local_if, a, b, condition, out=out, ref=(a if condition else b))


@pytest.mark.parametrize("condition", [True, False])
def test_temporary_if(cartesian_case, condition):
    if cartesian_case.backend in [
        gtfn_cpu.run_gtfn,
        gtfn_cpu.run_gtfn_imperative,
        dace_iterator.run_dace_iterator,
    ]:
        pytest.xfail("If-stmts are not supported yet.")

    @field_operator
    def temporary_if(a: cases.IField, b: cases.IField, condition: bool) -> cases.IField:
        if condition:
            tmp1 = a
            result = tmp1
        else:
            tmp2 = b
            result = tmp2
        return result

    a = cases.allocate(cartesian_case, temporary_if, "a")()
    b = cases.allocate(cartesian_case, temporary_if, "b")()
    out = cases.allocate(cartesian_case, temporary_if, cases.RETURN)()

    cases.verify(
        cartesian_case, temporary_if, a, b, condition, out=out, ref=(a if condition else b)
    )


@pytest.mark.parametrize("condition", [True, False])
def test_if_return(cartesian_case, condition):
    if cartesian_case.backend in [
        gtfn_cpu.run_gtfn,
        gtfn_cpu.run_gtfn_imperative,
        dace_iterator.run_dace_iterator,
    ]:
        pytest.xfail("If-stmts are not supported yet.")

    @field_operator
    def temporary_if(a: cases.IField, b: cases.IField, condition: bool) -> cases.IField:
        if condition:
            tmp1 = a
            return tmp1
        else:
            tmp2 = b
            return tmp2
        return a + b

    a = cases.allocate(cartesian_case, temporary_if, "a")()
    b = cases.allocate(cartesian_case, temporary_if, "b")()
    out = cases.allocate(cartesian_case, temporary_if, cases.RETURN)()

    cases.verify(
        cartesian_case, temporary_if, a, b, condition, out=out, ref=(a if condition else b)
    )


@pytest.mark.parametrize("condition", [True, False])
def test_if_stmt_if_branch_returns(cartesian_case, condition):
    if cartesian_case.backend in [
        gtfn_cpu.run_gtfn,
        gtfn_cpu.run_gtfn_imperative,
        dace_iterator.run_dace_iterator,
    ]:
        pytest.xfail("If-stmts are not supported yet.")

    @field_operator
    def if_branch_returns(a: cases.IField, b: cases.IField, condition: bool) -> cases.IField:
        if condition:
            tmp1 = a
            return tmp1
        return b

    a = cases.allocate(cartesian_case, if_branch_returns, "a")()
    b = cases.allocate(cartesian_case, if_branch_returns, "b")()
    out = cases.allocate(cartesian_case, if_branch_returns, cases.RETURN)()

    cases.verify(
        cartesian_case, if_branch_returns, a, b, condition, out=out, ref=(a if condition else b)
    )


@pytest.mark.parametrize("condition", [True, False])
def test_if_stmt_else_branch_returns(cartesian_case, condition):
    if cartesian_case.backend in [
        gtfn_cpu.run_gtfn,
        gtfn_cpu.run_gtfn_imperative,
        dace_iterator.run_dace_iterator,
    ]:
        pytest.xfail("If-stmts are not supported yet.")

    @field_operator
    def else_branch_returns(a: cases.IField, b: cases.IField, condition: bool) -> cases.IField:
        if condition:
            pass
        else:
            tmp1 = b
            return tmp1
        return a

    a = cases.allocate(cartesian_case, else_branch_returns, "a")()
    b = cases.allocate(cartesian_case, else_branch_returns, "b")()
    out = cases.allocate(cartesian_case, else_branch_returns, cases.RETURN)()

    cases.verify(
        cartesian_case, else_branch_returns, a, b, condition, out=out, ref=(a if condition else b)
    )


@pytest.mark.parametrize("condition", [True, False])
def test_if_stmt_both_branches_return(cartesian_case, condition):
    if cartesian_case.backend in [
        gtfn_cpu.run_gtfn,
        gtfn_cpu.run_gtfn_imperative,
        dace_iterator.run_dace_iterator,
    ]:
        pytest.xfail("If-stmts are not supported yet.")

    @field_operator
    def both_branches_return(a: cases.IField, b: cases.IField, condition: bool) -> cases.IField:
        if condition:
            tmp1 = a
            return tmp1
        else:
            tmp2 = b
            return tmp2

    a = cases.allocate(cartesian_case, both_branches_return, "a")()
    b = cases.allocate(cartesian_case, both_branches_return, "b")()
    out = cases.allocate(cartesian_case, both_branches_return, cases.RETURN)()

    cases.verify(
        cartesian_case, both_branches_return, a, b, condition, out=out, ref=(a if condition else b)
    )


@pytest.mark.parametrize("condition1, condition2", [[True, False], [True, False]])
def test_nested_if_stmt_conditional(cartesian_case, condition1, condition2):
    if cartesian_case.backend in [
        gtfn_cpu.run_gtfn,
        gtfn_cpu.run_gtfn_imperative,
        dace_iterator.run_dace_iterator,
    ]:
        pytest.xfail("If-stmts are not supported yet.")

    @field_operator
    def nested_if_conditional_return(
        inp: cases.IField, condition1: bool, condition2: bool
    ) -> cases.IField:
        if condition1:
            tmp1 = inp
            if condition2:
                return tmp1 + 1
            result = tmp1 + 2
        else:
            result = inp + 3
        return result

    inp = cases.allocate(cartesian_case, nested_if_conditional_return, "inp")()
    out = cases.allocate(cartesian_case, nested_if_conditional_return, cases.RETURN)()

    ref = {
        (True, True): np.asarray(inp) + 1,
        (True, False): np.asarray(inp) + 2,
        (False, True): np.asarray(inp) + 3,
        (False, False): np.asarray(inp) + 3,
    }

    cases.verify(
        cartesian_case,
        nested_if_conditional_return,
        inp,
        condition1,
        condition2,
        out=out,
        ref=ref[(condition1, condition2)],
    )


@pytest.mark.parametrize("condition", [True, False])
def test_nested_if(cartesian_case, condition):
    if cartesian_case.backend in [
        gtfn_cpu.run_gtfn,
        gtfn_cpu.run_gtfn_imperative,
        dace_iterator.run_dace_iterator,
    ]:
        pytest.xfail("If-stmts are not supported yet.")

    @field_operator
    def nested_if(a: cases.IField, b: cases.IField, condition: bool) -> cases.IField:
        if condition:
            if not condition:
                inner = a
            else:
                inner = a + 1
            result = inner
        else:
            result = b
            if condition:
                another_inner = 3
            else:
                another_inner = 5
            result = result + another_inner
        return result

    a = cases.allocate(cartesian_case, nested_if, "a")()
    b = cases.allocate(cartesian_case, nested_if, "b")()
    out = cases.allocate(cartesian_case, nested_if, cases.RETURN)()

    cases.verify(
        cartesian_case,
        nested_if,
        a,
        b,
        condition,
        out=out,
        ref=np.asarray(a) + 1 if condition else np.asarray(b) + 5,
    )


@pytest.mark.parametrize("condition1, condition2", [[True, False], [True, False]])
def test_if_without_else(cartesian_case, condition1, condition2):
    if cartesian_case.backend in [
        gtfn_cpu.run_gtfn,
        gtfn_cpu.run_gtfn_imperative,
        dace_iterator.run_dace_iterator,
    ]:
        pytest.xfail("If-stmts are not supported yet.")

    @field_operator
    def if_without_else(
        a: cases.IField, b: cases.IField, condition1: bool, condition2: bool
    ) -> cases.IField:
        result = b + 1

        if condition1:
            if not condition2:
                inner = a
            else:
                inner = a + 2
            result = inner
        return result

    a = cases.allocate(cartesian_case, if_without_else, "a")()
    b = cases.allocate(cartesian_case, if_without_else, "b")()
    out = cases.allocate(cartesian_case, if_without_else, cases.RETURN)()

    ref = {
        (True, True): np.asarray(a) + 2,
        (True, False): np.asarray(a),
        (False, True): np.asarray(b) + 1,
        (False, False): np.asarray(b) + 1,
    }

    cases.verify(
        cartesian_case,
        if_without_else,
        a,
        b,
        condition1,
        condition2,
        out=out,
        ref=ref[(condition1, condition2)],
    )


def test_if_non_scalar_condition():
    with pytest.raises(errors.DSLError, match="Condition for `if` must be scalar."):

        @field_operator
        def if_non_scalar_condition(
            a: Field[[IDim, JDim], float64], b: Field[[IDim, JDim], float64], condition: bool
        ):
            result = a
            if a == b:
                result = b
            return result


def test_if_non_boolean_condition():
    with pytest.raises(errors.DSLError, match="Condition for `if` must be of boolean type."):

        @field_operator
        def if_non_boolean_condition(
            a: Field[[IDim, JDim], float64], b: Field[[IDim, JDim], float64], condition: float64
        ):
            if condition:
                result = b
            else:
                result = a
            return result


def test_if_inconsistent_types():
    with pytest.raises(
        errors.DSLError,
        match="Inconsistent types between two branches for variable",
    ):

        @field_operator
        def if_inconsistent_types(
            a: Field[[IDim, JDim], float64], b: Field[[IDim, JDim], float64], condition: bool
        ):
            if condition:
                result = 1
            else:
                result = 2.0
            return result
