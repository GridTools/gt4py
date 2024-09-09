# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test specific features of DaCe backends."""

import ctypes
from typing import Any
import numpy as np
import pytest
import unittest

import gt4py.next as gtx
from gt4py.next import int32
from gt4py.next.ffront.fbuiltins import where
from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    cartesian_case,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)

pytestmark = pytest.mark.requires_dace
compiled_sdfg = pytest.importorskip("dace.codegen.compiled_sdfg")


def get_scalar_values_from_sdfg_args(
    args: tuple[list[ctypes._SimpleCData], list[ctypes._SimpleCData]],
) -> list[Any]:
    runtime_args, init_args = args
    return [
        arg.value for arg in [*runtime_args, *init_args] if not isinstance(arg, ctypes.c_void_p)
    ]


def test_dace_fastcall(cartesian_case, monkeypatch):
    """Test reuse of SDFG arguments between program calls by means of SDFG fastcall API."""

    if not cartesian_case.executor or "dace" not in cartesian_case.executor.__name__:
        pytest.skip("DaCe-specific testcase.")

    @gtx.field_operator
    def testee(
        a: cases.IField,
        a_idx: cases.IField,
        unused_field: cases.IField,
        a0: int32,
        a1: int32,
        a2: int32,
        unused_scalar: int32,
    ) -> cases.IField:
        t0 = where(a_idx == 0, a + a0, a)
        t1 = where(a_idx == 1, t0 + a1, t0)
        t2 = where(a_idx == 2, t1 + a2, t1)
        return t2

    numpy_ref = lambda a, a0, a1, a2: [a[0] + a0, a[1] + a1, a[2] + a2, *a[3:]]

    a = cases.allocate(cartesian_case, testee, "a")()
    a_index = cases.allocate(cartesian_case, testee, "a_idx", strategy=cases.IndexInitializer())()
    a_offset = np.random.randint(1, 100, size=4, dtype=np.int32)
    unused_field = cases.allocate(cartesian_case, testee, "unused_field")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    # Wrap `compiled_sdfg.CompiledSDFG.fast_call` with mock object
    mock_fast_call = unittest.mock.MagicMock()
    mock_fast_call_attr = getattr(compiled_sdfg.CompiledSDFG, "fast_call")

    def mocked_fast_call(self, *args, **kwargs):
        mock_fast_call.__call__(*args, **kwargs)
        return mock_fast_call_attr(self, *args, **kwargs)

    monkeypatch.setattr(compiled_sdfg.CompiledSDFG, "fast_call", mocked_fast_call)

    # Wrap `compiled_sdfg.CompiledSDFG._construct_args` with mock object
    mock_construct_args = unittest.mock.MagicMock()
    mock_construct_args_attr = getattr(compiled_sdfg.CompiledSDFG, "_construct_args")

    def mocked_construct_args(self, *args, **kwargs):
        mock_construct_args.__call__(*args, **kwargs)
        return mock_construct_args_attr(self, *args, **kwargs)

    monkeypatch.setattr(compiled_sdfg.CompiledSDFG, "_construct_args", mocked_construct_args)

    # Reset mock objects and run/verify GT4Py program
    def verify_testee():
        mock_construct_args.reset_mock()
        mock_fast_call.reset_mock()
        cases.verify(
            cartesian_case,
            testee,
            a,
            a_index,
            unused_field,
            *a_offset,
            out=out,
            ref=numpy_ref(a.asnumpy(), *a_offset[0:3]),
        )
        mock_fast_call.assert_called_once()

    # On first run, the SDFG arguments will have to be constructed
    verify_testee()
    mock_construct_args.assert_called_once()
    # here we store the reference to the tuple of arguments passed to `fast_call` on first run and compare on successive runs
    fast_call_args = mock_fast_call.call_args.args
    # and the scalar values in the order they appear in the program ABI
    fast_call_scalar_values = get_scalar_values_from_sdfg_args(fast_call_args)

    def check_one_scalar_arg_changed(prev_scalar_args):
        new_scalar_args = get_scalar_values_from_sdfg_args(mock_fast_call.call_args.args)
        diff = np.array(new_scalar_args) - np.array(prev_scalar_args)
        assert np.count_nonzero(diff) == 1

    def check_scalar_args_all_same(prev_scalar_args):
        new_scalar_args = get_scalar_values_from_sdfg_args(mock_fast_call.call_args.args)
        diff = np.array(new_scalar_args) - np.array(prev_scalar_args)
        assert np.count_nonzero(diff) == 0

    def check_pointer_args_all_same():
        for arg, prev in zip(mock_fast_call.call_args.args, fast_call_args, strict=True):
            if isinstance(arg, ctypes._Pointer):
                assert arg == prev

    # Now modify the scalar arguments, used and unused ones: reuse previous SDFG arguments
    for i in range(4):
        a_offset[i] += 1
        verify_testee()
        mock_construct_args.assert_not_called()
        assert mock_fast_call.call_args.args == fast_call_args
        check_pointer_args_all_same()
        if i < 3:
            # same arguments tuple object but one scalar value is changed
            check_one_scalar_arg_changed(fast_call_scalar_values)
            # update reference scalar values
            fast_call_scalar_values = get_scalar_values_from_sdfg_args(fast_call_args)
        else:
            # unused scalar argument: the symbol is removed from the SDFG arglist and therefore no change
            check_scalar_args_all_same(fast_call_scalar_values)

    # Modify content of current buffer: reuse previous SDFG arguments
    for buff in (a, unused_field):
        buff[0] += 1
        verify_testee()
        mock_construct_args.assert_not_called()
        # same arguments tuple object and same content
        assert mock_fast_call.call_args.args == fast_call_args
        check_pointer_args_all_same()
        check_scalar_args_all_same(fast_call_scalar_values)

    # Pass a new buffer, which should trigger reconstruct of SDFG arguments: fastcall API will not be used
    a = cases.allocate(cartesian_case, testee, "a")()
    verify_testee()
    mock_construct_args.assert_called_once()
    assert mock_fast_call.call_args.args != fast_call_args
