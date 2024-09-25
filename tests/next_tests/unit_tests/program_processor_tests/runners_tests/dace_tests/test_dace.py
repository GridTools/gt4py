# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test specific features of DaCe backends."""

import unittest

import numpy as np
import pytest

import gt4py._core.definitions as core_defs
import gt4py.next as gtx
from gt4py.next import common as gtx_common
from gt4py.next.ffront.fbuiltins import where

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    E2V,
    cartesian_case,
    unstructured_case,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
    mesh_descriptor,
)
from unittest.mock import patch

from . import pytestmark

dace = pytest.importorskip("dace")


def make_mocks(monkeypatch):
    # Wrap `compiled_sdfg.CompiledSDFG.fast_call` with mock object
    mock_fast_call = unittest.mock.MagicMock()
    mock_fast_call_attr = dace.codegen.compiled_sdfg.CompiledSDFG.fast_call

    def mocked_fast_call(self, *args, **kwargs):
        mock_fast_call.__call__(*args, **kwargs)
        fast_call_result = mock_fast_call_attr(self, *args, **kwargs)
        # invalidate all scalar positional arguments to ensure that they are properly set
        # next time the SDFG is executed before fast_call
        positional_args = set(self.sdfg.arg_names)
        sdfg_arglist = self.sdfg.arglist()
        for i, (arg_name, arg_type) in enumerate(sdfg_arglist.items()):
            if arg_name in positional_args and isinstance(arg_type, dace.data.Scalar):
                self._lastargs[0][i] = None
        return fast_call_result

    monkeypatch.setattr(dace.codegen.compiled_sdfg.CompiledSDFG, "fast_call", mocked_fast_call)

    # Wrap `compiled_sdfg.CompiledSDFG._construct_args` with mock object
    mock_construct_args = unittest.mock.MagicMock()
    mock_construct_args_attr = dace.codegen.compiled_sdfg.CompiledSDFG._construct_args

    def mocked_construct_args(self, *args, **kwargs):
        mock_construct_args.__call__(*args, **kwargs)
        return mock_construct_args_attr(self, *args, **kwargs)

    monkeypatch.setattr(
        dace.codegen.compiled_sdfg.CompiledSDFG, "_construct_args", mocked_construct_args
    )

    return mock_fast_call, mock_construct_args


def test_dace_fastcall(cartesian_case, monkeypatch):
    """Test reuse of SDFG arguments between program calls by means of SDFG fastcall API."""

    if not cartesian_case.executor or "dace" not in cartesian_case.executor.__name__:
        pytest.skip("requires dace backend")

    @gtx.field_operator
    def testee(
        a: cases.IField,
        a_idx: cases.IField,
        unused_field: cases.IField,
        a0: gtx.int32,
        a1: gtx.int32,
        a2: gtx.int32,
        unused_scalar: gtx.int32,
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

    mock_fast_call, mock_construct_args = make_mocks(monkeypatch)

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

    # Now modify the scalar arguments, used and unused ones: reuse previous SDFG arguments
    for i in range(4):
        a_offset[i] += 1
        verify_testee()
        mock_construct_args.assert_not_called()

    # Modify content of current buffer: reuse previous SDFG arguments
    for buff in (a, unused_field):
        buff[0] += 1
        verify_testee()
        mock_construct_args.assert_not_called()

    # Pass a new buffer, which should trigger reconstruct of SDFG arguments: fastcall API will not be used
    a = cases.allocate(cartesian_case, testee, "a")()
    verify_testee()
    mock_construct_args.assert_called_once()


def test_dace_fastcall_with_connectivity(unstructured_case, monkeypatch):
    """Test reuse of SDFG arguments between program calls by means of SDFG fastcall API."""

    if not unstructured_case.executor or "dace" not in unstructured_case.executor.__name__:
        pytest.skip("requires dace backend")

    connectivity_E2V = unstructured_case.offset_provider["E2V"]
    assert isinstance(connectivity_E2V, gtx_common.NeighborTable)

    # check that test connectivities are allocated on host memory
    # this is an assumption to test that fast_call cannot be used for gpu tests
    assert isinstance(connectivity_E2V.table, np.ndarray)

    @gtx.field_operator
    def testee(a: cases.VField) -> cases.EField:
        return a(E2V[0])

    (a,), kwfields = cases.get_default_data(unstructured_case, testee)
    numpy_ref = lambda a: a[connectivity_E2V.table[:, 0]]

    mock_fast_call, mock_construct_args = make_mocks(monkeypatch)

    # Reset mock objects and run/verify GT4Py program
    def verify_testee(offset_provider):
        mock_construct_args.reset_mock()
        mock_fast_call.reset_mock()
        cases.verify(
            unstructured_case,
            testee,
            a,
            **kwfields,
            offset_provider=offset_provider,
            ref=numpy_ref(a.asnumpy()),
        )
        mock_fast_call.assert_called_once()

    verify_testee(unstructured_case.offset_provider)
    mock_construct_args.assert_called_once()

    if gtx.allocators.is_field_allocator_for(
        unstructured_case.executor.allocator, core_defs.DeviceType.CPU
    ):
        verify_testee(unstructured_case.offset_provider)
        mock_construct_args.assert_not_called()
    else:
        import cupy as cp

        assert gtx.allocators.is_field_allocator_for(
            unstructured_case.executor.allocator, gtx.allocators.CUPY_DEVICE
        )

        # the test connectivities are numpy arrays, and they are copied to gpu memory
        # at each program call, therefore fast_call cannot be used in this case
        verify_testee(unstructured_case.offset_provider)
        mock_construct_args.assert_called()

        # copy the connectivity to gpu memory, so that fast_call should be used
        cupy_offset_provider = {
            "E2V": gtx_common.NeighborTable(
                max_neighbors=connectivity_E2V.max_neighbors,
                has_skip_values=connectivity_E2V.has_skip_values,
                origin_axis=connectivity_E2V.origin_axis,
                neighbor_axis=connectivity_E2V.neighbor_axis,
                index_type=connectivity_E2V.index_type,
                table=cp.asarray(connectivity_E2V.table),
            )
        }

        verify_testee(cupy_offset_provider)
        mock_construct_args.assert_called()
        verify_testee(cupy_offset_provider)
        mock_construct_args.assert_not_called()
