# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""GPU functional tests for multi-stream external workspace synchronization.

These tests require a CUDA or ROCm device and are skipped otherwise.
"""

import pytest

import numpy as np


dace = pytest.importorskip("dace")
cupy = pytest.importorskip("cupy")

import gt4py.next as gtx
from gt4py._core import definitions as core_defs
from gt4py.next import config
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations
from gt4py.next.program_processors.runners.dace.workflow import backend as dace_wf_backend

from next_tests.integration_tests import cases, cases_utils
from next_tests.integration_tests.cases_utils import KDim


pytestmark = pytest.mark.requires_gpu


def _make_external_allocator():
    """Return a simple external workspace allocator for GPU."""

    class _GPUAllocator:
        def allocate(self, request):
            raw = cupy.empty(request.nbytes + request.alignment - 1, dtype=cupy.uint8)
            offset = (-raw.__cuda_array_interface__["data"][0]) % request.alignment
            return raw[offset : offset + request.nbytes]

        def deallocate(self, buffer) -> None:
            pass

    return _GPUAllocator()


def test_external_mode_with_external_stream():
    """External mode runs correctly with a user-provided stream and multi-stream scheduling."""
    old_n_streams = config.MAX_CONCURRENT_GPU_STREAMS
    config.MAX_CONCURRENT_GPU_STREAMS = 4
    try:
        external_stream = cupy.cuda.Stream(non_blocking=True)

        backend = dace_wf_backend.make_dace_backend(
            gpu=True,
            auto_optimize=True,
            async_sdfg_call=True,
            cached_translation=False,
            optimization_args={
                "transient_memory_mode": gtx_transformations.TransientMemoryMode.EXTERNAL,
            },
            external_memory_allocator=_make_external_allocator(),
            external_sync_stream=external_stream,
        )

        @gtx.field_operator
        def testee_op(a: cases.IField, b: cases.IField) -> cases.IField:
            tmp = a + b
            return tmp + 1

        @gtx.program
        def testee(a: cases.IField, b: cases.IField, out: cases.IField):
            testee_op(a, b, out=out)

        test_case = cases.Case.from_cartesian_grid_descriptor(
            cases_utils.simple_cartesian_grid(),
            backend=backend,
            allocator=backend,
        )
        a = cases.allocate(test_case, testee, "a", strategy=cases.UniqueInitializer())()
        b = cases.allocate(test_case, testee, "b", strategy=cases.UniqueInitializer())()
        out = cases.allocate(test_case, testee, "out")()

        testee.with_backend(backend)(a, b, out=out, offset_provider={})

        assert np.allclose(out.asnumpy(), a.asnumpy() + b.asnumpy() + 1)
    finally:
        config.MAX_CONCURRENT_GPU_STREAMS = old_n_streams


def test_external_mode_with_default_stream_anchor():
    """External mode with multi-stream scheduling but no external stream uses the default stream."""
    old_n_streams = config.MAX_CONCURRENT_GPU_STREAMS
    config.MAX_CONCURRENT_GPU_STREAMS = 4
    try:
        backend = dace_wf_backend.make_dace_backend(
            gpu=True,
            auto_optimize=True,
            async_sdfg_call=True,
            cached_translation=False,
            optimization_args={
                "transient_memory_mode": gtx_transformations.TransientMemoryMode.EXTERNAL,
            },
            external_memory_allocator=_make_external_allocator(),
        )

        @gtx.field_operator
        def testee_op(a: cases.IField, b: cases.IField) -> cases.IField:
            tmp = a + b
            return tmp + 1

        @gtx.program
        def testee(a: cases.IField, b: cases.IField, out: cases.IField):
            testee_op(a, b, out=out)

        test_case = cases.Case.from_cartesian_grid_descriptor(
            cases_utils.simple_cartesian_grid(),
            backend=backend,
            allocator=backend,
        )
        a = cases.allocate(test_case, testee, "a", strategy=cases.UniqueInitializer())()
        b = cases.allocate(test_case, testee, "b", strategy=cases.UniqueInitializer())()
        out = cases.allocate(test_case, testee, "out")()

        testee.with_backend(backend)(a, b, out=out, offset_provider={})

        assert np.allclose(out.asnumpy(), a.asnumpy() + b.asnumpy() + 1)
    finally:
        config.MAX_CONCURRENT_GPU_STREAMS = old_n_streams
