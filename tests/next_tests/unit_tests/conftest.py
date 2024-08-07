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

from __future__ import annotations

import dataclasses
import os
import re
from typing import Generator, Optional

import pytest

import gt4py.next as gtx
from gt4py.next.iterator import runtime
from gt4py.next.program_processors import processor_interface as ppi


import next_tests

try:
    import xdist
except ImportError:
    xdist = None


@pytest.fixture(
    params=[
        (None, True),
        (next_tests.definitions.ProgramBackendId.ROUNDTRIP, True),
        (next_tests.definitions.ProgramBackendId.ROUNDTRIP_WITH_TEMPORARIES, True),
        (next_tests.definitions.ProgramBackendId.DOUBLE_ROUNDTRIP, True),
        (next_tests.definitions.ProgramBackendId.GTFN_CPU, True),
        (next_tests.definitions.ProgramBackendId.GTFN_CPU_IMPERATIVE, True),
        (next_tests.definitions.ProgramBackendId.GTFN_CPU_WITH_TEMPORARIES, True),
        # pytest.param((definitions.ProgramBackendId.GTFN_GPU, True), marks=pytest.mark.requires_gpu), # TODO(havogt): update tests to use proper allocation
        (next_tests.definitions.ProgramFormatterId.LISP_FORMATTER, False),
        (next_tests.definitions.ProgramFormatterId.ITIR_PRETTY_PRINTER, False),
        (next_tests.definitions.ProgramFormatterId.GTFN_CPP_FORMATTER, False),
        pytest.param(
            (next_tests.definitions.OptionalProgramBackendId.DACE_CPU, True),
            marks=pytest.mark.requires_dace,
        ),
        # TODO(havogt): update tests to use proper allocation
        # pytest.param(
        #     (next_tests.definitions.OptionalProgramBackendId.DACE_GPU, True),
        #     marks=(pytest.mark.requires_dace, pytest.mark.requires_gpu),
        # ),
    ],
    ids=lambda p: p[0].short_id() if p[0] is not None else "None",
)
def program_processor(
    request, testrun_uid
) -> Generator[tuple[ppi.ProgramProcessor, bool], None, None]:
    """
    Fixture creating program processors on-demand for tests.

    Notes:
        Check ADR 15 for details on the test-exclusion matrices.
    """
    processor_id, is_backend = request.param
    if processor_id is None:
        return None, is_backend

    gpu_env: Optional[str] = None
    processor = processor_id.load()
    assert is_backend == ppi.is_program_backend(processor)
    if is_backend:
        processor = processor.executor

    for marker, skip_mark, msg in next_tests.definitions.BACKEND_SKIP_TEST_MATRIX.get(
        processor_id, []
    ):
        if request.node.get_closest_marker(marker):
            skip_mark(msg.format(marker=marker, backend=processor_id))

    gpu_env: Optional[str] = None
    if xdist and request.node.get_closest_marker(pytest.mark.requires_gpu.name):
        import cupy
        num_gpu_devices = cupy.cuda.runtime.getDeviceCount()
        if num_gpu_devices > 0 and xdist.is_xdist_worker(request):
            wid = xdist.get_xdist_worker_id(request)
            wid_stripped = re.search(r".*(\d+)", wid).group(1)

            # override environment variable to make a single device visible to cupy
            gpu_env = os.getenv("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(int(wid_stripped) % num_gpu_devices)

    yield processor, is_backend

    if gpu_env:
        # restore environment variable
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_env


def run_processor(
    program: runtime.FendefDispatcher,
    processor: ppi.ProgramExecutor | ppi.ProgramFormatter,
    *args,
    **kwargs,
) -> None:
    if processor is None or ppi.is_processor_kind(processor, ppi.ProgramExecutor):
        program(*args, backend=processor, **kwargs)
    elif ppi.is_processor_kind(processor, ppi.ProgramFormatter):
        print(program.format_itir(*args, formatter=processor, **kwargs))
    else:
        raise TypeError(f"program processor kind not recognized: '{processor}'.")


@dataclasses.dataclass
class DummyConnectivity:
    max_neighbors: int
    has_skip_values: int
    origin_axis: gtx.Dimension = gtx.Dimension("dummy_origin")
    neighbor_axis: gtx.Dimension = gtx.Dimension("dummy_neighbor")
    index_type: type[int] = int

    def mapped_index(_, __) -> int:
        return 0
