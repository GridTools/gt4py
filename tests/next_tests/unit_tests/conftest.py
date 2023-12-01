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

import pytest

import gt4py.next as gtx
from gt4py.next.iterator import runtime, transforms
from gt4py.next.program_processors import processor_interface as ppi


try:
    from gt4py.next.program_processors.runners import dace_iterator
except ModuleNotFoundError as e:
    if "dace" in str(e):
        dace_iterator = None
    else:
        raise e

import next_tests
import next_tests.exclusion_matrices as definitions


@pytest.fixture(
    params=[
        transforms.LiftMode.FORCE_INLINE,
        transforms.LiftMode.FORCE_TEMPORARIES,
        transforms.LiftMode.SIMPLE_HEURISTIC,
    ],
    ids=lambda p: f"lift_mode={p.name}",
)
def lift_mode(request):
    return request.param


OPTIONAL_PROCESSORS = []
if dace_iterator:
    OPTIONAL_PROCESSORS.append((definitions.OptionalProgramBackendId.DACE_CPU, True))
    # TODO(havogt): update tests to use proper allocation
    # OPTIONAL_PROCESSORS.append(
    #     pytest.param(
    #         (definitions.OptionalProgramBackendId.DACE_GPU, True), marks=pytest.mark.requires_gpu
    #     )
    # ),


@pytest.fixture(
    params=[
        (None, True),
        (definitions.ProgramBackendId.ROUNDTRIP, True),
        (definitions.ProgramBackendId.DOUBLE_ROUNDTRIP, True),
        (definitions.ProgramBackendId.GTFN_CPU, True),
        (definitions.ProgramBackendId.GTFN_CPU_IMPERATIVE, True),
        (definitions.ProgramBackendId.GTFN_CPU_WITH_TEMPORARIES, True),
        # pytest.param((definitions.ProgramBackendId.GTFN_GPU, True), marks=pytest.mark.requires_gpu), # TODO(havogt): update tests to use proper allocation
        (definitions.ProgramFormatterId.LISP_FORMATTER, False),
        (definitions.ProgramFormatterId.ITIR_PRETTY_PRINTER, False),
        (definitions.ProgramFormatterId.ITIR_TYPE_CHECKER, False),
        (definitions.ProgramFormatterId.GTFN_CPP_FORMATTER, False),
    ]
    + OPTIONAL_PROCESSORS,
    ids=lambda p: p[0].short_id() if p[0] is not None else "None",
)
def program_processor(request) -> tuple[ppi.ProgramProcessor, bool]:
    """
    Fixture creating program processors on-demand for tests.

    Notes:
        Check ADR 15 for details on the test-exclusion matrices.
    """
    processor_id, is_backend = request.param
    if processor_id is None:
        return None, is_backend

    processor = processor_id.load()
    assert is_backend == ppi.is_program_backend(processor)

    for marker, skip_mark, msg in next_tests.exclusion_matrices.BACKEND_SKIP_TEST_MATRIX.get(
        processor_id, []
    ):
        if request.node.get_closest_marker(marker):
            skip_mark(msg.format(marker=marker, backend=processor_id))

    return processor, is_backend


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
        raise TypeError(f"program processor kind not recognized: {processor}!")


@dataclasses.dataclass
class DummyConnectivity:
    max_neighbors: int
    has_skip_values: int
    origin_axis: gtx.Dimension = gtx.Dimension("dummy_origin")
    neighbor_axis: gtx.Dimension = gtx.Dimension("dummy_neighbor")
    index_type: type[int] = int

    def mapped_index(_, __) -> int:
        return 0
