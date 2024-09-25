# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses

import pytest

import gt4py.next as gtx
from gt4py.next.iterator import runtime
from gt4py.next.program_processors import processor_interface as ppi


import next_tests


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
    if is_backend:
        processor = processor.executor

    for marker, skip_mark, msg in next_tests.definitions.BACKEND_SKIP_TEST_MATRIX.get(
        processor_id, []
    ):
        if marker == next_tests.definitions.ALL or request.node.get_closest_marker(marker):
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
