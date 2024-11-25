# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import TypeAlias

import pytest

import gt4py.next as gtx
from gt4py.next import backend, common
from gt4py.next.embedded import nd_array_field
from gt4py.next.iterator import runtime
from gt4py.next.program_processors import program_formatter

import next_tests


ProgramProcessor: TypeAlias = backend.Backend | program_formatter.ProgramFormatter


def _program_processor(request) -> tuple[ProgramProcessor, bool]:
    """
    Fixture creating program processors on-demand for tests.

    Notes:
        Check ADR 15 for details on the test-exclusion matrices.
    """
    processor_id, is_backend = request.param
    if processor_id is None:
        return None, is_backend

    processor = processor_id.load()

    for marker, skip_mark, msg in next_tests.definitions.BACKEND_SKIP_TEST_MATRIX.get(
        processor_id, []
    ):
        if marker == next_tests.definitions.ALL or request.node.get_closest_marker(marker):
            skip_mark(msg.format(marker=marker, backend=processor_id))

    return processor, is_backend


program_processor = pytest.fixture(
    _program_processor,
    params=[
        (None, True),
        (next_tests.definitions.ProgramBackendId.ROUNDTRIP, True),
        (next_tests.definitions.ProgramBackendId.ROUNDTRIP_WITH_TEMPORARIES, True),
        (next_tests.definitions.ProgramBackendId.DOUBLE_ROUNDTRIP, True),
        (next_tests.definitions.ProgramBackendId.GTFN_CPU, True),
        (next_tests.definitions.ProgramBackendId.GTFN_CPU_IMPERATIVE, True),
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

program_processor_no_transforms = pytest.fixture(
    _program_processor,
    params=[
        (None, True),
        (next_tests.definitions.ProgramBackendId.GTFN_CPU_NO_TRANSFORMS, True),
        (next_tests.definitions.ProgramBackendId.ROUNDTRIP_NO_TRANSFORMS, True),
    ],
    ids=lambda p: p[0].short_id() if p[0] is not None else "None",
)


def run_processor(
    program: runtime.FendefDispatcher,
    processor: ProgramProcessor,
    *args,
    **kwargs,
) -> None:
    if processor is None or isinstance(processor, ProgramProcessor):
        program(*args, backend=processor, **kwargs)
    else:
        raise TypeError(f"program processor kind not recognized: '{processor}'.")


@dataclasses.dataclass
class DummyConnectivity(common.Connectivity):
    max_neighbors: int
    has_skip_values: int
    source_dim: gtx.Dimension = gtx.Dimension("dummy_origin")
    codomain: gtx.Dimension = gtx.Dimension("dummy_neighbor")


def nd_array_implementation_params():
    for xp in nd_array_field._nd_array_implementations:
        if hasattr(nd_array_field, "cp") and xp == nd_array_field.cp:
            yield pytest.param(xp, id=xp.__name__, marks=pytest.mark.requires_gpu)
        else:
            yield pytest.param(xp, id=xp.__name__)


@pytest.fixture(params=nd_array_implementation_params())
def nd_array_implementation(request):
    yield request.param
