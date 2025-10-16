# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import Field, Dims, gtfn_cpu
from gt4py.next.program_processors.runners import dace as dace_backends


# from .. import definitions

if TYPE_CHECKING:
    from pytest_benchmark import fixture as ptb_fixture


# from icon4py.model.common.dimension import E2C

Cell = gtx.Dimension("Cell")
K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)


@gtx.field_operator
def sample_fop(
    field1: Field[Dims[Cell, K], gtx.float64],
    field2: Field[Dims[Cell, K], gtx.float64],
    field3: Field[Dims[Cell, K], gtx.float64],
    field4: Field[Dims[Cell, K], gtx.float64],
    field5: Field[Dims[Cell, K], gtx.float64],
    field6: Field[Dims[Cell, K], gtx.float64],
) -> Field[Dims[Cell, K], gtx.float64]:
    out_e = field3
    return out_e


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def sample_program(
    field1: Field[Dims[Cell, K], gtx.float64],
    field2: Field[Dims[Cell, K], gtx.float64],
    field3: Field[Dims[Cell, K], gtx.float64],
    field4: Field[Dims[Cell, K], gtx.float64],
    field5: Field[Dims[Cell, K], gtx.float64],
    field6: Field[Dims[Cell, K], gtx.float64],
    out_e: Field[Dims[Cell, K], gtx.float64],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    sample_fop(
        field1,
        field2,
        field3,
        field4,
        field5,
        field6,
        out=out_e,
        domain={
            Cell: (horizontal_start, horizontal_end),
            K: (vertical_start, vertical_end),
        },
    )


def allocate_fields(domain: Sequence[tuple[gtx.Dimension, int]], *args: str) -> dict[str, Field]:
    sizes = [*zip(*domain)][1]
    return {arg: gtx.as_field(domain, np.random.rand(*sizes)) for arg in args}


@pytest.mark.parametrize("vertical_size", [1])
@pytest.mark.parametrize("horizontal_size", [1])
def benchmark_call(
    benchmark: ptb_fixture.BenchmarkFixture, horizontal_size: int, vertical_size: int
):
    domain = [(Cell, horizontal_size), (K, vertical_size)]
    all_fields = allocate_fields(
        domain, "out_e", "field1", "field2", "field3", "field4", "field5", "field6"
    )
    input_fields = [all_fields[f] for f in all_fields if f.startswith("field")]
    output_field = all_fields["out_e"]

    # Warmup and compile
    compiled_program = sample_program.with_backend(gtfn_cpu)
    compiled_program(
        *input_fields,
        out_e=output_field,
        horizontal_start=0,
        horizontal_end=horizontal_size,
        vertical_start=0,
        vertical_end=vertical_size,
    )

    benchmark(
        compiled_program,
        *input_fields,
        out_e=output_field,
        horizontal_start=0,
        horizontal_end=horizontal_size,
        vertical_start=0,
        vertical_end=vertical_size,
    )


if __name__ == "__main__":
    horizontal_size = 5
    vertical_size = 2
    domain = [(Cell, horizontal_size), (K, vertical_size)]
    all_fields = allocate_fields(
        domain, "out_e", "field1", "field2", "field3", "field4", "field5", "field6"
    )
    input_fields = [all_fields[f] for f in all_fields if f.startswith("field")]
    output_field = all_fields["out_e"]

    compiled_program = sample_program.with_backend(dace_backends.run_dace_cpu_cached)
    # compiled_program = sample_program.with_backend(gtfn_cpu)

    for i in range(6):
        compiled_program(
            *input_fields,
            output_field,
            0,
            horizontal_size,
            0,
            vertical_size,
            # out_e=output_field,
            # horizontal_start=0,
            # horizontal_end=horizontal_size,
            # vertical_start=0,
            # vertical_end=vertical_size,
        )
