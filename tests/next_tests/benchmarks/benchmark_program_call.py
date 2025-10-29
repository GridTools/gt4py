# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from gt4py import next as gtx
from gt4py.next import Dims, gtfn_cpu, gtfn_gpu, broadcast, typing as gtx_typing

BACKENDS: Final = [gtfn_cpu]

try:
    from gt4py.next.program_processors.runners import dace as dace_backends

    BACKENDS += [dace_backends.run_dace_cpu_cached]
except ImportError:
    pass


if TYPE_CHECKING:
    from pytest_benchmark import fixture as ptb_fixture


Cell = gtx.Dimension("Cell")
IDim = gtx.Dimension("IDim")


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def benchmark_const_no_args_program(
    benchmark: ptb_fixture.BenchmarkFixture, backend: gtx_typing.Backend
):
    @gtx.field_operator
    def const() -> gtx.Field[Dims[IDim], gtx.float64]:
        return broadcast(1.0, (IDim,))

    @gtx.program
    def const_no_args(out: gtx.Field[Dims[IDim], gtx.float64]):
        const(out=out)

    size = 1
    out_field = gtx.empty([(IDim, size)], dtype=gtx.float64)

    # Initial compilation
    compiled_program = const_no_args.with_backend(backend)
    compiled_program(out=out_field)

    benchmark(compiled_program, out=out_field)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def benchmark_copy_01_arg_program(
    benchmark: ptb_fixture.BenchmarkFixture, backend: gtx_typing.Backend
):
    @gtx.field_operator
    def identity_fop(
        in_field: gtx.Field[Dims[IDim], gtx.float64],
    ) -> gtx.Field[Dims[IDim], gtx.float64]:
        return in_field

    @gtx.program
    def copy_01_arg(
        in_field: gtx.Field[Dims[IDim], gtx.float64], out: gtx.Field[Dims[IDim], gtx.float64]
    ):
        identity_fop(in_field, out=out)

    size = 1
    in_field = gtx.full([(IDim, size)], 1, dtype=gtx.float64)
    out_field = gtx.empty([(IDim, size)], dtype=gtx.float64)

    # Initial compilation
    compiled_program = copy_01_arg.with_backend(backend)
    compiled_program(in_field, out=out_field)

    benchmark(compiled_program, in_field, out=out_field)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def benchmark_horizontal_copy_01_arg_program(
    benchmark: ptb_fixture.BenchmarkFixture, backend: gtx_typing.Backend
):
    @gtx.field_operator
    def identity_01_fop(
        in_field: gtx.Field[Dims[Cell], gtx.float64],
    ) -> gtx.Field[Dims[Cell], gtx.float64]:
        return in_field

    @gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
    def horizontal_copy_01_arg_program(
        in_field: gtx.Field[Dims[Cell], gtx.float64],
        out: gtx.Field[Dims[Cell], gtx.float64],
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ) -> None:
        identity_01_fop(
            in_field,
            out=out,
            domain={
                Cell: (horizontal_start, horizontal_end),
            },
        )

    size = 1
    input_field = gtx.empty([(Cell, size)], dtype=gtx.float64)
    out_field = gtx.empty([(Cell, size)], dtype=gtx.float64)

    # Initial compilation
    compiled_program = horizontal_copy_01_arg_program.with_backend(backend)
    compiled_program(
        input_field,
        out=out_field,
        horizontal_start=0,
        horizontal_end=size,
    )

    benchmark(
        compiled_program,
        input_field,
        out=out_field,
        horizontal_start=0,
        horizontal_end=size,
    )


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def benchmark_horizontal_copy_05_arg_program(
    benchmark: ptb_fixture.BenchmarkFixture, backend: gtx_typing.Backend
):
    @gtx.field_operator
    def identity_05_fop(
        field0: gtx.Field[Dims[Cell], gtx.float64],
        field1: gtx.Field[Dims[Cell], gtx.float64],
        field2: gtx.Field[Dims[Cell], gtx.float64],
        field3: gtx.Field[Dims[Cell], gtx.float64],
        field4: gtx.Field[Dims[Cell], gtx.float64],
    ) -> gtx.Field[Dims[Cell], gtx.float64]:
        return field4

    @gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
    def horizontal_copy_05_arg_program(
        field0: gtx.Field[Dims[Cell], gtx.float64],
        field1: gtx.Field[Dims[Cell], gtx.float64],
        field2: gtx.Field[Dims[Cell], gtx.float64],
        field3: gtx.Field[Dims[Cell], gtx.float64],
        field4: gtx.Field[Dims[Cell], gtx.float64],
        out: gtx.Field[Dims[Cell], gtx.float64],
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ) -> None:
        identity_05_fop(
            field0,
            field1,
            field2,
            field3,
            field4,
            out=out,
            domain={
                Cell: (horizontal_start, horizontal_end),
            },
        )

    size = 1
    input_fields = [gtx.empty([(Cell, size)], dtype=gtx.float64) for _ in range(5)]
    out_field = gtx.empty([(Cell, size)], dtype=gtx.float64)

    # Initial compilation
    compiled_program = horizontal_copy_05_arg_program.with_backend(backend)
    compiled_program(
        *input_fields,
        out=out_field,
        horizontal_start=0,
        horizontal_end=size,
    )

    benchmark(
        compiled_program,
        *input_fields,
        out=out_field,
        horizontal_start=0,
        horizontal_end=size,
    )


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def benchmark_horizontal_copy_25_arg_program(
    benchmark: ptb_fixture.BenchmarkFixture, backend: gtx_typing.Backend
):
    @gtx.field_operator
    def identity_25_fop(
        field0: gtx.Field[Dims[Cell], gtx.float64],
        field1: gtx.Field[Dims[Cell], gtx.float64],
        field2: gtx.Field[Dims[Cell], gtx.float64],
        field3: gtx.Field[Dims[Cell], gtx.float64],
        field4: gtx.Field[Dims[Cell], gtx.float64],
        field5: gtx.Field[Dims[Cell], gtx.float64],
        field6: gtx.Field[Dims[Cell], gtx.float64],
        field7: gtx.Field[Dims[Cell], gtx.float64],
        field8: gtx.Field[Dims[Cell], gtx.float64],
        field9: gtx.Field[Dims[Cell], gtx.float64],
        field10: gtx.Field[Dims[Cell], gtx.float64],
        field11: gtx.Field[Dims[Cell], gtx.float64],
        field12: gtx.Field[Dims[Cell], gtx.float64],
        field13: gtx.Field[Dims[Cell], gtx.float64],
        field14: gtx.Field[Dims[Cell], gtx.float64],
        field15: gtx.Field[Dims[Cell], gtx.float64],
        field16: gtx.Field[Dims[Cell], gtx.float64],
        field17: gtx.Field[Dims[Cell], gtx.float64],
        field18: gtx.Field[Dims[Cell], gtx.float64],
        field19: gtx.Field[Dims[Cell], gtx.float64],
        field20: gtx.Field[Dims[Cell], gtx.float64],
        field21: gtx.Field[Dims[Cell], gtx.float64],
        field22: gtx.Field[Dims[Cell], gtx.float64],
        field23: gtx.Field[Dims[Cell], gtx.float64],
        field24: gtx.Field[Dims[Cell], gtx.float64],
    ) -> gtx.Field[Dims[Cell], gtx.float64]:
        return field24

    @gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
    def horizontal_copy_25_arg_program(
        field0: gtx.Field[Dims[Cell], gtx.float64],
        field1: gtx.Field[Dims[Cell], gtx.float64],
        field2: gtx.Field[Dims[Cell], gtx.float64],
        field3: gtx.Field[Dims[Cell], gtx.float64],
        field4: gtx.Field[Dims[Cell], gtx.float64],
        field5: gtx.Field[Dims[Cell], gtx.float64],
        field6: gtx.Field[Dims[Cell], gtx.float64],
        field7: gtx.Field[Dims[Cell], gtx.float64],
        field8: gtx.Field[Dims[Cell], gtx.float64],
        field9: gtx.Field[Dims[Cell], gtx.float64],
        field10: gtx.Field[Dims[Cell], gtx.float64],
        field11: gtx.Field[Dims[Cell], gtx.float64],
        field12: gtx.Field[Dims[Cell], gtx.float64],
        field13: gtx.Field[Dims[Cell], gtx.float64],
        field14: gtx.Field[Dims[Cell], gtx.float64],
        field15: gtx.Field[Dims[Cell], gtx.float64],
        field16: gtx.Field[Dims[Cell], gtx.float64],
        field17: gtx.Field[Dims[Cell], gtx.float64],
        field18: gtx.Field[Dims[Cell], gtx.float64],
        field19: gtx.Field[Dims[Cell], gtx.float64],
        field20: gtx.Field[Dims[Cell], gtx.float64],
        field21: gtx.Field[Dims[Cell], gtx.float64],
        field22: gtx.Field[Dims[Cell], gtx.float64],
        field23: gtx.Field[Dims[Cell], gtx.float64],
        field24: gtx.Field[Dims[Cell], gtx.float64],
        out: gtx.Field[Dims[Cell], gtx.float64],
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ) -> None:
        identity_25_fop(
            field0,
            field1,
            field2,
            field3,
            field4,
            field5,
            field6,
            field7,
            field8,
            field9,
            field10,
            field11,
            field12,
            field13,
            field14,
            field15,
            field16,
            field17,
            field18,
            field19,
            field20,
            field21,
            field22,
            field23,
            field24,
            out=out,
            domain={
                Cell: (horizontal_start, horizontal_end),
            },
        )

    size = 1
    input_fields = [gtx.empty([(Cell, size)], dtype=gtx.float64) for _ in range(25)]
    out_field = gtx.empty([(Cell, size)], dtype=gtx.float64)

    # Initial compilation
    compiled_program = horizontal_copy_25_arg_program.with_backend(backend)
    compiled_program(
        *input_fields,
        out=out_field,
        horizontal_start=0,
        horizontal_end=size,
    )

    benchmark(
        compiled_program,
        *input_fields,
        out=out_field,
        horizontal_start=0,
        horizontal_end=size,
    )


# This is useful for running this module as a python script,
# mostly for profiling/debugging backends.
if __name__ == "__main__":
    import functools
    import sys

    def benchmark(*args, **kwargs):
        func = functools.partial(*args, **kwargs)
        for _ in range(5):
            func()

    backends = []
    for arg in sys.argv[1:]:
        if arg.startswith("--backend="):
            backend_name = arg.split("=", 1)[1]
            match backend_name:
                case "dace-cpu":
                    backends.append(dace_backends.run_dace_cpu_cached)
                case "dace-gpu":
                    backends.append(dace_backends.run_dace_gpu_cached)
                case "gtfn-cpu":
                    backends.append(gtfn_cpu)
                case "gtfn-gpu":
                    backends.append(gtfn_gpu)
                case _:
                    raise ValueError(f"Unknown backend: {backend_name}")

    backends = backends or BACKENDS

    for backend in backends:
        print(f"Running benchmarks with backend: {backend.name}")
        benchmark_horizontal_copy_05_arg_program(benchmark, backend)
