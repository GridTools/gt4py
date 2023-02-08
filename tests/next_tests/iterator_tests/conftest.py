# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

from dataclasses import dataclass

import pytest

from gt4py.next import common
from gt4py.next.iterator import ir as itir, pretty_parser, pretty_printer, runtime, transforms
from gt4py.next.program_processors import processor_interface as ppi
from gt4py.next.program_processors.formatters import gtfn, lisp, type_check
from gt4py.next.program_processors.runners import double_roundtrip, gtfn_cpu, roundtrip


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


@ppi.program_formatter
def pretty_format_and_check(root: itir.FencilDefinition, *args, **kwargs) -> str:
    pretty = pretty_printer.pformat(root)
    parsed = pretty_parser.pparse(pretty)
    assert parsed == root
    return pretty


def get_processor_id(processor):
    if hasattr(processor, "__module__") and hasattr(processor, "__name__"):
        module_path = processor.__module__.split(".")[-1]
        name = processor.__name__
        return f"{module_path}.{name}"
    return repr(processor)


@pytest.fixture(
    params=[
        # (processor, do_validate)
        (None, True),
        (lisp.format_lisp, False),
        (pretty_format_and_check, False),
        (roundtrip.executor, True),
        (type_check.check, False),
        (double_roundtrip.executor, True),
        (gtfn_cpu.run_gtfn, True),
        (gtfn_cpu.run_gtfn_imperative, True),
        (gtfn.format_sourcecode, False),
    ],
    ids=lambda p: get_processor_id(p[0]),
)
def program_processor(request):
    return request.param


@pytest.fixture
def program_processor_no_gtfn_exec(program_processor):
    if (
        program_processor[0] == gtfn_cpu.run_gtfn
        or program_processor[0] == gtfn_cpu.run_gtfn_imperative
    ):
        pytest.xfail("gtfn backend not yet supported.")
    return program_processor


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


@dataclass
class DummyConnectivity:
    max_neighbors: int
    has_skip_values: int
    origin_axis: common.Dimension = common.Dimension("dummy_origin")
    neighbor_axis: common.Dimension = common.Dimension("dummy_neighbor")
    index_type: type[int] = int

    def mapped_index(_, __) -> int:
        return 0
