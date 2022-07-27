import pytest

import functional.fencil_processors.formatters.gtfn
from functional.fencil_processors import type_check
from functional.fencil_processors.formatters import lisp
from functional.fencil_processors.runners import double_roundtrip, gtfn_cpu, roundtrip
from functional.iterator import ir as itir
from functional.iterator.pretty_parser import pparse
from functional.iterator.pretty_printer import pformat
from functional.iterator.processor_interface import (
    FencilExecutor,
    FencilFormatter,
    fencil_formatter,
)


@pytest.fixture(params=[False, True], ids=lambda p: f"use_tmps={p}")
def use_tmps(request):
    return request.param


@fencil_formatter
def pretty_format_and_check(root: itir.FencilDefinition, *args, **kwargs) -> str:
    pretty = pformat(root)
    parsed = pparse(pretty)
    assert parsed == root
    return pretty


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
        (functional.fencil_processors.formatters.gtfn.format_sourcecode, False),
    ],
    ids=lambda p: f"backend={p[0].__module__.split('.')[-1] + '.' + p[0].__name__ if p[0] else p[0]}",
)
def fencil_processor(request):
    return request.param


@pytest.fixture
def fencil_processor_no_gtfn_exec(fencil_processor):
    if fencil_processor[0] == gtfn_cpu.run_gtfn:
        pytest.xfail("gtfn backend not yet supported.")
    return fencil_processor


def run_processor(fencil, processor, *args, **kwargs):
    if processor is None or isinstance(processor, FencilExecutor):
        fencil(*args, backend=processor, **kwargs)
    elif isinstance(processor, FencilFormatter):
        print(fencil.format_itir(*args, formatter=processor, **kwargs))
    else:
        raise TypeError(f"fencil processor kind not recognized: {processor}!")
