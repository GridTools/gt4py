from __future__ import annotations

import pytest

from functional.fencil_processors import processor_interface as fpi, type_check
from functional.fencil_processors.formatters import gtfn, lisp
from functional.fencil_processors.runners import double_roundtrip, gtfn_cpu, roundtrip
from functional.iterator import ir as itir, pretty_parser, pretty_printer, runtime, transforms


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


@fpi.fencil_formatter
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
        (gtfn.format_sourcecode, False),
    ],
    ids=lambda p: get_processor_id(p[0]),
)
def fencil_processor(request):
    return request.param


@pytest.fixture
def fencil_processor_no_gtfn_exec(fencil_processor):
    if fencil_processor[0] == gtfn_cpu.run_gtfn:
        pytest.xfail("gtfn backend not yet supported.")
    return fencil_processor


def run_processor(
    fencil: runtime.FendefDispatcher,
    processor: fpi.FencilExecutor | fpi.FencilFormatter,
    *args,
    **kwargs,
) -> None:
    if processor is None or fpi.is_processor_kind(processor, fpi.FencilExecutor):
        fencil(*args, backend=processor, **kwargs)
    elif fpi.is_processor_kind(processor, fpi.FencilFormatter):
        print(fencil.format_itir(*args, formatter=processor, **kwargs))
    else:
        raise TypeError(f"fencil processor kind not recognized: {processor}!")
