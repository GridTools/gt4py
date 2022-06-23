import pytest

from functional.fencil_processors import double_roundtrip, gtfn, lisp, roundtrip
from functional.fencil_processors.processor_interface import ProcessorType, fencil_formatter
from functional.iterator import ir as itir
from functional.iterator.pretty_parser import pparse
from functional.iterator.pretty_printer import pformat


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
        (gtfn.format_sourcecode, False),
        (pretty_format_and_check, False),
        (roundtrip.executor, True),
        (double_roundtrip.executor, True),
    ],
    ids=lambda p: f"backend={p[0].__module__.split('.')[-1] + '.' + p[0].__name__ if p[0] else p[0]}",
)
def fencil_processor(request):
    return request.param


def run_processor(fencil, processor, *args, **kwargs):
    if processor is None or processor.processor_type is ProcessorType.EXECUTOR:
        fencil(*args, backend=processor, **kwargs)
    elif processor.processor_type is ProcessorType.FORMATTER:
        print(fencil.string_format(*args, formatter=processor, **kwargs))
    else:
        raise TypeError(
            f"fencil processor type not recognized of {processor}: {getattr(processor, 'processor_type')}!"
        )
