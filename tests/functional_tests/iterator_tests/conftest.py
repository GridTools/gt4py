from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import functional.fencil_processors.formatters.gtfn
from functional.fencil_processors import type_check
from functional.fencil_processors.formatters import lisp
from functional.fencil_processors.runners import double_roundtrip, embedded, roundtrip
from functional.iterator import ir as itir
from functional.iterator.pretty_parser import pparse
from functional.iterator.pretty_printer import pformat
from functional.iterator.processor_interface import (
    EmbeddedFencilExecutor,
    FencilExecutor,
    FencilFormatter,
    fencil_formatter,
)


if TYPE_CHECKING:
    from functional.iterator.runtime import FendefDispatcher


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
        (functional.fencil_processors.formatters.gtfn.format_sourcecode, False),
        (pretty_format_and_check, False),
        (embedded.executor, True),
        (roundtrip.executor, True),
        (type_check.check, False),
        (double_roundtrip.executor, True),
    ],
    ids=lambda p: f"backend={p[0].__module__.split('.')[-1] + '.' + p[0].__name__ if p[0] else p[0]}",
)
def fencil_processor(request):
    return request.param


def run_processor(
    fencil: FendefDispatcher,
    processor: FencilExecutor | FencilFormatter | EmbeddedFencilExecutor,
    *args,
    **kwargs,
) -> None:
    if processor is None or isinstance(processor, (FencilExecutor, EmbeddedFencilExecutor)):
        fencil(*args, backend=processor, **kwargs)
    elif isinstance(processor, FencilFormatter):
        print(fencil.format_itir(*args, formatter=processor, **kwargs))
    else:
        raise TypeError(f"fencil processor kind not recognized: {processor}!")
