import pytest

from functional.iterator.backends import double_roundtrip, gtfn, lisp, pretty_print, roundtrip


@pytest.fixture(params=[False, True], ids=lambda p: f"use_tmps={p}")
def use_tmps(request):
    return request.param


@pytest.fixture(
    params=[
        # (backend, do_validate)
        (None, True),
        (lisp.print_sourcecode, False),
        (gtfn.print_sourcecode, False),
        (pretty_print.pretty_print_and_check, False),
        (roundtrip.executor, True),
        (double_roundtrip.executor, True),
    ],
    ids=lambda p: f"backend={p[0].__module__.split('.')[-1] + '.' + p[0].__name__ if p[0] else p[0]}",
)
def backend(request):
    return request.param
