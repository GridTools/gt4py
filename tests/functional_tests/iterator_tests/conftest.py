import pytest


@pytest.fixture(params=[False, True], ids=lambda p: f"use_tmps={p}")
def use_tmps(request):
    return request.param


@pytest.fixture(
    params=[
        # (backend, do_validate)
        (None, True),
        ("lisp", False),
        ("cpptoy", False),
        ("pretty_print", False),
        ("roundtrip", True),
        ("double_roundtrip", True),
    ],
    ids=lambda p: f"backend={p[0]}",
)
def backend(request):
    return request.param
