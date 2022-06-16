import pytest

from functional.iterator.transforms import LiftMode


@pytest.fixture(
    params=[
        LiftMode.FORCE_INLINE,
        LiftMode.FORCE_TEMPORARIES,
        LiftMode.SIMPLE_HEURISTIC,
    ],
    ids=lambda p: f"lift_mode={p.name}",
)
def lift_mode(request):
    return request.param


@pytest.fixture(
    params=[
        # (backend, do_validate)
        (None, True),
        ("lisp", False),
        ("gtfn", False),
        ("type_check", False),
        ("pretty_print", False),
        ("type_check", False),
        ("roundtrip", True),
        ("double_roundtrip", True),
    ],
    ids=lambda p: f"backend={p[0]}",
)
def backend(request):
    return request.param
