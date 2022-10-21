import numpy as np
import pytest

from gt4py import backend as gt_backend
from gt4py import gtscript
from gt4py import storage as gt_storage

from ..definitions import ALL_BACKENDS, PERFORMANCE_BACKENDS
from .stencil_definitions import copy_stencil


try:
    import cupy as cp
except ImportError:
    cp = None


def _get_array_library(backend: str):
    if gt_backend.from_name(backend).storage_info["device"] == "gpu":
        assert cp is not None
        return cp
    else:
        return np


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("order", ["C", "F"])
def test_numpy_allocators(backend, order):
    xp = _get_array_library(backend)
    shape = (20, 10, 5)
    inp = xp.array(xp.random.randn(*shape), order=order, dtype=xp.float_)
    outp = xp.zeros(shape=shape, order=order, dtype=xp.float_)

    stencil = gtscript.stencil(definition=copy_stencil, backend=backend)
    stencil(field_a=inp, field_b=outp)

    cp.testing.assert_array_equal(outp, inp)


@pytest.mark.parametrize("backend", PERFORMANCE_BACKENDS)
def test_bad_layout_warns(backend):
    xp = _get_array_library(backend)
    backend_type = gt_backend.from_name(backend)

    shape = (10, 10, 10)

    inp = xp.array(xp.random.randn(*shape), dtype=xp.float_)
    outp = gt_storage.zeros(backend=backend, shape=shape, dtype=xp.float_, aligned_index=(0, 0, 0))

    # set up non-optimal storage layout:
    if backend_type.storage_info["is_optimal_layout"](inp, "IJK"):
        # permute in a circular manner
        inp = xp.transpose(inp, axes=(1, 2, 0))

    stencil = gtscript.stencil(definition=copy_stencil, backend=backend)

    with pytest.warns(
        UserWarning,
        match="The layout of the field 'field_a' is not recommended for this backend."
        "This may lead to performance degradation. Please consider using the"
        "provided allocators in `gt4py.storage`.",
    ):
        stencil(field_a=inp, field_b=outp)
