# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile-level tests of the pybind11 ``run_computation`` binding of the C++ backends."""

import re

import numpy as np
import pytest

from gt4py import cartesian as gt4pyc, storage as gt_storage
from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import PARALLEL, Field, computation, interval

from cartesian_tests.definitions import ALL_BACKENDS


# Backends that generate a pybind11 extension (the numpy and debug backends do not).
PYEXT_BACKENDS = [
    param
    for param in ALL_BACKENDS
    if gt4pyc.backend.from_name(param.values[0]).languages["computation"] in ("c++", "cuda")
]

SHAPE = (3, 4, 5)


# Field names `py`, `gt` and `std` coincide with the namespace aliases used inside the generated
# binding; they must stay legal stencil parameter names (a parameter cannot shadow a namespace in
# a qualified name).
def axpy_definition(
    py: Field["dtype"],  # type: ignore
    gt: Field["dtype"],  # type: ignore
    std: Field["dtype"],  # type: ignore
    out: Field["dtype"],  # type: ignore
    *,
    alpha: "dtype",  # type: ignore
):
    with computation(PARALLEL), interval(...):
        out = alpha * py + gt - std  # noqa: F841


def _pyext_module(stencil):
    """The pybind11 extension module the generated stencil module calls into."""
    return type(stencil).run.__globals__["pyext_module"]


def _as_numpy(array) -> np.ndarray:
    return np.asarray(array.get() if hasattr(array, "get") else array)


@pytest.mark.parametrize("backend", PYEXT_BACKENDS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_run_computation_binding(backend, dtype):
    stencil = gtscript.stencil(backend=backend, definition=axpy_definition, dtypes={"dtype": dtype})

    rng = np.random.default_rng(seed=1234)
    fields = {
        name: gt_storage.from_array(
            rng.random(SHAPE, dtype=dtype), backend=backend, aligned_index=(0, 0, 0), dtype=dtype
        )
        for name in ("py", "gt", "std")
    }
    out = gt_storage.zeros(SHAPE, backend=backend, aligned_index=(0, 0, 0), dtype=dtype)
    alpha = dtype(0.75)

    stencil(**fields, out=out, alpha=alpha)

    expected = alpha * _as_numpy(fields["py"]) + _as_numpy(fields["gt"]) - _as_numpy(fields["std"])
    np.testing.assert_allclose(_as_numpy(out), expected, rtol=8 * np.finfo(dtype).eps)

    # the extension's docstring names every parameter with its C++ type, in call order
    pybind_type = (
        "object" if gt4pyc.backend.from_name(backend).storage_info["device"] == "gpu" else "buffer"
    )
    ctype = {np.float32: "float", np.float64: "double"}[dtype]
    doc = _pyext_module(stencil).run_computation.__doc__
    for name in ("py", "gt", "std", "out"):
        assert f"py::{pybind_type} {name}, std::array<gt::int_t,3> {name}_origin" in doc
    assert f"{ctype} alpha" in doc
    assert (
        doc.index("domain")
        < doc.index(" py,")
        < doc.index(" out,")
        < doc.index("alpha")
        < doc.index("exec_info")
    )


@pytest.mark.parametrize("backend", PYEXT_BACKENDS)
def test_run_computation_rejects_wrong_arity_and_types(backend):
    stencil = gtscript.stencil(
        backend=backend, definition=axpy_definition, dtypes={"dtype": np.float64}
    )
    run_computation = _pyext_module(stencil).run_computation

    with pytest.raises(
        TypeError, match=r"run_computation\(\): expected (\d+) positional arguments, got 0"
    ) as info:
        run_computation()
    arity = int(re.search(r"expected (\d+)", str(info.value)).group(1))
    # domain + 4 fields x (buffer, origin) + alpha + exec_info
    assert arity == 1 + 4 * 2 + 1 + 1

    with pytest.raises(
        TypeError,
        match=r"run_computation\(\): positional argument 0 \('domain'\) cannot be converted from Python type 'NoneType'",
    ):
        run_computation(*([None] * arity))
