# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for how programs and operators dispatch on the kind of their backend."""

import dataclasses
import warnings

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import backend as next_backend
from gt4py.next.program_processors.runners import roundtrip


IDim = gtx.Dimension("IDim")
KDim = gtx.Dimension("KDim", kind=gtx.DimensionKind.VERTICAL)


@dataclasses.dataclass(frozen=True)
class _OtherBackend(next_backend.Backend):
    """A backend that is neither compiled nor embedded."""


_other_backend = _OtherBackend(name="other_backend", allocator=None)


def _copy_def(a: gtx.Field[[IDim], gtx.float64]) -> gtx.Field[[IDim], gtx.float64]:
    return a


def _sum_def(state: float, a: float) -> float:
    return state + a


def _make_testees(backend):
    @gtx.field_operator(backend=backend)
    def copy(a: gtx.Field[[IDim], gtx.float64]) -> gtx.Field[[IDim], gtx.float64]:
        return a

    @gtx.program(backend=backend)
    def copy_program(a: gtx.Field[[IDim], gtx.float64], out: gtx.Field[[IDim], gtx.float64]):
        copy(a, out=out)

    @gtx.scan_operator(axis=KDim, forward=True, init=0.0, backend=backend)
    def scan(state: float, a: float) -> float:
        return state + a

    return copy, copy_program, scan


@pytest.mark.parametrize("not_a_backend", [object(), "roundtrip", roundtrip.default.compiler])
def test_construction_rejects_non_backends(not_a_backend):
    with pytest.raises(TypeError, match="Expected a 'gt4py.next.backend.Backend'"):
        gtx.field_operator(_copy_def, backend=not_a_backend)
    with pytest.raises(TypeError, match="Expected a 'gt4py.next.backend.Backend'"):
        gtx.scan_operator(axis=KDim, forward=True, init=0.0, backend=not_a_backend)(_sum_def)

    copy, copy_program, _ = _make_testees(None)
    with pytest.raises(TypeError, match="Expected a 'gt4py.next.backend.Backend'"):
        copy.with_backend(not_a_backend)
    with pytest.raises(TypeError, match="Expected a 'gt4py.next.backend.Backend'"):
        copy_program.with_backend(not_a_backend)


def test_unsupported_backend_kind_raises_on_use():
    # Creating programs and operators with such a backend works ...
    copy, copy_program, scan = _make_testees(_other_backend)
    a = gtx.as_field([IDim], np.ones(3))
    out = gtx.zeros({IDim: 3})
    k_field = gtx.as_field([KDim], np.ones(3))

    # ... but using them does not.
    with pytest.raises(TypeError, match="other_backend"):
        copy_program(a, out, offset_provider={})
    with pytest.raises(TypeError, match="other_backend"):
        copy(a, out=out, offset_provider={})
    with pytest.raises(TypeError, match="other_backend"):
        scan(k_field, out=k_field, offset_provider={})
    with pytest.raises(TypeError, match="other_backend"):
        copy_program.compile(offset_provider={})
    with pytest.raises(TypeError, match="other_backend"):
        gtx.zeros({IDim: 3}, allocator=copy_program)


def test_none_backend_runs_embedded():
    copy, copy_program, _ = _make_testees(None)
    a = gtx.as_field([IDim], np.arange(3.0))
    out = gtx.zeros({IDim: 3})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # "Using Python execution" warning
        copy_program(a, out, offset_provider={})
    assert np.allclose(out.asnumpy(), a.asnumpy())

    with pytest.raises(RuntimeError, match="without backend"):
        copy_program.compile(offset_provider={})


def test_compiled_backend_allocator_and_toolchain():
    _, copy_program, _ = _make_testees(roundtrip.default)

    assert copy_program.__gt_allocator__ is roundtrip.default.allocator

    copy_program.compile(offset_provider={})
    toolchain = copy_program._compiled_programs.toolchain
    assert toolchain.backend is roundtrip.default
    assert toolchain.frontend is next_backend.DEFAULT_FRONTEND
    assert copy_program._compiled_programs.backend is roundtrip.default
