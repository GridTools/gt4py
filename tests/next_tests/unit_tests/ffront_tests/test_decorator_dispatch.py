# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for how programs and operators dispatch on the kind of their backend."""

import dataclasses

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import backend as next_backend
from gt4py.next.ffront import decorator
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


@gtx.field_operator
def _copy_op(a: gtx.Field[[IDim], gtx.float64]) -> gtx.Field[[IDim], gtx.float64]:
    return a


def _copy_program_def(a: gtx.Field[[IDim], gtx.float64], out: gtx.Field[[IDim], gtx.float64]):
    _copy_op(a, out=out)


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

    copy, copy_program, _ = _make_testees(gtx.embedded_numpy)
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


_NONE_MESSAGE = "'backend=None' is not supported.*'gtx.embedded_numpy'"


def test_none_backend_is_rejected():
    with pytest.raises(TypeError, match=_NONE_MESSAGE):
        gtx.program(backend=None)(_copy_program_def)
    with pytest.raises(TypeError, match=_NONE_MESSAGE):
        gtx.field_operator(_copy_def, backend=None)
    with pytest.raises(TypeError, match=_NONE_MESSAGE):
        gtx.scan_operator(axis=KDim, forward=True, init=0.0, backend=None)(_sum_def)
    with pytest.raises(TypeError, match=_NONE_MESSAGE):
        decorator.Program.from_function(_copy_program_def, backend=None)
    with pytest.raises(TypeError, match=_NONE_MESSAGE):
        decorator.FieldOperator.from_function(_copy_def, backend=None)

    copy, copy_program, scan = _make_testees(gtx.embedded_numpy)
    for testee in (copy, copy_program, scan):
        with pytest.raises(TypeError, match=_NONE_MESSAGE):
            testee.with_backend(None)
        with pytest.raises(TypeError, match=_NONE_MESSAGE):
            dataclasses.replace(testee, backend=None)


def test_default_backend_is_embedded_numpy():
    assert decorator.DEFAULT_BACKEND is gtx.embedded_numpy

    copy = gtx.field_operator(_copy_def)
    copy_program = gtx.program(_copy_program_def)
    scan = gtx.scan_operator(axis=KDim, forward=True, init=0.0)(_sum_def)
    assert copy.backend is gtx.embedded_numpy
    assert copy_program.backend is gtx.embedded_numpy
    assert scan.backend is gtx.embedded_numpy

    a = gtx.as_field([IDim], np.arange(3.0))
    out = gtx.zeros({IDim: 3})
    with pytest.warns(UserWarning, match="Using Python execution"):
        copy_program(a, out, offset_provider={})
    assert np.allclose(out.asnumpy(), a.asnumpy())
    assert gtx.zeros({IDim: 3}, allocator=copy_program).ndarray.flags["C_CONTIGUOUS"]


def test_embedded_backend_runs_embedded():
    copy, copy_program, scan = _make_testees(gtx.embedded_numpy)
    a = gtx.as_field([IDim], np.arange(3.0))
    out = gtx.zeros({IDim: 3})
    k_field = gtx.as_field([KDim], np.ones(3))
    k_out = gtx.zeros({KDim: 3})

    with pytest.warns(UserWarning, match="Using Python execution"):
        copy_program(a, out, offset_provider={})
    assert np.allclose(out.asnumpy(), a.asnumpy())

    out = gtx.zeros({IDim: 3})
    copy(a, out=out, offset_provider={})
    assert np.allclose(out.asnumpy(), a.asnumpy())

    scan(k_field, out=k_out, offset_provider={})
    assert np.allclose(k_out.asnumpy(), [1.0, 2.0, 3.0])

    with pytest.raises(RuntimeError, match="'embedded_numpy' is an embedded backend"):
        copy_program.compile(offset_provider={})
    with pytest.raises(RuntimeError, match="'embedded_numpy' is an embedded backend"):
        copy.compile(offset_provider={})


@pytest.mark.parametrize(
    "backend",
    [gtx.embedded_numpy, next_backend.EmbeddedBackend(name="other_numpy")],
    ids=lambda backend: backend.name,
)
def test_embedded_backend_allocator(backend):
    _, copy_program, _ = _make_testees(backend)

    assert copy_program.__gt_allocator__ is np
    field = gtx.zeros({IDim: 3}, allocator=copy_program)
    assert isinstance(field.ndarray, np.ndarray)
    field = gtx.ones({IDim: 3}, allocator=backend)
    assert isinstance(field.ndarray, np.ndarray)


def test_compiled_backend_allocator_and_toolchain():
    _, copy_program, _ = _make_testees(roundtrip.default)

    assert copy_program.__gt_allocator__ is roundtrip.default.allocator

    copy_program.compile(offset_provider={})
    toolchain = copy_program._compiled_programs.toolchain
    assert toolchain.backend is roundtrip.default
    assert toolchain.frontend is next_backend.DEFAULT_TRANSFORMS
    assert copy_program._compiled_programs.backend is roundtrip.default
