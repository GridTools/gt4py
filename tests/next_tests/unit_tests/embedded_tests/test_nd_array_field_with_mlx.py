# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Embedded fields of MLX arrays.

The tests run on MLX's CPU device, as the session does (see `tests/conftest.py`), except
for the ones marked `requires_mlx_metal`, which switch to the GPU. Tests of the GPU's error
paths also run without one: they make MLX report the GPU as its default device, and never
compute on it.
"""

import numpy as np
import pytest

from gt4py import next as gtx
from gt4py._core import definitions as core_defs, ndarray_utils as core_ndarray_utils
from gt4py.next import common
from gt4py.next.common import Dimension, DimensionKind
from gt4py.next.embedded import array_ops, context as embedded_context
from gt4py.next.ffront import fbuiltins
from gt4py.next.ffront.fbuiltins import neighbor_sum, sin


pytestmark = pytest.mark.requires_mlx

try:
    import mlx.core as mx

    from gt4py.next.embedded.array_libraries import _mlx_namespace as xp
except ImportError:  # the tests are skipped by `requires_mlx`
    mx = xp = None

I = Dimension("I")
V = Dimension("V")
E = Dimension("E")
E2VDim = Dimension("E2V", kind=DimensionKind.LOCAL)
E2V = gtx.FieldOffset("E2V", source=V, target=(E, E2VDim))


def _field(values, dtype=np.float64, dim=I):
    data = np.asarray(values, dtype=dtype)
    return gtx.as_field([dim], data, allocator=mx)


@pytest.fixture
def gpu_default(monkeypatch):
    """MLX as on Apple silicon, whose default device is the GPU; nothing may compute on it."""
    monkeypatch.setattr(mx, "default_device", lambda: mx.Device(mx.gpu))


@pytest.fixture
def evaluated(monkeypatch):
    """The arrays `mx.eval` is called on."""
    arrays = []
    eval_ = mx.eval

    def record(*args):
        arrays.extend(args)
        return eval_(*args)

    monkeypatch.setattr(mx, "eval", record)
    return arrays


# -- the namespace around `mlx.core` --


def test_namespace_registered_for_mlx_arrays():
    ops = array_ops.of(mx.zeros((2,)))

    assert ops.xp is xp
    assert ops.array_ns is mx
    assert ops.device(mx.zeros((2,))) == mx.default_device()


@pytest.mark.parametrize(
    "data, expected_dtype",
    [
        (np.arange(3.0), np.float64),
        (np.float64(0.1), np.float64),
        ([0.1, 0.2], np.float64),
        (0.1, np.float64),
        (3, np.int64),
        (True, np.bool_),
        (np.arange(3, dtype=np.int32), np.int32),
    ],
)
def test_asarray_keeps_numpy_dtypes(data, expected_dtype):
    # `mlx.core.asarray` gives float32 and int32 for all of them
    result = xp.asarray(data)

    assert np.array(result).dtype == expected_dtype
    np.testing.assert_array_equal(np.array(result), np.asarray(data, dtype=expected_dtype))


def test_asarray_converts_and_copies_numpy_data():
    # with Metal, `mlx.core.asarray` shares the memory of NumPy data and reinterprets its bytes
    data = np.arange(6).reshape(2, 3)

    result = xp.asarray(data, dtype=mx.float64)
    same_dtype = xp.asarray(data)
    data[...] = -1

    np.testing.assert_array_equal(np.array(result), np.arange(6.0).reshape(2, 3))
    np.testing.assert_array_equal(np.array(same_dtype), np.arange(6).reshape(2, 3))


def test_creation_functions_default_to_numpy_dtypes():
    assert xp.empty((2,)).dtype == mx.float64
    assert xp.zeros((2,)).dtype == mx.float64
    assert xp.ones((2,)).dtype == mx.float64
    assert xp.full((2,), 0.1).dtype == mx.float64
    assert xp.full((2,), 1).dtype == mx.int64
    assert xp.arange(3).dtype == mx.int64
    assert xp.arange(0.0, 1.0, 0.25).dtype == mx.float64
    np.testing.assert_array_equal(np.array(xp.full((2,), 0.1)), np.full((2,), 0.1))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_cbrt(dtype):
    x = np.array([-27.0, -2.0, 0.0, 0.5, 8.0, 1e6], dtype=dtype)

    result = np.array(xp.cbrt(mx.array(x, dtype=getattr(mx, np.dtype(dtype).name))))

    assert result.dtype == dtype
    np.testing.assert_allclose(result, np.cbrt(x), rtol=1e-6 if dtype == np.float32 else 1e-14)


def test_fmod():
    x1 = np.array([5.5, -5.5, 5.5, -5.5, 6.0, -6.0, 0.0])
    x2 = np.array([2.0, 2.0, -2.0, -2.0, 3.0, 3.0, 2.0])

    result = np.array(xp.fmod(mx.array(x1, dtype=mx.float64), mx.array(x2, dtype=mx.float64)))

    np.testing.assert_array_equal(result, np.fmod(x1, x2))


def test_sum_with_dtype():
    x = mx.array([100, 100, 100], dtype=mx.int8)

    result = xp.sum(x, axis=0, dtype=mx.int8)

    # MLX sums int8 in int32; the requested dtype wraps around as in NumPy
    assert result.dtype == mx.int8
    assert np.array(result) == np.sum(np.array([100, 100, 100], dtype=np.int8), dtype=np.int8)


def test_nonzero():
    x = np.array([[0, 1, 0], [2, 0, 3]])

    result = xp.nonzero(mx.array(x))

    assert len(result) == 2
    for indices, expected in zip(result, np.nonzero(x)):
        assert isinstance(indices, mx.array)
        np.testing.assert_array_equal(np.array(indices), expected)


def test_float64_on_gpu_raises(gpu_default):
    for create in (
        lambda: xp.asarray(np.arange(3.0)),
        lambda: xp.asarray(0.5),
        lambda: xp.zeros((2,)),
        lambda: xp.full((2,), 1.0),
        lambda: xp.arange(0.0, 1.0),
        lambda: xp.astype(mx.zeros((2,), dtype=mx.float32), mx.float64),
    ):
        with pytest.raises(TypeError, match="float64.*dtype=float32.*set_default_device"):
            create()


def test_float64_on_cpu_stream_is_allowed(gpu_default):
    # an explicit CPU device overrides the default device
    assert xp.zeros((2,), dtype=mx.float64, device=mx.cpu).dtype == mx.float64


# -- fields --


def test_no_silent_downcast():
    field = _field([1.0, 2.0])

    for result in (
        field + np.float64(0.1),
        field * 0.1,
        fbuiltins.where(_field([True, False], dtype=np.bool_), field, np.float64(0.1)),
        fbuiltins.astype(_field([1, 2], dtype=np.int32), np.float64),
    ):
        assert result.dtype == core_defs.dtype(np.float64)
        assert result.ndarray.dtype == mx.float64
    np.testing.assert_array_equal((field * 0.1).asnumpy(), np.array([1.0, 2.0]) * 0.1)


_UNARY_MATH = [
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "arcsinh",
    "arctanh", "exp", "log", "sqrt", "cbrt", "floor", "ceil", "trunc", "abs",
]  # fmt: skip
_BINARY_MATH = ["minimum", "maximum", "fmod", "power"]


@pytest.mark.parametrize("name", _UNARY_MATH + _BINARY_MATH)
def test_float64_math_values(name):
    # float64 values, not only the dtype: MLX computes `sin`, `cos` and `exp` with float32
    # accuracy, and the namespace computes them with NumPy instead
    rng = np.random.default_rng(0)
    args = [rng.uniform(0.05, 0.95, 64)] + (
        [rng.uniform(0.5, 4.0, 64)] if name in _BINARY_MATH else []
    )

    result = getattr(fbuiltins, name)(*(_field(arg) for arg in args))

    assert result.ndarray.dtype == mx.float64
    np.testing.assert_allclose(result.asnumpy(), getattr(np, name)(*args), rtol=1e-12, atol=0)


def test_float64_values_of_python_floats():
    field = _field([0.0, 0.0])
    mask = _field([True, False], dtype=np.bool_)

    field[...] = 0.1

    assert field.asnumpy()[0] == 0.1
    assert fbuiltins.where(mask, field, 0.3).asnumpy()[1] == 0.3
    assert (field * 0.3).asnumpy()[0] == 0.1 * 0.3


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_integer_division_in_float64(dtype):
    # as in NumPy; MLX divides integers in float32
    a, b = np.array([2**24 + 1, 1, 7], dtype=dtype), np.array([1, 3, 2], dtype=dtype)

    result = _field(a, dtype=dtype) / _field(b, dtype=dtype)

    assert result.ndarray.dtype == mx.float64
    np.testing.assert_array_equal(result.asnumpy(), a / b)


def test_integer_division_on_gpu_raises(gpu_default):
    # the DSL types `int32 / int32` as int32, so the error names the division, not only float64
    a = gtx.as_field([I], mx.array([1, 2], dtype=mx.int32))

    with pytest.raises(TypeError, match="cannot divide integers on the GPU.*'//'.*'float32'"):
        a / a
    assert (a // a).ndarray.dtype == mx.int32


def test_cbrt_values():
    x = np.array([-27.0, -0.0, 0.0, 8.0, 1e-300, 64.0**3])

    result = np.array(xp.cbrt(mx.array(x, dtype=mx.float64)))

    np.testing.assert_array_equal(result, np.cbrt(x))  # exact cubes, and the sign of -0.0
    assert np.signbit(result[1])
    integer_result = xp.cbrt(mx.array([27, -8], dtype=mx.int32))
    assert integer_result.dtype == mx.float64
    np.testing.assert_array_equal(np.array(integer_result), np.cbrt(np.array([27.0, -8.0])))


def test_integer_cbrt_on_gpu_raises(gpu_default):
    with pytest.raises(TypeError, match="float64"):
        xp.cbrt(mx.array([27], dtype=mx.int32))


def test_mixed_dtypes_promote_as_in_mlx():
    # unlike NumPy, MLX promotes like JAX; the DSL rejects mixed dtypes, field arithmetic does not
    assert (
        _field([1, 2], dtype=np.int64) + _field([1.0, 2.0], dtype=np.float32)
    ).ndarray.dtype == (mx.float32)
    assert (_field([1, 2], dtype=np.int32) * 0.5).ndarray.dtype == mx.float32


def test_float32_field_stays_float32():
    field = _field([1.0, 2.0], dtype=np.float32)

    assert (field * 0.5 + field).ndarray.dtype == mx.float32
    assert (field + np.float32(0.5)).ndarray.dtype == mx.float32


def test_astype_to_float64_on_gpu_raises(gpu_default):
    field = gtx.as_field([I], mx.zeros((2,), dtype=mx.float32))

    with pytest.raises(TypeError, match="float64"):
        fbuiltins.astype(field, np.float64)


def test_setitem(evaluated):
    field = _field([0.0, 1.0, 2.0, 3.0])
    buffer = field.ndarray

    field[common.domain({I: (1, 3)})] = gtx.as_field(
        common.domain({I: (1, 3)}), np.array([10.0, 20.0]), allocator=mx
    )
    field[common.domain({I: (3, 4)})] = np.float64(30.0)

    # MLX writes into the same array object, and computes the write right away
    assert field.ndarray is buffer
    assert any(array is buffer for array in evaluated)
    np.testing.assert_array_equal(field.asnumpy(), [0.0, 10.0, 20.0, 30.0])


def test_setitem_casts_to_field_dtype():
    field = _field([0.0, 1.0], dtype=np.float32)

    field[...] = np.float64(0.5)

    assert field.ndarray.dtype == mx.float32
    np.testing.assert_array_equal(field.asnumpy(), [0.5, 0.5])


def test_setitem_of_float64_scalar_into_float32_field_on_gpu(gpu_default):
    # the NumPy scalar is converted to the dtype of the field, never to a float64 array
    field = gtx.as_field([I], mx.zeros((2,), dtype=mx.float32))

    field[...] = np.float64(0.5)

    assert field.ndarray.dtype == mx.float32
    np.testing.assert_array_equal(field.asnumpy(), [0.5, 0.5])


def test_asnumpy_is_a_copy():
    field = _field([1.0, 2.0])

    values = field.asnumpy()
    field[...] = 5.0
    values_after = field.asnumpy()
    values_after[0] = -1.0

    # a NumPy view of the memory of an MLX array would change with writes into the field
    np.testing.assert_array_equal(values, [1.0, 2.0])
    np.testing.assert_array_equal(field.asnumpy(), [5.0, 5.0])


def test_as_scalar():
    field = gtx.as_field(common.Domain(), mx.array(2.5, dtype=mx.float64))

    result = field.as_scalar()

    assert isinstance(result, np.float64)
    assert result == 2.5


def test_buffer_info_unsupported():
    with pytest.raises(NotImplementedError, match="only in embedded"):
        _field([1.0]).__gt_buffer_info__


def test_as_field_keeps_library_of_results():
    result = _field([1.0, 2.0]) + _field([3.0, 4.0])

    field = gtx.as_field([I], result.ndarray)

    assert isinstance(field.ndarray, mx.array)
    assert array_ops.of(field.ndarray).xp is xp


# -- connectivities --


def _e2v(table, skip_value=None):
    table = np.asarray(table, dtype=np.int32)
    return gtx.as_connectivity(
        [E, E2VDim], V, mx.array(table, dtype=mx.int32), skip_value=skip_value
    )


def test_premap_with_skip_values():
    values = np.array([1.0, 10.0, 100.0, 1000.0])
    table = np.array([[0, 1], [2, -1], [-1, 3]])
    v = _field(values, dim=V)
    e2v = _e2v(table, skip_value=-1)

    with embedded_context.update(offset_provider={"E2V": e2v}):
        result = fbuiltins.neighbor_sum(v.premap(e2v), axis=E2VDim)

    assert isinstance(result.ndarray, mx.array)
    np.testing.assert_array_equal(
        result.asnumpy(), np.where(table != -1, values[table], 0.0).sum(axis=1)
    )


def test_connectivity_domain_inference():
    # the inverse image of a range of the codomain uses `nonzero` on the table
    e2v = gtx.as_connectivity([E], V, mx.array([5, 6, 7, 8, 9], dtype=mx.int32))

    restricted = _field(np.arange(10.0), dim=V)[common.domain({V: (6, 9)})].premap(e2v)

    assert restricted.domain == common.domain({E: (1, 4)})
    np.testing.assert_array_equal(restricted.asnumpy(), [6.0, 7.0, 8.0])


# -- lazy evaluation --


@gtx.field_operator
def _double(a: gtx.Field[[I], gtx.float64]) -> gtx.Field[[I], gtx.float64]:
    return a + a


@gtx.field_operator
def _step(a: gtx.Field[[I], gtx.float64]) -> gtx.Field[[I], gtx.float64]:
    return _double(a) * a


def test_out_is_evaluated(evaluated):
    a = _field([1.0, 2.0])
    out = _field([0.0, 0.0])

    _step(a, out=out, offset_provider={})

    assert any(array is out.ndarray for array in evaluated)
    np.testing.assert_array_equal(out.asnumpy(), [2.0, 8.0])


def test_results_of_outermost_call_are_evaluated(evaluated):
    state = _field([1.0, 2.0])

    with embedded_context.update(offset_provider={}):
        for _ in range(3):
            state = _step(state)
            # each result is computed, so a loop does not build up a graph over all steps;
            # the result of the nested call of `_double` is not
            assert len(evaluated) == 1 and evaluated.pop() is state.ndarray

    np.testing.assert_array_equal(state.asnumpy(), [128.0, 32768.0])


# -- on the GPU --


@pytest.fixture
def on_gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


@gtx.field_operator
def _axpy_f32(
    a: gtx.Field[[I], gtx.float32], x: gtx.Field[[I], gtx.float32], y: gtx.Field[[I], gtx.float32]
) -> gtx.Field[[I], gtx.float32]:
    return a * sin(x) + y


@gtx.field_operator
def _neighbor_sum_f32(v: gtx.Field[[V], gtx.float32]) -> gtx.Field[[E], gtx.float32]:
    return neighbor_sum(v(E2V), axis=E2VDim)


def _run_float32_field_operator():
    rng = np.random.default_rng(0)
    a, x, y = (rng.random(16, dtype=np.float32) for _ in range(3))
    fields = [gtx.as_field([I], v, allocator=mx) for v in (a, x, y)]
    out = gtx.zeros({I: 16}, dtype=np.float32, allocator=mx)

    _axpy_f32(*fields, out=out, offset_provider={})

    assert out.ndarray.dtype == mx.float32
    np.testing.assert_allclose(out.asnumpy(), a * np.sin(x) + y, rtol=1e-5)


def _run_float32_neighbor_sum(skip_value):
    values = np.random.default_rng(0).random(5, dtype=np.float32)
    table = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], dtype=np.int32)
    if skip_value is not None:
        table[1, 1] = table[3, 0] = skip_value
    e2v = _e2v(table, skip_value=skip_value)
    v = gtx.as_field([V], values, allocator=mx)
    out = gtx.zeros({E: 5}, dtype=np.float32, allocator=mx)

    _neighbor_sum_f32(v, out=out, offset_provider={"E2V": e2v})

    assert out.ndarray.dtype == mx.float32
    expected = np.where(table != -1, values[table], np.float32(0)).sum(axis=1)
    np.testing.assert_allclose(out.asnumpy(), expected, rtol=1e-5)


def test_float32_field_operator_requests_no_float64(gpu_default):
    # computes on the CPU, but any float64 request in embedded execution raises as on the GPU
    _run_float32_field_operator()


@pytest.mark.parametrize("skip_value", [None, -1])
def test_float32_neighbor_sum_requests_no_float64(gpu_default, skip_value):
    _run_float32_neighbor_sum(skip_value)


@pytest.mark.requires_mlx_metal
class TestMetal:
    @pytest.fixture(autouse=True)
    def _gpu(self, on_gpu):
        pass

    def test_float32_field_operator(self):
        _run_float32_field_operator()

    @pytest.mark.parametrize("skip_value", [None, -1])
    def test_float32_neighbor_sum(self, skip_value):
        _run_float32_neighbor_sum(skip_value)

    def test_float64_raises(self):
        with pytest.raises(TypeError, match="float64.*set_default_device"):
            gtx.zeros({I: 3}, allocator=mx)
        with pytest.raises(TypeError, match="float64"):
            gtx.as_field([I], np.arange(3.0), allocator=mx)

    def test_device_translation(self):
        translator = core_ndarray_utils.get_device_translator(mx)

        assert translator(core_defs.Device(core_defs.DeviceType.METAL, 0)) == mx.gpu
        with pytest.raises(ValueError, match="default device"):
            translator(core_defs.Device(core_defs.DeviceType.CPU, 0))
        field = gtx.zeros(
            {I: 3},
            dtype=np.float32,
            allocator=mx,
            device=core_defs.Device(core_defs.DeviceType.METAL, 0),
        )
        assert array_ops.of(field.ndarray).device(field.ndarray) == mx.gpu

    def test_dlpack_device_is_metal(self):
        array = mx.zeros((3,), dtype=mx.float32)
        mx.eval(array)

        assert core_ndarray_utils.device_from_dlpack(*array.__dlpack_device__()) == (
            core_defs.Device(core_defs.DeviceType.METAL, 0)
        )

    def test_as_field_keeps_mlx_array(self):
        # a Metal array has no default allocator of its device
        field = gtx.as_field([I], mx.zeros((3,), dtype=mx.float32))

        assert isinstance(field.ndarray, mx.array)
        np.testing.assert_array_equal(field.asnumpy(), np.zeros(3, dtype=np.float32))
