# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import re

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import astype, common, int32, maximum, neighbor_sum, sqrt
from gt4py.next.ffront.experimental import as_offset


IDim = gtx.Dimension("IDim")
KDim = gtx.Dimension("KDim", kind=gtx.DimensionKind.VERTICAL)
Vertex = gtx.Dimension("Vertex")
Edge = gtx.Dimension("Edge")
E2VDim = gtx.Dimension("E2V", kind=gtx.DimensionKind.LOCAL)
E2V = gtx.FieldOffset("E2V", source=Vertex, target=(Edge, E2VDim))
Ioff = gtx.FieldOffset("Ioff", source=IDim, target=(IDim,))

E2V_TABLE = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=np.int32)


@gtx.field_operator
def edge_sum(f: gtx.Field[[Vertex], gtx.float64]) -> gtx.Field[[Edge], gtx.float64]:
    return neighbor_sum(f(E2V), axis=E2VDim)


@gtx.program
def edge_sum_program(f: gtx.Field[[Vertex], gtx.float64], out: gtx.Field[[Edge], gtx.float64]):
    edge_sum(f, out=out)


@gtx.field_operator
def shift_scale(
    f: gtx.Field[[IDim], gtx.float64], a: gtx.float64
) -> gtx.Field[[IDim], gtx.float64]:
    return a * f(IDim + 1)


@gtx.program
def shift_scale_program(
    f: gtx.Field[[IDim], gtx.float64], a: gtx.float64, out: gtx.Field[[IDim], gtx.float64]
):
    shift_scale(f, a, out=out)


@gtx.program
def shift_scale_bounded(
    f: gtx.Field[[IDim], gtx.float64],
    a: gtx.float64,
    out: gtx.Field[[IDim], gtx.float64],
    stop: int32,
):
    shift_scale(f, a, out=out, domain={IDim: (0, stop)})


@gtx.field_operator
def dynamic_shift(
    f: gtx.Field[[IDim], gtx.float64], offsets: gtx.Field[[IDim], int32]
) -> gtx.Field[[IDim], gtx.float64]:
    return f(as_offset(Ioff, offsets))


@gtx.field_operator
def scalar_builtins(
    f: gtx.Field[[IDim], gtx.float64], a: gtx.float64, n: int32
) -> gtx.Field[[IDim], gtx.float64]:
    return (sqrt(a) + maximum(a, 5.0) + astype(n, gtx.float64)) * f


@gtx.scan_operator(axis=KDim, forward=True, init=0.0)
def scaled_cumsum(state: gtx.float64, x: gtx.float64, a: gtx.float64) -> gtx.float64:
    return state + a * x


@gtx.field_operator
def scale_if(f: gtx.Field[[IDim], gtx.float64], flag: bool) -> gtx.Field[[IDim], gtx.float64]:
    if flag:
        return 2.0 * f
    return f


@gtx.field_operator
def swap(
    a: gtx.Field[[IDim], gtx.float64], b: gtx.Field[[IDim], gtx.float64]
) -> tuple[gtx.Field[[IDim], gtx.float64], gtx.Field[[IDim], gtx.float64]]:
    return b, a


@pytest.fixture(
    params=[
        pytest.param("jax", marks=pytest.mark.requires_jax),
        pytest.param("torch", marks=pytest.mark.requires_torch),
    ]
)
def library(request):
    return request.param


@pytest.fixture(params=["cpu", pytest.param("gpu", marks=pytest.mark.requires_gpu)])
def device(request, library):
    if library == "jax":
        import jax

        try:
            return jax.devices(request.param)[0]
        except RuntimeError:
            pytest.skip(f"no jax '{request.param}' device")
    import torch

    if request.param == "gpu" and not torch.cuda.is_available():
        pytest.skip("no torch CUDA device")
    return torch.device("cuda" if request.param == "gpu" else "cpu")


@pytest.fixture
def backend(library):
    if library == "jax":
        from gt4py.next.program_processors.runners import jax_jit

        return jax_jit.jax_jit
    from gt4py.next.program_processors.runners import torch_compile

    return torch_compile.torch_compile


@pytest.fixture
def jax_only(library):
    if library != "jax":
        pytest.skip("inspects the 'jax.jit' cache")


def _array(device, values):
    values = np.asarray(values)
    if type(device).__module__.startswith("torch"):
        import torch

        return torch.as_tensor(values, device=device)
    import jax

    return jax.device_put(values, device)


def _field(device, values, dims):
    values = np.asarray(values)
    return common._field(
        _array(device, values),
        domain=common.domain({dim: size for dim, size in zip(dims, values.shape)}),
    )


def _e2v(device, table=E2V_TABLE):
    return common._connectivity(
        _array(device, table),
        codomain=Vertex,
        domain=common.domain({Edge: table.shape[0], E2VDim: table.shape[1]}),
    )


def _runner(entry_point):
    return entry_point._jit_entry.jitted


def test_program_with_connectivity(device, backend):
    f = _field(device, [1.0, 2.0, 3.0, 4.0], [Vertex])
    out = _field(device, np.zeros(5), [Edge])
    expected = _field(device, np.zeros(5), [Edge])
    conn = _e2v(device)
    out_object = out

    edge_sum_program.with_backend(backend)(f, out, offset_provider={"E2V": conn})
    edge_sum_program(f, expected, offset_provider={"E2V": conn})

    assert out is out_object
    np.testing.assert_array_equal(out.asnumpy(), expected.asnumpy())
    np.testing.assert_array_equal(out.asnumpy(), [3.0, 5.0, 7.0, 5.0, 4.0])


def test_table_is_a_runtime_argument(device, backend, jax_only):
    f = _field(device, [1.0, 2.0, 3.0, 4.0], [Vertex])
    out = _field(device, np.zeros(5), [Edge])
    prog = edge_sum_program.with_backend(backend)
    conn = _e2v(device)
    prog(f, out, offset_provider={"E2V": conn})

    stablehlo = _runner(prog).lower(f, out, offset_provider={"E2V": conn}).as_text()
    main_signature = re.search(r"@main\(([^)]*)\)", stablehlo).group(1)
    assert "tensor<5x2xi32>" in main_signature


def test_retrace_per_table_buffer(device, backend, jax_only):
    f = _field(device, [1.0, 2.0, 3.0, 4.0], [Vertex])
    out = _field(device, np.zeros(5), [Edge])
    prog = edge_sum_program.with_backend(backend)
    conn = _e2v(device)

    prog(f, out, offset_provider={"E2V": conn})
    prog(_field(device, [4.0, 3.0, 2.0, 1.0], [Vertex]), out, offset_provider={"E2V": conn})
    assert _runner(prog)._cache_size() == 1
    np.testing.assert_array_equal(out.asnumpy(), [7.0, 5.0, 3.0, 5.0, 6.0])

    prog(f, out, offset_provider={"E2V": _e2v(device, E2V_TABLE.copy())})
    assert _runner(prog)._cache_size() == 2


def test_traced_scalar_does_not_retrace(device, backend, jax_only):
    f = _field(device, np.arange(6.0), [IDim])
    prog = shift_scale_program.with_backend(backend)
    for a in (2.0, 3.0):
        prog(f, a, _field(device, np.zeros(5), [IDim]), offset_provider={})

    assert _runner(prog)._cache_size() == 1


def test_cartesian_offset_and_scalar(device, backend):
    f = _field(device, np.arange(6.0), [IDim])
    prog = shift_scale_program.with_backend(backend)
    for a in (2.0, 3.0):
        out = _field(device, np.zeros(5), [IDim])
        expected = _field(device, np.zeros(5), [IDim])
        prog(f, a, out, offset_provider={})
        shift_scale_program(f, a, expected, offset_provider={})
        np.testing.assert_array_equal(out.asnumpy(), expected.asnumpy())


def test_static_param_in_program_domain(device, backend):
    f = _field(device, np.arange(6.0), [IDim])
    prog = shift_scale_bounded.with_backend(backend).with_static_params("stop")
    out = _field(device, np.full(5, -1.0), [IDim])
    expected = _field(device, np.full(5, -1.0), [IDim])

    prog(f, 2.0, out, 3, offset_provider={})
    shift_scale_bounded(f, 2.0, expected, 3, offset_provider={})

    np.testing.assert_array_equal(out.asnumpy(), expected.asnumpy())
    np.testing.assert_array_equal(out.asnumpy(), [2.0, 4.0, 6.0, -1.0, -1.0])


def test_field_operator_out_and_domain(device, backend):
    f = _field(device, np.arange(6.0), [IDim])
    out = _field(device, np.full(5, -1.0), [IDim])
    expected = _field(device, np.full(5, -1.0), [IDim])

    shift_scale.with_backend(backend)(f, 2.0, out=out, offset_provider={}, domain={IDim: (1, 4)})
    shift_scale(f, 2.0, out=expected, offset_provider={}, domain={IDim: (1, 4)})

    np.testing.assert_array_equal(out.asnumpy(), expected.asnumpy())
    np.testing.assert_array_equal(out.asnumpy(), [-1.0, 4.0, 6.0, 8.0, -1.0])


def test_dimension_in_offset_provider_is_static(device, backend):
    f = _field(device, np.arange(6.0), [IDim])
    offsets = _field(device, np.asarray([1, 0, 2, 1, 0], dtype=np.int32), [IDim])
    out = _field(device, np.zeros(5), [IDim])
    expected = _field(device, np.zeros(5), [IDim])

    dynamic_shift.with_backend(backend)(f, offsets, out=out, offset_provider={"Ioff": IDim})
    dynamic_shift(f, offsets, out=expected, offset_provider={"Ioff": IDim})

    np.testing.assert_array_equal(out.asnumpy(), expected.asnumpy())
    np.testing.assert_array_equal(out.asnumpy(), [1.0, 1.0, 4.0, 4.0, 4.0])


def test_scalar_only_builtins(device, backend):
    f = _field(device, np.arange(3.0), [IDim])
    out = _field(device, np.zeros(3), [IDim])
    expected = _field(device, np.zeros(3), [IDim])

    scalar_builtins.with_backend(backend)(f, 4.0, int32(2), out=out, offset_provider={})
    scalar_builtins(f, 4.0, int32(2), out=expected, offset_provider={})

    np.testing.assert_array_equal(out.asnumpy(), expected.asnumpy())
    np.testing.assert_array_equal(out.asnumpy(), [0.0, 9.0, 18.0])


def test_many_int_values_do_not_retrace(device, backend):
    f = _field(device, np.arange(3.0), [IDim])
    fo = scalar_builtins.with_backend(backend)
    for n in range(10):
        out = _field(device, np.zeros(3), [IDim])
        fo(f, 4.0, int32(n), out=out, offset_provider={})
        np.testing.assert_array_equal(out.asnumpy(), (7.0 + n) * np.arange(3.0))


def test_scan_with_scalar(device, backend):
    x = _field(device, np.arange(1.0, 5.0), [KDim])
    out = _field(device, np.zeros(4), [KDim])
    expected = _field(device, np.zeros(4), [KDim])

    scaled_cumsum.with_backend(backend)(x, 2.0, out=out, offset_provider={})
    scaled_cumsum(x, 2.0, out=expected, offset_provider={})

    np.testing.assert_array_equal(out.asnumpy(), expected.asnumpy())
    np.testing.assert_array_equal(out.asnumpy(), [2.0, 6.0, 12.0, 20.0])


def test_static_param_of_field_operator(device, backend):
    f = _field(device, np.arange(3.0), [IDim])
    fo = scale_if.with_backend(backend).with_compilation_options(static_params=("flag",))
    for flag in (True, False):
        out = _field(device, np.zeros(3), [IDim])
        expected = _field(device, np.zeros(3), [IDim])
        fo(f, flag, out=out, offset_provider={})
        scale_if(f, flag, out=expected, offset_provider={})
        np.testing.assert_array_equal(out.asnumpy(), expected.asnumpy())


def test_field_operator_tuple_out_leaves_inputs(device, backend):
    a = _field(device, np.arange(3.0), [IDim])
    b = _field(device, np.arange(3.0) + 10, [IDim])
    a_buffer, b_buffer = a.ndarray, b.ndarray
    out = (_field(device, np.zeros(3), [IDim]), _field(device, np.zeros(3), [IDim]))

    swap.with_backend(backend)(a, b, out=out, offset_provider={})

    assert a.ndarray is a_buffer and b.ndarray is b_buffer
    np.testing.assert_array_equal(out[0].asnumpy(), b.asnumpy())
    np.testing.assert_array_equal(out[1].asnumpy(), a.asnumpy())


def test_compile_is_rejected(backend):
    with pytest.raises(RuntimeError, match="does not compile"):
        edge_sum_program.with_backend(backend).compile(offset_provider={"E2V": None})
