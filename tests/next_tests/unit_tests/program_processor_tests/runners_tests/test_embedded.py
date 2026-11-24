# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""What the embedded backends add to embedded execution; the integration matrix runs the features."""

import dataclasses
import re

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import astype, broadcast, common, int32, neighbor_sum

from next_tests import definitions


IDim = gtx.Dimension("IDim")
Vertex = gtx.Dimension("Vertex")
Edge = gtx.Dimension("Edge")
E2VDim = gtx.Dimension("E2V", kind=gtx.DimensionKind.LOCAL)
E2V = gtx.FieldOffset("E2V", source=Vertex, target=(Edge, E2VDim))

E2V_TABLE = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=np.int32)


@dataclasses.dataclass
class Pair:
    first: gtx.Field[[IDim], gtx.float64]
    second: gtx.Field[[IDim], gtx.float64]


@gtx.field_operator
def edge_sum(f: gtx.Field[[Vertex], gtx.float64]) -> gtx.Field[[Edge], gtx.float64]:
    return neighbor_sum(f(E2V), axis=E2VDim)


@gtx.program
def edge_sum_program(f: gtx.Field[[Vertex], gtx.float64], out: gtx.Field[[Edge], gtx.float64]):
    edge_sum(f, out=out)


@gtx.field_operator
def scale(f: gtx.Field[[IDim], gtx.float64], n: int32) -> gtx.Field[[IDim], gtx.float64]:
    return astype(n, gtx.float64) * f


@gtx.program
def scale_bounded(
    f: gtx.Field[[IDim], gtx.float64], out: gtx.Field[[IDim], gtx.float64], stop: int32
):
    scale(f, 2, out=out, domain={IDim: (0, stop)})


@gtx.field_operator
def scale_if(f: gtx.Field[[IDim], gtx.float64], flag: bool) -> gtx.Field[[IDim], gtx.float64]:
    if flag:
        return 2.0 * f
    return f


@gtx.field_operator
def split(f: gtx.Field[[IDim], gtx.float64]) -> Pair:
    return Pair(first=f, second=2.0 * f)


@gtx.field_operator
def plus_one(a: gtx.Field[[IDim], gtx.float64]) -> gtx.Field[[IDim], gtx.float64]:
    return a + 1.0


@gtx.field_operator
def identity(a: gtx.Field[[IDim], gtx.float64]) -> gtx.Field[[IDim], gtx.float64]:
    return a


@gtx.program
def write_then_read(
    a: gtx.Field[[IDim], gtx.float64],
    b: gtx.Field[[IDim], gtx.float64],
    out: gtx.Field[[IDim], gtx.float64],
):
    plus_one(a, out=a)
    identity(b, out=out)


@gtx.field_operator
def infinity() -> gtx.Field[[IDim], gtx.float64]:
    return broadcast(1.0 / 0.0, (IDim,))


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
    if request.param == "cpu":
        pytest.skip(definitions.TORCH_COMPILE_CPU_MESSAGE)
    return torch.device("cuda" if request.param == "gpu" else "cpu")


@pytest.fixture
def backend(library):
    if library == "jax":
        from gt4py.next.program_processors.runners.experimental import jax_jit

        return jax_jit.jax_jit
    from gt4py.next.program_processors.runners.experimental import torch_compile

    return torch_compile.torch_compile


@pytest.fixture
def no_recompiles(library):
    """Fail on a second trace of the same entry point."""
    if library == "jax":
        yield
        return
    import torch

    with torch._dynamo.config.patch(recompile_limit=1, fail_on_recompile_limit_hit=True):
        yield


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


def _jitted(entry_point):
    """The `jax.jit` object, under the call adapter of `embedded` and the write-back of `jax_jit`."""
    return entry_point._embedded_runner.__wrapped__.__wrapped__


def test_out_assigned_in_the_callers_field(device, backend):
    f = _field(device, [1.0, 2.0, 3.0, 4.0], [Vertex])
    f_buffer = f.ndarray
    out = _field(device, np.zeros(5), [Edge])
    out_object = out

    edge_sum_program.with_backend(backend)(f, out, offset_provider={"E2V": _e2v(device)})

    assert out is out_object and f.ndarray is f_buffer
    np.testing.assert_array_equal(out.asnumpy(), [3.0, 5.0, 7.0, 5.0, 4.0])


@pytest.mark.requires_jax
def test_table_is_a_runtime_argument():
    import jax

    device = jax.devices("cpu")[0]
    f = _field(device, [1.0, 2.0, 3.0, 4.0], [Vertex])
    out = _field(device, np.zeros(5), [Edge])
    from gt4py.next.program_processors.runners.experimental import jax_jit

    prog = edge_sum_program.with_backend(jax_jit.jax_jit)
    conn = _e2v(device)
    prog(f, out, offset_provider={"E2V": conn})

    stablehlo = _jitted(prog).lower(f, out, offset_provider={"E2V": conn}).as_text()
    assert "tensor<5x2xi32>" in re.search(r"@main\(([^)]*)\)", stablehlo).group(1)


@pytest.mark.requires_jax
def test_retrace_per_table_buffer():
    import jax

    from gt4py.next.program_processors.runners.experimental import jax_jit

    device = jax.devices("cpu")[0]
    out = _field(device, np.zeros(5), [Edge])
    prog = edge_sum_program.with_backend(jax_jit.jax_jit)
    conn = _e2v(device)

    prog(_field(device, [1.0, 2.0, 3.0, 4.0], [Vertex]), out, offset_provider={"E2V": conn})
    prog(_field(device, [4.0, 3.0, 2.0, 1.0], [Vertex]), out, offset_provider={"E2V": conn})
    assert _jitted(prog)._cache_size() == 1

    prog(
        _field(device, [4.0, 3.0, 2.0, 1.0], [Vertex]),
        out,
        offset_provider={"E2V": _e2v(device, E2V_TABLE.copy())},
    )
    assert _jitted(prog)._cache_size() == 2


def test_scalar_values_do_not_retrace(device, backend, no_recompiles):
    f = _field(device, np.arange(3.0), [IDim])
    fo = scale.with_backend(backend)
    for n in range(10):
        out = _field(device, np.zeros(3), [IDim])
        fo(f, int32(n), out=out, offset_provider={})
        np.testing.assert_array_equal(out.asnumpy(), n * np.arange(3.0))


def test_static_param_of_program(device, backend):
    f = _field(device, np.arange(5.0), [IDim])
    out = _field(device, np.full(5, -1.0), [IDim])

    scale_bounded.with_backend(backend).with_static_params("stop")(f, out, 3, offset_provider={})

    np.testing.assert_array_equal(out.asnumpy(), [0.0, 2.0, 4.0, -1.0, -1.0])


def test_static_param_of_field_operator(device, backend):
    f = _field(device, np.arange(3.0), [IDim])
    fo = scale_if.with_backend(backend).with_compilation_options(static_params=("flag",))
    for flag, factor in ((True, 2.0), (False, 1.0)):
        out = _field(device, np.zeros(3), [IDim])
        fo(f, flag, out=out, offset_provider={})
        np.testing.assert_array_equal(out.asnumpy(), factor * np.arange(3.0))


def test_domain_of_field_operator_writes_into_out(device, backend):
    f = _field(device, np.arange(5.0), [IDim])
    out = _field(device, np.full(5, -1.0), [IDim])

    scale.with_backend(backend)(f, int32(3), out=out, domain={IDim: (1, 4)}, offset_provider={})

    np.testing.assert_array_equal(out.asnumpy(), [-1.0, 3.0, 6.0, 9.0, -1.0])


def test_dataclass_only_as_out(device, backend):
    f = _field(device, np.arange(3.0), [IDim])
    out = Pair(
        first=_field(device, np.zeros(3), [IDim]), second=_field(device, np.zeros(3), [IDim])
    )

    split.with_backend(backend)(f, out=out, offset_provider={})

    np.testing.assert_array_equal(out.first.asnumpy(), np.arange(3.0))
    np.testing.assert_array_equal(out.second.asnumpy(), 2.0 * np.arange(3.0))


def test_aliases_see_the_assignment(device, backend):
    a = _field(device, np.arange(3.0), [IDim])
    out = _field(device, np.zeros(3), [IDim])
    expected_a = _field(device, np.arange(3.0), [IDim])
    expected = _field(device, np.zeros(3), [IDim])

    write_then_read.with_backend(backend)(a, a, out, offset_provider={})
    write_then_read(expected_a, expected_a, expected, offset_provider={})

    np.testing.assert_array_equal(out.asnumpy(), expected.asnumpy())
    np.testing.assert_array_equal(a.asnumpy(), expected_a.asnumpy())
    np.testing.assert_array_equal(out.asnumpy(), np.arange(3.0) + 1.0)


def test_error_of_the_embedded_code_is_raised(device, backend):
    out = _field(device, np.zeros(3), [IDim])

    with pytest.raises(ZeroDivisionError):
        infinity.with_backend(backend)(out=out, offset_provider={})


def test_missing_out_is_rejected(device, backend):
    with pytest.raises(gtx.errors.MissingArgumentError, match="'out'"):
        scale.with_backend(backend)(
            _field(device, np.arange(3.0), [IDim]), int32(1), offset_provider={}
        )


def test_backend_allocates_arrays_of_its_library(library, backend):
    field = gtx.zeros(gtx.domain({IDim: 3}), allocator=backend)

    if library == "jax":
        import jax

        assert isinstance(field.ndarray, jax.Array)
    else:
        import torch

        assert isinstance(field.ndarray, torch.Tensor)


@pytest.mark.requires_torch
@pytest.mark.requires_gpu
def test_torch_compile_allocates_on_the_given_device():
    from gt4py._core import definitions as core_defs
    from gt4py.next.program_processors.runners.experimental import torch_compile

    field = gtx.zeros(
        gtx.domain({IDim: 3}),
        allocator=torch_compile.torch_compile,
        device=core_defs.Device(core_defs.DeviceType.CUDA, 0),
    )

    assert field.ndarray.is_cuda


@pytest.mark.requires_jax
def test_jax_backends_allocate_and_run_on_the_given_device():
    import jax

    from gt4py._core import definitions as core_defs
    from gt4py.next import experimental

    cpu = jax.local_devices(backend="cpu")[0]
    domain = gtx.domain({IDim: 3})
    device = core_defs.Device(core_defs.DeviceType.CPU, 0)
    f = gtx.as_field(domain, np.arange(3.0), allocator=experimental.embedded_jax, device=device)
    out = gtx.zeros(domain, allocator=experimental.jax_jit, device=device)

    plus_one.with_backend(experimental.jax_jit)(f, out=out, offset_provider={})

    assert f.ndarray.devices() == out.ndarray.devices() == {cpu}
    np.testing.assert_array_equal(out.asnumpy(), np.arange(3.0) + 1.0)


@pytest.fixture(
    params=[
        "numpy",
        pytest.param("cupy", marks=pytest.mark.requires_gpu),
        pytest.param("jax", marks=pytest.mark.requires_jax),
        pytest.param("torch", marks=pytest.mark.requires_torch),
    ]
)
def eager_backend(request):
    from gt4py.next import experimental

    return getattr(experimental, f"embedded_{request.param}")


def test_eager_backend_runs_programs(eager_backend):
    domain = gtx.domain({IDim: 3})
    a = gtx.as_field(domain, np.arange(3.0), allocator=eager_backend)
    b = gtx.as_field(domain, np.zeros(3), allocator=eager_backend)
    out = gtx.zeros(domain, allocator=eager_backend)

    write_then_read.with_backend(eager_backend)(a, b, out, offset_provider={})

    np.testing.assert_array_equal(a.asnumpy(), np.arange(3.0) + 1.0)
    np.testing.assert_array_equal(out.asnumpy(), np.zeros(3))


def test_eager_backend_runs_field_operators(eager_backend):
    domain = gtx.domain({IDim: 3})
    f = gtx.as_field(domain, np.arange(3.0), allocator=eager_backend)
    out = gtx.zeros(domain, allocator=eager_backend)

    plus_one.with_backend(eager_backend)(f, out=out, domain={IDim: (1, 3)}, offset_provider={})

    np.testing.assert_array_equal(out.asnumpy(), [0.0, 2.0, 3.0])


@pytest.fixture(
    params=[
        pytest.param("embedded_jax", marks=pytest.mark.requires_jax),
        pytest.param("embedded_torch", marks=pytest.mark.requires_torch),
        pytest.param("jax_jit", marks=pytest.mark.requires_jax),
        pytest.param("torch_compile", marks=pytest.mark.requires_torch),
    ]
)
def non_numpy_backend(request):
    from gt4py.next import experimental

    return getattr(experimental, request.param)


def test_fields_of_another_library_are_rejected(non_numpy_backend):
    domain = gtx.domain({IDim: 3})
    f = gtx.as_field(domain, np.arange(3.0), allocator=non_numpy_backend)
    numpy_out = gtx.zeros(domain, allocator=np)

    with pytest.raises(TypeError, match="numpy.ndarray"):
        plus_one.with_backend(non_numpy_backend)(f, out=numpy_out, offset_provider={})


def test_connectivities_of_another_library_are_rejected(non_numpy_backend):
    f = gtx.as_field([Vertex], np.arange(4.0), allocator=non_numpy_backend)
    out = gtx.zeros(gtx.domain({Edge: 5}), allocator=non_numpy_backend)
    numpy_e2v = common._connectivity(
        E2V_TABLE, codomain=Vertex, domain=common.domain({Edge: 5, E2VDim: 2})
    )

    with pytest.raises(TypeError, match="numpy.ndarray"):
        edge_sum_program.with_backend(non_numpy_backend)(f, out, offset_provider={"E2V": numpy_e2v})


@gtx.field_operator
def first(p: Pair) -> gtx.Field[[IDim], gtx.float64]:
    return p.first


def test_arguments_are_bound_as_by_the_entry_point(eager_backend):
    domain = gtx.domain({IDim: 3})
    f = gtx.as_field(domain, np.arange(3.0), allocator=eager_backend)
    out = gtx.zeros(domain, allocator=eager_backend)
    fo = first.with_backend(eager_backend)

    with pytest.raises(TypeError):
        fo(Pair(first=f, second=f), f, out=out, offset_provider={})
    with pytest.raises(TypeError):
        fo(Pair(first=f, second=f), p=Pair(first=out, second=out), out=out, offset_provider={})


def test_unknown_static_params_are_rejected(eager_backend):
    fo = scale.with_backend(eager_backend).with_compilation_options(static_params=("m",))
    f = gtx.as_field(gtx.domain({IDim: 3}), np.arange(3.0), allocator=eager_backend)

    with pytest.raises(ValueError, match="'m'"):
        fo(f, int32(1), out=f, offset_provider={})


def test_exported_from_experimental(library, backend):
    from gt4py.next import experimental

    assert getattr(experimental, {"jax": "jax_jit", "torch": "torch_compile"}[library]) is backend


def test_eager_backend_is_created_once(eager_backend):
    from gt4py.next import backend as next_backend, experimental

    assert isinstance(eager_backend, next_backend.EmbeddedBackend)
    assert getattr(experimental, eager_backend.name) is eager_backend


def test_missing_optional_package_is_reported(monkeypatch):
    import sys

    from gt4py.next import experimental

    monkeypatch.delitem(vars(experimental), "embedded_cupy", raising=False)
    monkeypatch.setitem(sys.modules, "cupy", None)

    with pytest.raises(AttributeError, match="optional 'cupy' package"):
        experimental.embedded_cupy


@pytest.mark.parametrize(
    "name",
    [
        "embedded_numpy",
        pytest.param("embedded_cupy", marks=pytest.mark.requires_gpu),
        pytest.param("embedded_jax", marks=pytest.mark.requires_jax),
        pytest.param("embedded_torch", marks=pytest.mark.requires_torch),
        pytest.param("jax_jit", marks=pytest.mark.requires_jax),
        pytest.param("torch_compile", marks=pytest.mark.requires_torch),
    ],
)
def test_compile_is_rejected(name):
    from gt4py.next import experimental

    with pytest.raises(RuntimeError, match="does not compile"):
        edge_sum_program.with_backend(getattr(experimental, name)).compile(
            offset_provider={"E2V": None}
        )
