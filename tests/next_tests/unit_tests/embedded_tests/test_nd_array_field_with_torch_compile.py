# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import NamedTuple

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import common
from gt4py.next.embedded import nd_array_field
from gt4py.next import astype, neighbor_sum
from gt4py.next.experimental import as_offset


pytestmark = pytest.mark.requires_torch

K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)
Vertex = gtx.Dimension("Vertex")
Edge = gtx.Dimension("Edge")
E2VDim = gtx.Dimension("E2V", kind=gtx.DimensionKind.LOCAL)
E2V = gtx.FieldOffset("E2V", source=Vertex, target=(Edge, E2VDim))
Koff = gtx.FieldOffset("Koff", source=K, target=(K,))


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("cuda", marks=(pytest.mark.requires_gpu, pytest.mark.requires_torch_cuda)),
    ]
)
def device(request):
    import torch

    torch._dynamo.reset()
    return torch.device(request.param)


def _compile(fn, backend="aot_eager"):
    """`fn` compiled with `fullgraph`, and the Dynamo graphs of its compilations."""
    import torch

    graphs = []
    compiler = torch._dynamo.lookup_backend(backend)

    def recorder(gm, example_inputs):
        graphs.append(gm)
        # a backend that only runs the Dynamo graph would run the non-strictly traced
        # functions eagerly, on the concrete tensors
        return compiler(gm, example_inputs)

    return torch.compile(fn, fullgraph=True, backend=recorder), graphs


def _field(dims, values, device):
    import torch

    tensor = torch.asarray(values, device=device)
    return common._field(tensor, domain=gtx.domain(dict(zip(dims, tensor.shape))))


def _connectivity(table, device, **kwargs):
    import torch

    tensor = torch.asarray(table, device=device)
    return common._connectivity(
        tensor, codomain=Vertex, domain=gtx.domain({Edge: tensor.shape[0], E2VDim: 2}), **kwargs
    )


def _with_out(field_operator):
    def step(v, e2v, out):
        field_operator(v, out=out, offset_provider={"E2V": e2v})
        return out

    return step


def _skip_value_e2v_table(num_vertices: int) -> np.ndarray:
    """Two edges per vertex, both entries equal, every seventh second entry a skip value."""
    vertex = np.arange(4 * num_vertices) // 2
    table = np.stack([vertex, vertex], axis=1).astype(np.int32)
    table[::7, 1] = -1
    return table


def _reference_neighbor_sum(values: np.ndarray, table: np.ndarray) -> np.ndarray:
    return np.where(table != -1, values[table], 0.0).sum(axis=1)


@gtx.field_operator
def _scale_shift(a: gtx.Field[gtx.Dims[K], float], n: gtx.int32) -> gtx.Field[gtx.Dims[K], float]:
    return a(K + 1) * 2.0 + astype(n, float)


@gtx.program
def _scale_shift_program(
    a: gtx.Field[gtx.Dims[K], float], n: gtx.int32, out: gtx.Field[gtx.Dims[K], float]
):
    _scale_shift(a, n, out=out, domain={K: (0, 3)})


def test_torch_compile_program_and_field_operator(device):
    a = _field([K], np.arange(4.0), device)
    out, out2, expected, expected2 = (_field([K], np.zeros(4), device) for _ in range(4))

    def step(a, out, out2):
        _scale_shift_program(a, 3, out, offset_provider={})
        _scale_shift(out, 1, out=out2, offset_provider={}, domain={K: (0, 2)})
        return out2

    step(a, expected, expected2)
    compiled, graphs = _compile(step)
    result = compiled(a, out, out2)

    assert len(graphs) == 1
    assert result.ndarray.device.type == device.type
    np.testing.assert_array_equal(out.asnumpy(), expected.asnumpy())
    np.testing.assert_array_equal(result.asnumpy(), expected2.asnumpy())


class _Pair(NamedTuple):
    a: gtx.Field[gtx.Dims[K], float]
    b: gtx.Field[gtx.Dims[K], float]


@gtx.field_operator
def _pair_sum(
    p: tuple[gtx.Field[gtx.Dims[K], float], gtx.Field[gtx.Dims[K], float]],
) -> gtx.Field[gtx.Dims[K], float]:
    return p[0] + p[1]


def test_torch_compile_namedtuple_argument(device):
    a = _field([K], np.arange(3.0), device)
    b = _field([K], np.arange(3.0) * 10.0, device)
    out = _field([K], np.zeros(3), device)

    def step(a, b, out):
        _pair_sum(_Pair(a, b), out=out, offset_provider={})
        return out

    compiled, graphs = _compile(step)

    np.testing.assert_array_equal(compiled(a, b, out).asnumpy(), np.arange(3.0) * 11.0)
    assert len(graphs) == 1


@gtx.field_operator
def _neighbor_sum(v: gtx.Field[gtx.Dims[Vertex], float]) -> gtx.Field[gtx.Dims[Edge], float]:
    return neighbor_sum(v(E2V), axis=E2VDim)


@gtx.field_operator
def _first_neighbor(v: gtx.Field[gtx.Dims[Vertex], float]) -> gtx.Field[gtx.Dims[Edge], float]:
    return v(E2V[0])


@gtx.program
def _neighbor_sum_program(
    v: gtx.Field[gtx.Dims[Vertex], float], out: gtx.Field[gtx.Dims[Edge], float]
):
    _neighbor_sum(v, out=out)


@pytest.mark.parametrize("num_field_vertices", [300, 600], ids=["narrowing", "covering"])
def test_torch_compile_neighbor_sum_with_connectivity_argument(device, num_field_vertices):
    import torch

    table = _skip_value_e2v_table(300)
    num_edges = 2 * num_field_vertices
    values = np.arange(float(num_field_vertices))
    e2v = _connectivity(table, device, skip_value=-1)
    v = _field([Vertex], values, device)

    out, expected = (_field([Edge], np.zeros(num_edges), device) for _ in range(2))

    def neighbor_sum_of(v, e2v, out):
        _neighbor_sum(v, out=out, offset_provider={"E2V": e2v})
        return out

    neighbor_sum_of(v, e2v, expected)
    compiled, graphs = _compile(neighbor_sum_of)
    result = compiled(v, e2v, out)

    np.testing.assert_allclose(result.asnumpy(), expected.asnumpy())
    np.testing.assert_allclose(result.asnumpy(), _reference_neighbor_sum(values, table[:num_edges]))
    (graph,) = graphs
    # the table is a graph input, not a constant
    input_shapes = [
        tuple(node.meta["example_value"].shape) for node in graph.graph.find_nodes(op="placeholder")
    ]
    assert table.shape in input_shapes
    constants = [getattr(graph, node.target) for node in graph.graph.find_nodes(op="get_attr")]
    assert not any(isinstance(c, torch.Tensor) for c in constants)


def test_torch_compile_program_with_closed_over_offset_provider(device):
    import torch

    table = _skip_value_e2v_table(2)
    with device:
        e2v = gtx.as_connectivity([Edge, E2VDim], Vertex, table, skip_value=-1, allocator=torch)
    v = _field([Vertex], np.arange(4.0), device)
    out = _field([Edge], np.zeros(table.shape[0]), device)
    offset_provider = {"E2V": e2v}

    def step(v, out):
        _neighbor_sum_program(v, out, offset_provider=offset_provider)
        return out

    compiled, graphs = _compile(step)

    np.testing.assert_allclose(
        compiled(v, out).asnumpy(), _reference_neighbor_sum(np.arange(4.0), table)
    )
    assert len(graphs) == 1


def test_torch_compile_premap_with_restricted_connectivity(device):
    table = _skip_value_e2v_table(8)
    e2v = _connectivity(table, device, skip_value=-1)
    v = _field([Vertex], np.arange(4.0), device)

    out = _field([Edge], np.zeros(8), device)

    compiled, graphs = _compile(_with_out(_first_neighbor))
    result = compiled(v, e2v, out)

    np.testing.assert_array_equal(result.asnumpy(), np.arange(4.0)[table[:8, 0]])
    assert len(graphs) == 1


def test_torch_compile_retraces_per_connectivity_buffer(device):
    import torch

    tensor = torch.asarray(_skip_value_e2v_table(8), device=device)
    v = _field([Vertex], np.arange(8.0), device)

    def connectivity(tensor):
        return common._connectivity(
            tensor,
            codomain=Vertex,
            domain=gtx.domain({Edge: tensor.shape[0], E2VDim: 2}),
            skip_value=-1,
        )

    out = _field([Edge], np.zeros(16), device)

    compiled, graphs = _compile(_with_out(_neighbor_sum))
    e2v = connectivity(tensor)
    compiled(v, e2v, out)
    compiled(v, e2v, out)
    assert len(graphs) == 1

    compiled(v, connectivity(tensor), out)
    assert len(graphs) == 1

    table = _skip_value_e2v_table(8)
    table[0, 0] = 7
    np.testing.assert_allclose(
        compiled(v, connectivity(torch.asarray(table, device=device)), out).asnumpy(),
        _reference_neighbor_sum(np.arange(8.0), table[:16]),
    )
    assert len(graphs) == 2


@pytest.mark.parametrize("create", ["restrict", "from_array"])
def test_torch_compile_connectivity_not_created_by_constructor_raises(device, create):
    table = _skip_value_e2v_table(4)
    e2v = _connectivity(table, device, skip_value=-1)
    v = _field([Vertex], np.arange(4.0), device)
    if create == "restrict":
        e2v = e2v[gtx.domain({Edge: (0, 4), E2VDim: (0, 2)})]
    else:
        e2v = nd_array_field.NdArrayConnectivityField.from_array(
            e2v.ndarray[:4], Vertex, domain=gtx.domain({Edge: 4, E2VDim: 2}), skip_value=-1
        )
    out = _field([Edge], np.zeros(4), device)

    compiled, _ = _compile(_with_out(_neighbor_sum))

    with pytest.raises(Exception) as excinfo:
        compiled(v, e2v, out)
    assert "gtx.as_connectivity" in _chain(excinfo.value)


def _chain(error: BaseException) -> str:
    """The messages of `error` and its causes: `torch.compile` reports the error of a trace further down."""
    messages = []
    while error is not None:
        messages.append(str(error))
        error = error.__cause__ or error.__context__
    return "\n".join(messages)


def test_torch_compile_non_contiguous_inverse_image_raises(device):
    import torch

    e2v = _connectivity(torch.asarray([[0, 0], [5, 5], [1, 1]], dtype=torch.int32), device)
    v = _field([Vertex], np.arange(3.0), device)
    out = _field([Edge], np.zeros(3), device)

    compiled, _ = _compile(_with_out(_first_neighbor))

    with pytest.raises(Exception) as excinfo:
        compiled(v, e2v, out)
    assert "non-contiguous" in _chain(excinfo.value)


@gtx.field_operator
def _shift_by_field(
    a: gtx.Field[gtx.Dims[Edge, K], float], offset: gtx.Field[gtx.Dims[Edge, K], gtx.int32]
) -> gtx.Field[gtx.Dims[Edge, K], float]:
    return a(as_offset(Koff, offset))


def test_torch_compile_as_offset_computed_in_trace(device):
    values = np.arange(8.0).reshape(2, 4)
    offsets = np.asarray([[1, 1, 0, -1], [0, 2, 1, 0]], dtype=np.int32)
    a = _field([Edge, K], values, device)

    out = _field([Edge, K], np.zeros((2, 4)), device)

    def step(a, offset, out):
        _shift_by_field(a, offset, out=out, offset_provider={})
        return out

    compiled, graphs = _compile(step)
    result = compiled(a, _field([Edge, K], offsets, device), out)

    assert result.domain == a.domain
    np.testing.assert_array_equal(
        result.asnumpy(), np.take_along_axis(values, np.arange(4) + offsets, axis=1)
    )
    assert len(graphs) == 1

    # out-of-range targets read some element of the field under a trace, without an error
    for shift in (3, -3):
        result = compiled(a, _field([Edge, K], np.full((2, 4), shift, dtype=np.int32), device), out)
        targets = np.arange(4) + shift
        in_range = (targets >= 0) & (targets < 4)
        np.testing.assert_array_equal(result.asnumpy()[:, in_range], values[:, targets[in_range]])
        assert np.isin(result.asnumpy(), values).all()
    assert len(graphs) == 1


def test_torch_compile_grad_through_neighbor_sum_with_connectivity_argument(device):
    import torch

    table = np.asarray([[0, 1], [2, -1]], dtype=np.int32)
    e2v = _connectivity(table, device, skip_value=-1)
    x = torch.tensor([4.0, 9.0, 16.0, 1.0], dtype=torch.float64, device=device, requires_grad=True)

    v = common._field(x, domain=gtx.domain({Vertex: 4}))

    out = _field([Edge], np.zeros(2), device)

    def loss(v, e2v, out):
        _neighbor_sum(v, out=out, offset_provider={"E2V": e2v})
        return torch.sum(out.ndarray**2)

    compiled, graphs = _compile(loss)
    result = compiled(v, e2v, out)
    result.backward()

    np.testing.assert_allclose(result.item(), 13.0**2 + 16.0**2)
    np.testing.assert_allclose(x.grad.cpu().numpy(), [26.0, 26.0, 32.0, 0.0])
    assert len(graphs) == 1


def test_torch_compile_neighbor_sum_with_inductor(device):
    table = _skip_value_e2v_table(4)
    values = np.arange(8.0)
    e2v = _connectivity(table, device, skip_value=-1)
    v = _field([Vertex], values, device)
    out = _field([Edge], np.zeros(table.shape[0]), device)

    def step(v, e2v, out):
        _neighbor_sum_program(v, out, offset_provider={"E2V": e2v})
        return out

    compiled, graphs = _compile(step, backend="inductor")

    np.testing.assert_allclose(
        compiled(v, e2v, out).asnumpy(), _reference_neighbor_sum(values, table)
    )
    assert len(graphs) == 1
