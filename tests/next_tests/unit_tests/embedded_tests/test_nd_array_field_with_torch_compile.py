# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from gt4py.next import common
from gt4py.next.common import Dimension, DimensionKind, NamedIndex
from gt4py.next.embedded import context as embedded_context, nd_array_field
from gt4py.next.ffront import experimental, fbuiltins


pytestmark = pytest.mark.requires_torch

D0 = Dimension("D0")
D1 = Dimension("D1")
V = Dimension("V")
E = Dimension("E")
E2VDim = Dimension("E2V", kind=DimensionKind.LOCAL)


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


def _compile(fn, backend="aot_eager", **kwargs):
    """
    `fn` called through `nonstrict_trace` in a region compiled with `fullgraph`, and the Dynamo graphs.

    Dynamo does not trace embedded code, so a region calls it non-strictly.
    """
    import torch

    graphs = []
    compiler = torch._dynamo.lookup_backend(backend)

    def recorder(gm, example_inputs):
        graphs.append(gm)
        # a backend that only runs the Dynamo graph would run the non-strictly traced
        # function eagerly, on the concrete tensors
        return compiler(gm, example_inputs)

    traced = torch.compiler.nonstrict_trace(fn)
    return (
        torch.compile(lambda *args: traced(*args), fullgraph=True, backend=recorder, **kwargs),
        graphs,
    )


def _field(values, domain, device):
    import torch

    return common._field(torch.asarray(values, device=device), domain=domain)


def _connectivity(table, device, *, domain=None, **kwargs):
    import torch

    tensor = torch.asarray(table, device=device)
    domain = domain or common.domain({E: tensor.shape[0], E2VDim: tensor.shape[1]})
    return common._connectivity(tensor, codomain=V, domain=domain, **kwargs)


def _skip_value_e2v_table(num_vertices: int) -> np.ndarray:
    """Two edges per vertex, both entries equal, every seventh second entry a skip value."""
    vertex = np.arange(4 * num_vertices) // 2
    table = np.stack([vertex, vertex], axis=1).astype(np.int32)
    table[::7, 1] = -1
    return table


def _premap_and_reduce(field, conn):
    with embedded_context.update(offset_provider={"E2V": conn}):
        return fbuiltins.neighbor_sum(field.premap(conn), axis=E2VDim)


def test_torch_compile_field_arguments(device):
    domain = common.domain({D0: (1, 3), D1: (2, 5)})
    a = _field(np.arange(6.0).reshape(2, 3), domain, device)
    b = _field(np.ones((2, 3)), domain, device)

    compiled, graphs = _compile(lambda x, y: x + y)
    result = compiled(a, b)

    assert isinstance(result, common.Field)
    assert result.domain == domain
    assert result.ndarray.device.type == device.type
    np.testing.assert_allclose(result.asnumpy(), a.asnumpy() + b.asnumpy())
    assert len(graphs) == 1


def test_torch_compile_retraces_per_field_domain(device):
    compiled, graphs = _compile(lambda x: x + x, dynamic=False)

    for num_graphs, (start, stop) in enumerate([(0, 3), (5, 8), (0, 4)], start=1):
        field = _field(np.arange(float(stop - start)), common.domain({D0: (start, stop)}), device)
        result = compiled(field)

        assert result.domain == field.domain
        np.testing.assert_array_equal(result.asnumpy(), 2.0 * field.asnumpy())
        assert len(graphs) == num_graphs


@pytest.mark.xfail(
    reason="Dynamo makes the bounds of the second domain symbolic, which cannot be in a pytree context",
    raises=Exception,
    strict=True,
)
def test_torch_compile_another_field_domain_without_dynamic_false(device):
    compiled, _ = _compile(lambda x: x + x)

    compiled(_field(np.arange(3.0), common.domain({D0: (0, 3)}), device))
    compiled(_field(np.arange(3.0), common.domain({D0: (5, 8)}), device))


def test_torch_pytree_roundtrip(device):
    import torch.utils._pytree as torch_pytree

    domain = common.domain({D0: (1, 3), D1: (2, 5)})
    field = _field(np.ones((2, 3)), domain, device)

    children, treespec = torch_pytree.tree_flatten(field)

    assert len(children) == 1
    assert children[0] is field.ndarray
    assert torch_pytree.tree_unflatten(children, treespec).domain == domain
    # the domain is part of the tree structure, hence a domain change forces a retrace
    other_field = _field(np.ones((2, 3)), common.domain({D0: (0, 2), D1: (2, 5)}), device)
    assert torch_pytree.tree_structure(other_field) != treespec


def test_torch_pytree_rejects_fields_of_other_libraries():
    import torch.utils._pytree as torch_pytree

    field = common._field(np.ones(3), domain=common.domain({D0: 3}))

    with pytest.raises(TypeError, match="Only fields of torch tensors"):
        torch_pytree.tree_flatten(field)


def test_torch_pytree_rejects_connectivities_of_other_libraries():
    import torch.utils._pytree as torch_pytree

    conn = common._connectivity(
        np.asarray([0, 1], dtype=np.int32), codomain=V, domain=common.domain({E: 2})
    )

    with pytest.raises(TypeError, match="Only connectivities of torch tensors"):
        torch_pytree.tree_flatten(conn)


@pytest.mark.parametrize("num_field_vertices", [300, 600], ids=["narrowing", "covering"])
def test_torch_compile_premap_with_connectivity_argument(device, num_field_vertices):
    import torch

    table = _skip_value_e2v_table(300)
    field_domain = common.domain({V: (0, num_field_vertices)})
    expected = _premap_and_reduce(
        common._field(np.arange(float(num_field_vertices)), domain=field_domain),
        common._connectivity(
            table,
            codomain=V,
            domain=common.domain({E: table.shape[0], E2VDim: 2}),
            skip_value=-1,
        ),
    )

    compiled, graphs = _compile(_premap_and_reduce)
    result = compiled(
        _field(np.arange(float(num_field_vertices)), field_domain, device),
        _connectivity(table, device, skip_value=-1),
    )

    assert result.domain == expected.domain
    np.testing.assert_allclose(result.asnumpy(), expected.asnumpy())
    (graph,) = graphs
    # the table is a graph input, not a constant
    input_shapes = [
        tuple(node.meta["example_value"].shape) for node in graph.graph.find_nodes(op="placeholder")
    ]
    assert table.shape in input_shapes
    constants = [getattr(graph, node.target) for node in graph.graph.find_nodes(op="get_attr")]
    assert not any(isinstance(c, torch.Tensor) for c in constants)


def test_torch_compile_premap_with_connectivity_closed_over_by_the_region(device):
    import torch

    table = _skip_value_e2v_table(2)
    conn = _connectivity(table, device, skip_value=-1)
    field = _field(np.arange(4.0), common.domain({V: 4}), device)
    traced = torch.compiler.nonstrict_trace(_premap_and_reduce)

    result = torch.compile(lambda f: traced(f, conn), fullgraph=True, backend="aot_eager")(field)

    np.testing.assert_allclose(result.asnumpy(), _premap_and_reduce(field, conn).asnumpy())


def test_torch_compile_premap_with_restricted_connectivity_argument(device):
    table = _skip_value_e2v_table(8)
    field_domain = common.domain({V: (0, 4)})

    def premap_first_neighbor(field, conn):
        return field.premap(conn[NamedIndex(E2VDim, 0)])

    field = _field(np.arange(4.0), field_domain, device)
    conn = _connectivity(table, device, skip_value=-1)
    expected = premap_first_neighbor(field, conn)

    compiled, graphs = _compile(premap_first_neighbor)
    result = compiled(field, conn)

    assert result.domain == expected.domain
    assert result.domain == common.domain({E: (0, 8)})
    np.testing.assert_array_equal(result.asnumpy(), expected.asnumpy())
    assert len(graphs) == 1


def test_torch_compile_retraces_per_connectivity_buffer(device):
    import torch

    table = torch.asarray(_skip_value_e2v_table(8), device=device)
    field = _field(np.arange(8.0), common.domain({V: (0, 8)}), device)

    def connectivity(table):
        return _connectivity(table, device, skip_value=-1)

    compiled, graphs = _compile(lambda f, c: f.premap(c))
    conn = connectivity(table)
    compiled(field, conn)
    compiled(field, conn)
    assert len(graphs) == 1

    compiled(field, connectivity(table))
    assert len(graphs) == 1

    copy = table.clone()
    copy[0, 0] = 7
    result = compiled(field, connectivity(copy))
    assert len(graphs) == 2
    np.testing.assert_array_equal(result.asnumpy(), field.premap(connectivity(copy)).asnumpy())


def test_torch_compile_retraces_after_in_place_write_of_table(device):
    import torch

    table = torch.asarray(np.arange(6, dtype=np.int32), device=device)
    field = _field(np.arange(4.0), common.domain({V: (0, 4)}), device)

    def connectivity():
        return _connectivity(table, device, domain=common.domain({E: 6}))

    compiled, graphs = _compile(lambda f, c: f.premap(c))
    assert compiled(field, connectivity()).domain == common.domain({E: (0, 4)})

    table[:] = torch.asarray(np.arange(6, dtype=np.int32)[::-1].copy(), device=device)
    expected = field.premap(connectivity())
    result = compiled(field, connectivity())

    assert len(graphs) == 2
    assert result.domain == expected.domain == common.domain({E: (2, 6)})
    np.testing.assert_array_equal(result.asnumpy(), expected.asnumpy())


def _chain(error: BaseException) -> str:
    """The messages of `error` and its causes: `torch.compile` reports the error of a trace further down."""
    messages = []
    while error is not None:
        messages.append(str(error))
        error = error.__cause__ or error.__context__
    return "\n".join(messages)


def test_torch_compile_premap_non_contiguous_inverse_image_raises(device):
    conn = _connectivity(
        np.asarray([0, 5, 1], dtype=np.int32), device, domain=common.domain({E: (0, 3)})
    )
    field = _field(np.arange(3.0), common.domain({V: (0, 3)}), device)

    compiled, _ = _compile(lambda f, c: f.premap(c))

    with pytest.raises(Exception) as excinfo:
        compiled(field, conn)
    assert "non-contiguous" in _chain(excinfo.value)


@pytest.mark.parametrize("create", ["restrict", "from_array"])
def test_torch_compile_premap_with_connectivity_not_created_by_constructor_raises(device, create):
    table = _skip_value_e2v_table(2)
    conn = _connectivity(table, device, skip_value=-1)
    if create == "restrict":
        conn = conn[common.domain({E: (0, 4), E2VDim: (0, 2)})]
    else:
        conn = nd_array_field.NdArrayConnectivityField.from_array(
            conn.ndarray, V, domain=conn.domain, skip_value=-1
        )
    field = _field(np.arange(4.0), common.domain({V: 4}), device)

    compiled, _ = _compile(_premap_and_reduce)

    with pytest.raises(Exception) as excinfo:
        compiled(field, conn)
    assert "gtx.as_connectivity" in _chain(excinfo.value)


def test_torch_compile_premap_with_as_offset_computed_in_trace(device):
    C = Dimension("C")
    K = Dimension("K", kind=DimensionKind.VERTICAL)
    Koff = fbuiltins.FieldOffset("Koff", source=K, target=(K,))
    domain = common.domain({C: (0, 2), K: (0, 4)})
    values = np.arange(8.0).reshape(2, 4)
    offsets = np.asarray([[1, 1, 0, -1], [0, 2, 1, 0]], dtype=np.int32)

    def shifted(field, offset_field):
        return field.premap(experimental.as_offset(Koff, offset_field))

    field = _field(values, domain, device)
    compiled, graphs = _compile(shifted)
    result = compiled(field, _field(offsets, domain, device))

    assert result.domain == domain
    np.testing.assert_array_equal(
        result.asnumpy(), np.take_along_axis(values, np.arange(4) + offsets, axis=1)
    )

    # out-of-range targets read some element of the field under a trace, without an error
    for shift in (3, -3):
        result = compiled(field, _field(np.full((2, 4), shift, dtype=np.int32), domain, device))
        assert result.domain == domain
        targets = np.arange(4) + shift
        in_range = (targets >= 0) & (targets < 4)
        np.testing.assert_array_equal(result.asnumpy()[:, in_range], values[:, targets[in_range]])
        assert np.isin(result.asnumpy(), values).all()
    assert len(graphs) == 1


def test_torch_compile_grad_through_premap_with_connectivity_argument(device):
    import torch

    conn = _connectivity(np.asarray([[0, 1], [2, -1]], dtype=np.int32), device, skip_value=-1)
    x = torch.tensor([4.0, 9.0, 16.0, 1.0], dtype=torch.float64, device=device, requires_grad=True)
    field = common._field(x, domain=common.domain({V: 4}))
    traced = torch.compiler.nonstrict_trace(_premap_and_reduce)

    def loss(field, conn):
        return torch.sum(traced(field, conn).ndarray ** 2)

    result = torch.compile(loss, fullgraph=True, backend="aot_eager")(field, conn)
    result.backward()

    np.testing.assert_allclose(result.item(), 13.0**2 + 16.0**2)
    np.testing.assert_allclose(x.grad.cpu().numpy(), [26.0, 26.0, 32.0, 0.0])


def test_torch_compile_premap_with_inductor(device):
    if device.type == "cpu":
        # the C++ compiler of inductor's CPU backend comes from `CXX`, which CI sets to `hipcc`
        # on AMD nodes; the graph is the same as with `aot_eager`
        pytest.skip("inductor's CPU backend depends on the host C++ compiler")
    table = _skip_value_e2v_table(4)
    field = _field(np.arange(8.0), common.domain({V: 8}), device)
    conn = _connectivity(table, device, skip_value=-1)

    compiled, graphs = _compile(_premap_and_reduce, backend="inductor")

    np.testing.assert_allclose(
        compiled(field, conn).asnumpy(), _premap_and_reduce(field, conn).asnumpy()
    )
    assert len(graphs) == 1
