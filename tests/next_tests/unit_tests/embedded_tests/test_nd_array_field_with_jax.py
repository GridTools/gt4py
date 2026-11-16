# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import re

import numpy as np
import pytest

from gt4py.next import common
from gt4py.next.common import Dimension, DimensionKind, NamedIndex
from gt4py.next.embedded import context as embedded_context, nd_array_field
from gt4py.next.ffront import fbuiltins


D0 = Dimension("D0")
D1 = Dimension("D1")


@pytest.mark.requires_jax
def test_jax_jit_field_arguments():
    import jax

    domain = common.domain({D0: (1, 3), D1: (2, 5)})
    a = common._field(
        jax.numpy.asarray(np.arange(6, dtype=np.float64).reshape(2, 3)), domain=domain
    )
    b = common._field(jax.numpy.ones((2, 3), dtype=np.float64), domain=domain)

    @jax.jit
    def add(x, y):
        return x + y

    result = add(a, b)

    assert isinstance(result, common.Field)
    assert result.domain == domain
    np.testing.assert_allclose(result.asnumpy(), a.asnumpy() + b.asnumpy())


@pytest.mark.requires_jax
def test_jax_pytree_roundtrip():
    import jax

    domain = common.domain({D0: (1, 3), D1: (2, 5)})
    field = common._field(jax.numpy.ones((2, 3), dtype=np.float64), domain=domain)

    children, treedef = jax.tree_util.tree_flatten(field)

    assert len(children) == 1
    assert children[0] is field.ndarray

    restored = jax.tree_util.tree_unflatten(treedef, children)
    assert restored.domain == domain

    other_field = common._field(
        jax.numpy.ones((2, 3), dtype=np.float64), domain=common.domain({D0: (0, 2), D1: (2, 5)})
    )
    # the domain is part of the tree structure, hence a domain change forces a 'jax.jit' retrace
    assert jax.tree_util.tree_structure(other_field) != treedef


@pytest.mark.requires_jax
def test_jax_pytree_rejects_fields_of_other_libraries():
    import jax

    field = common._field(np.ones(3), domain=common.domain({D0: 3}))

    with pytest.raises(AssertionError, match="Only fields of JAX arrays"):
        jax.jit(lambda f: f * 2.0)(field)


@pytest.mark.requires_jax
def test_jax_pytree_non_array_leaves():
    import jax

    D_BATCH = Dimension("D_BATCH")
    field = common._field(jax.numpy.arange(3.0), domain=common.domain({D0: 3}))
    batched = common._field(jax.numpy.ones((2, 3)), domain=common.domain({D_BATCH: 2, D0: 3}))

    shapes = jax.tree.map(lambda x: x.shape, field)
    assert jax.tree.leaves(shapes) == [3]
    assert jax.eval_shape(lambda f: f * 2.0, field).ndarray.shape == (3,)
    # a field with an axis index as leaf, as `vmap` prefix for the batched field's buffer
    axes = nd_array_field.NdArrayField(common.domain({D0: 3}), 0)
    sums = jax.vmap(lambda f: f.ndarray.sum(), in_axes=(axes,))(
        nd_array_field.NdArrayField(common.domain({D0: 3}), batched.ndarray)
    )
    np.testing.assert_allclose(sums, [3.0, 3.0])


@pytest.mark.requires_jax
def test_jax_traced_array_dispatch():
    import jax

    domain = common.domain({D0: (0, 3)})
    x = jax.numpy.arange(3, dtype=np.float64)

    def loss(arr):
        field = common._field(arr, domain=domain)
        assert isinstance(field, nd_array_field.NdArrayField)
        assert field.array_ns is jax.numpy
        return jax.numpy.sum((field * field).ndarray)

    np.testing.assert_allclose(jax.grad(loss)(x), 2.0 * x.__array__())
    np.testing.assert_allclose(jax.jit(loss)(x), loss(x))

    offsets = jax.numpy.asarray([[2, 0], [1, 2]])
    codomain = common.domain({D0: (0, 2), D1: (0, 2)})

    def make_connectivity(arr):
        return common._connectivity(arr, codomain=D0, domain=codomain).ndarray

    np.testing.assert_array_equal(jax.jit(make_connectivity)(offsets), offsets)


@pytest.mark.requires_jax
def test_jax_grad_through_premap_with_skip_values():
    import jax

    V = Dimension("V")
    E = Dimension("E")
    E2VDim = Dimension("E2V", kind=DimensionKind.LOCAL)
    e2v = common._connectivity(
        jax.numpy.asarray([[0, 1], [2, -1]], dtype=np.int32),
        codomain=V,
        domain=common.domain({E: (0, 2), E2VDim: (0, 2)}),
        skip_value=-1,
    )

    def loss(x):
        field = common._field(x, domain=common.domain({V: (0, 4)}))
        with embedded_context.update(offset_provider={"E2V": e2v}):
            result = fbuiltins.neighbor_sum(fbuiltins.sqrt(field.premap(e2v)), axis=E2VDim)
        return jax.numpy.sum(result.ndarray)

    x = jax.numpy.asarray([4.0, 9.0, 16.0, -1.0])

    np.testing.assert_allclose(loss(x), 9.0)
    np.testing.assert_allclose(jax.grad(loss)(x), [0.25, 1.0 / 6.0, 0.125, 0.0])


def _skip_value_e2v_table(num_vertices: int) -> np.ndarray:
    """Two edges per vertex, both entries equal, every seventh second entry a skip value."""
    vertex = np.arange(4 * num_vertices) // 2
    table = np.stack([vertex, vertex], axis=1).astype(np.int32)
    table[::7, 1] = -1
    return table


@pytest.mark.requires_jax
@pytest.mark.parametrize("num_field_vertices", [300, 600], ids=["narrowing", "covering"])
def test_jax_jit_premap_with_connectivity_argument(num_field_vertices):
    import jax

    V = Dimension("V")
    E = Dimension("E")
    E2VDim = Dimension("E2V", kind=DimensionKind.LOCAL)
    table = _skip_value_e2v_table(300)
    conn_domain = common.domain({E: (0, table.shape[0]), E2VDim: (0, 2)})
    field_domain = common.domain({V: (0, num_field_vertices)})

    def premap_and_reduce(field, conn):
        with embedded_context.update(offset_provider={"E2V": conn}):
            return fbuiltins.neighbor_sum(field.premap(conn), axis=E2VDim)

    expected = premap_and_reduce(
        common._field(np.arange(num_field_vertices, dtype=np.float64), domain=field_domain),
        common._connectivity(table, codomain=V, domain=conn_domain, skip_value=-1),
    )

    jitted = jax.jit(premap_and_reduce)
    field = common._field(
        jax.numpy.arange(num_field_vertices, dtype=np.float64), domain=field_domain
    )
    conn = common._connectivity(
        jax.numpy.asarray(table), codomain=V, domain=conn_domain, skip_value=-1
    )
    result = jitted(field, conn)

    assert result.domain == expected.domain
    np.testing.assert_allclose(result.asnumpy(), expected.asnumpy())

    stablehlo = jitted.lower(field, conn).as_text()
    main_arguments = re.search(r"@main\(([^)]*)\)", stablehlo).group(1)
    assert f"tensor<{table.shape[0]}x2xi32>" in main_arguments
    # large constants print as hex
    assert 'dense<"0x' not in stablehlo
    assert 'dense<"0x' in jax.jit(lambda field: field.premap(conn)).lower(field).as_text()


@pytest.mark.requires_jax
def test_jax_jit_premap_with_closed_over_restricted_connectivity():
    import jax

    V = Dimension("V")
    E = Dimension("E")
    E2VDim = Dimension("E2V", kind=DimensionKind.LOCAL)
    table = np.asarray([[0, 1], [1, 0], [2, 2]], dtype=np.int32)
    conn = common._connectivity(
        jax.numpy.asarray(table),
        codomain=V,
        domain=common.domain({E: (0, 3), E2VDim: (0, 2)}),
    )
    field = common._field(jax.numpy.asarray([10.0, 20.0]), domain=common.domain({V: (0, 2)}))

    result = jax.jit(lambda field: field.premap(conn[common.NamedIndex(E2VDim, 1)]))(field)

    assert result.domain == common.domain({E: (0, 2)})
    np.testing.assert_array_equal(result.asnumpy(), [20.0, 10.0])


@pytest.mark.requires_jax
def test_jax_jit_premap_with_as_offset_computed_in_trace():
    import jax

    from gt4py.next.ffront import experimental

    C = Dimension("C")
    K = Dimension("K", kind=DimensionKind.VERTICAL)
    Koff = fbuiltins.FieldOffset("Koff", source=K, target=(K,))
    domain = common.domain({C: (0, 2), K: (0, 4)})
    values = np.arange(8, dtype=np.float64).reshape(2, 4)
    offsets = np.asarray([[1, 1, 0, -1], [0, 2, 1, 0]], dtype=np.int32)

    def shifted(field, offset_field):
        return field.premap(experimental.as_offset(Koff, offset_field))

    field = common._field(jax.numpy.asarray(values), domain=domain)
    offset_field = common._field(jax.numpy.asarray(offsets), domain=domain)
    expected = shifted(field, offset_field)

    result = jax.jit(shifted)(field, offset_field)

    assert result.domain == domain
    np.testing.assert_array_equal(result.asnumpy(), expected.asnumpy())
    np.testing.assert_array_equal(
        result.asnumpy(), np.take_along_axis(values, np.arange(4) + offsets, axis=1)
    )

    # out-of-range targets read the boundary value under a trace
    beyond = common._field(jax.numpy.asarray(np.full((2, 4), 3, dtype=np.int32)), domain=domain)
    np.testing.assert_array_equal(
        jax.jit(shifted)(field, beyond).asnumpy(), np.repeat(values[:, -1:], 4, axis=1)
    )
    below = common._field(jax.numpy.asarray(np.full((2, 4), -3, dtype=np.int32)), domain=domain)
    np.testing.assert_array_equal(
        jax.jit(shifted)(field, below).asnumpy(),
        np.take_along_axis(
            values, np.maximum(np.arange(4) - 3, 0)[None, :].repeat(2, axis=0), axis=1
        ),
    )


@pytest.mark.requires_jax
def test_jax_jit_premap_with_closed_over_as_offset_narrows_like_eager():
    import jax

    from gt4py.next.ffront import experimental

    K = Dimension("K", kind=DimensionKind.VERTICAL)
    Koff = fbuiltins.FieldOffset("Koff", source=K, target=(K,))
    domain = common.domain({K: (0, 4)})
    field = common._field(jax.numpy.asarray([0.0, 1.0, 2.0, 3.0]), domain=domain)
    offset_field = common._field(jax.numpy.ones(4, dtype=np.int32), domain=domain)

    result = jax.jit(lambda field: field.premap(experimental.as_offset(Koff, offset_field)))(field)

    assert result.domain == common.domain({K: (0, 3)})
    np.testing.assert_array_equal(result.asnumpy(), [1.0, 2.0, 3.0])


@pytest.mark.requires_jax
def test_jax_jit_premap_with_field_offset_does_not_leak_tracers():
    import jax

    V = Dimension("V")
    E = Dimension("E")
    E2VDim = Dimension("E2V", kind=DimensionKind.LOCAL)
    E2V = fbuiltins.FieldOffset("E2V", source=V, target=(E, E2VDim))
    conn = common._connectivity(
        jax.numpy.asarray([[0, 1], [2, 3]], dtype=np.int32),
        codomain=V,
        domain=common.domain({E: (0, 2), E2VDim: (0, 2)}),
    )
    field = common._field(jax.numpy.arange(4, dtype=np.float64), domain=common.domain({V: (0, 4)}))

    def shifted(field, conn):
        with embedded_context.update(offset_provider={"E2V": conn}):
            return field(E2V)

    with jax.checking_leaks():
        result = jax.jit(shifted)(field, conn)

    np.testing.assert_array_equal(result.asnumpy(), [[0.0, 1.0], [2.0, 3.0]])


@pytest.mark.requires_jax
def test_nested_jax_jit_premap_narrows():
    import jax

    V = Dimension("V")
    E = Dimension("E")
    E2VDim = Dimension("E2V", kind=DimensionKind.LOCAL)
    conn = common._connectivity(
        jax.numpy.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.int32),
        codomain=V,
        domain=common.domain({E: (0, 3), E2VDim: (0, 2)}),
    )
    field = common._field(jax.numpy.arange(4, dtype=np.float64), domain=common.domain({V: (0, 4)}))

    inner = jax.jit(lambda field, conn: field.premap(conn))
    result = jax.jit(lambda field, conn: inner(field, conn))(field, conn)

    assert result.domain == common.domain({E: (0, 2), E2VDim: (0, 2)})
    np.testing.assert_array_equal(result.asnumpy(), [[0.0, 1.0], [2.0, 3.0]])


@pytest.mark.requires_jax
def test_jax_jit_retraces_per_connectivity_buffer():
    import jax

    V = Dimension("V")
    E = Dimension("E")
    E2VDim = Dimension("E2V", kind=DimensionKind.LOCAL)
    table = jax.numpy.asarray(_skip_value_e2v_table(8))
    conn_domain = common.domain({E: (0, table.shape[0]), E2VDim: (0, 2)})
    field = common._field(jax.numpy.arange(8, dtype=np.float64), domain=common.domain({V: (0, 8)}))
    traces = 0

    @jax.jit
    def premap(field, conn):
        nonlocal traces
        traces += 1
        return field.premap(conn)

    conn = common._connectivity(table, codomain=V, domain=conn_domain, skip_value=-1)
    premap(field, conn)
    premap(field, conn)
    assert traces == 1

    premap(field, common._connectivity(table, codomain=V, domain=conn_domain, skip_value=-1))
    assert traces == 1

    premap(
        field,
        common._connectivity(
            jax.numpy.array(table, copy=True), codomain=V, domain=conn_domain, skip_value=-1
        ),
    )
    assert traces == 2

    premap(field, conn[NamedIndex(E2VDim, 0)])
    premap(field, conn[NamedIndex(E2VDim, 0)])
    assert traces == 3


@pytest.mark.requires_jax
def test_jax_jit_premap_non_contiguous_inverse_image_raises():
    import jax

    V = Dimension("V")
    E = Dimension("E")
    conn = common._connectivity(
        jax.numpy.asarray([0, 5, 1], dtype=np.int32),
        codomain=V,
        domain=common.domain({E: (0, 3)}),
    )
    field = common._field(jax.numpy.arange(3, dtype=np.float64), domain=common.domain({V: (0, 3)}))

    with pytest.raises(ValueError, match="non-contiguous"):
        jax.jit(lambda field, conn: field.premap(conn))(field, conn)


@pytest.mark.requires_jax
def test_jax_jit_premap_with_restricted_connectivity_argument():
    import jax

    V = Dimension("V")
    E = Dimension("E")
    E2VDim = Dimension("E2V", kind=DimensionKind.LOCAL)
    table = _skip_value_e2v_table(8)
    conn_domain = common.domain({E: (0, table.shape[0]), E2VDim: (0, 2)})
    field_domain = common.domain({V: (0, 4)})

    def premap_first_neighbor(field, conn):
        return field.premap(conn[NamedIndex(E2VDim, 0)])

    expected = premap_first_neighbor(
        common._field(np.arange(4, dtype=np.float64), domain=field_domain),
        common._connectivity(table, codomain=V, domain=conn_domain, skip_value=-1),
    )
    result = jax.jit(premap_first_neighbor)(
        common._field(jax.numpy.arange(4, dtype=np.float64), domain=field_domain),
        common._connectivity(
            jax.numpy.asarray(table), codomain=V, domain=conn_domain, skip_value=-1
        ),
    )

    assert result.domain == expected.domain
    np.testing.assert_allclose(result.asnumpy(), expected.asnumpy())


def _jax_narrowing_setup():
    """A field on V:(1,4) and two tables on E:(0,5) mapping into it on E:(1,4) and on E:(0,3)."""
    import jax

    V = Dimension("V")
    E = Dimension("E")
    field = common._field(jax.numpy.asarray([10.0, 20.0, 30.0]), domain=common.domain({V: (1, 4)}))
    conn_domain = common.domain({E: (0, 5)})
    conn_a = common._connectivity(
        jax.numpy.asarray([0, 1, 2, 3, 5], dtype=np.int32), codomain=V, domain=conn_domain
    )
    conn_b = common._connectivity(
        jax.numpy.asarray([1, 2, 3, 0, 5], dtype=np.int32), codomain=V, domain=conn_domain
    )
    return E, field, conn_a, conn_b


@pytest.mark.requires_jax
def test_jax_jit_premap_with_connectivity_from_array():
    import jax

    E, field, conn_a, _ = _jax_narrowing_setup()
    conn = nd_array_field.NdArrayConnectivityField.from_array(
        conn_a.ndarray, conn_a.codomain, domain=conn_a.domain
    )

    for result in (
        jax.jit(lambda field, conn: field.premap(conn))(field, conn),
        jax.jit(lambda field: field.premap(conn))(field),
    ):
        assert result.domain == common.domain({E: (1, 4)})
        np.testing.assert_array_equal(result.asnumpy(), [10.0, 20.0, 30.0])


@pytest.mark.requires_jax
@pytest.mark.parametrize("predicate", [True, False])
def test_jax_jit_premap_with_connectivity_selected_by_cond(predicate):
    import jax

    E, field, conn_a, conn_b = _jax_narrowing_setup()

    def premap_selected(predicate, field, conn1, conn2):
        return field.premap(jax.lax.cond(predicate, lambda: conn1, lambda: conn2))

    with pytest.raises(TypeError, match="pytree structure"):
        jax.jit(premap_selected)(predicate, field, conn_a, conn_b)
    with pytest.raises(TypeError, match="pytree structure"):
        jax.jit(lambda predicate, field, conn: premap_selected(predicate, field, conn, conn_b))(
            predicate, field, conn_a
        )

    # the same table in both branches
    same_table = common._connectivity(
        conn_a.ndarray, codomain=conn_a.codomain, domain=conn_a.domain
    )
    result = jax.jit(premap_selected)(predicate, field, conn_a, same_table)
    assert result.domain == common.domain({E: (1, 4)})
    np.testing.assert_array_equal(result.asnumpy(), [10.0, 20.0, 30.0])


@pytest.mark.requires_jax
def test_jax_jit_premap_with_connectivity_selected_by_switch():
    import jax

    _, field, conn_a, conn_b = _jax_narrowing_setup()

    def premap_selected(index, field, conn1, conn2):
        return field.premap(jax.lax.switch(index, [lambda: conn1, lambda: conn2]))

    with pytest.raises(TypeError, match="pytree structure"):
        jax.jit(premap_selected)(1, field, conn_a, conn_b)


@pytest.mark.requires_jax
def test_jax_scan_with_connectivity_carry():
    import jax

    _, field, conn_a, conn_b = _jax_narrowing_setup()

    def premap_sums(conn, next_conn):
        def body(carry, x):
            return (carry if next_conn is None else next_conn), x * jax.numpy.sum(
                field.premap(carry).ndarray
            )

        return jax.lax.scan(body, conn, jax.numpy.asarray([1.0, 2.0]))[1]

    np.testing.assert_array_equal(premap_sums(conn_a, None), [60.0, 120.0])
    np.testing.assert_array_equal(
        jax.jit(lambda conn: premap_sums(conn, None))(conn_a), [60.0, 120.0]
    )

    with pytest.raises(TypeError, match="pytree structure"):
        premap_sums(conn_a, conn_b)
    with pytest.raises(TypeError, match="pytree structure"):
        jax.jit(premap_sums)(conn_a, conn_b)


@pytest.mark.requires_jax
@pytest.mark.parametrize("nested", [False, True])
def test_jax_jit_premap_with_replaced_table_has_no_values(nested):
    import jax

    E, field, conn_a, _ = _jax_narrowing_setup()
    premap = (
        jax.jit(lambda field, conn: field.premap(conn))
        if nested
        else nd_array_field.NdArrayField.premap
    )

    def premap_next(field, conn):
        return premap(field, dataclasses.replace(conn, _ndarray=conn.ndarray + 1))

    result = jax.jit(premap_next)(field, conn_a)

    # as for every table computed in the trace: not narrowed, out-of-range targets clamped
    assert result.domain == common.domain({E: (0, 5)})
    np.testing.assert_array_equal(result.asnumpy(), [10.0, 20.0, 30.0, 30.0, 30.0])


@pytest.mark.requires_jax
def test_jax_jit_premap_clamps_in_the_gather():
    import jax

    E, field, conn_a, _ = _jax_narrowing_setup()

    def premap(field, table):
        return field.premap(dataclasses.replace(conn_a, _ndarray=table))

    # targets -2, -1, 0, 1, 3 of a field on V:(1,4)
    table = conn_a.ndarray - 2
    result = jax.jit(premap)(field, table)

    assert result.domain == common.domain({E: (0, 5)})
    np.testing.assert_array_equal(result.asnumpy(), [10.0, 10.0, 10.0, 10.0, 30.0])
    for lowered in (
        jax.jit(premap).lower(field, table).as_text(),
        jax.jit(lambda field, conn: field.premap(conn)).lower(field, conn_a).as_text(),
    ):
        assert "stablehlo.gather" in lowered
        assert not re.search(r"stablehlo\.(clamp|minimum|maximum|select)", lowered)


@pytest.mark.requires_jax
@pytest.mark.parametrize("predicate, factor", [(True, 1.0), (False, 2.0)])
def test_jax_jit_premap_in_cond_branches_closing_over_connectivity(predicate, factor):
    import jax

    _, field, conn_a, _ = _jax_narrowing_setup()

    def premap_in_branches(predicate, field, conn):
        return jax.lax.cond(
            predicate,
            lambda: field.premap(conn).ndarray * 1.0,
            lambda: field.premap(conn).ndarray * 2.0,
        )

    result = jax.jit(premap_in_branches)(predicate, field, conn_a)

    np.testing.assert_array_equal(result, np.asarray([10.0, 20.0, 30.0]) * factor)


@pytest.mark.requires_jax
def test_jax_jit_premap_in_nested_jit_closing_over_connectivity():
    import jax

    E, field, conn_a, _ = _jax_narrowing_setup()

    def premap_twice(field, conn):
        inner = jax.jit(lambda field: field.premap(conn))(field)
        return inner, field.premap(conn)

    for result in jax.jit(premap_twice)(field, conn_a):
        assert result.domain == common.domain({E: (1, 4)})
        np.testing.assert_array_equal(result.asnumpy(), [10.0, 20.0, 30.0])


@pytest.mark.requires_jax
def test_jax_jit_premap_in_scan_body_closing_over_connectivity():
    import jax

    _, field, conn_a, _ = _jax_narrowing_setup()

    def premap_in_scan(field, conn):
        def body(carry, _):
            return carry + field.premap(conn).ndarray, None

        total, _ = jax.lax.scan(body, jax.numpy.zeros(3), jax.numpy.arange(2))
        return total + field.premap(conn).ndarray

    result = jax.jit(premap_in_scan)(field, conn_a)

    np.testing.assert_array_equal(result, [30.0, 60.0, 90.0])


@pytest.mark.requires_jax
def test_jax_pytree_rejects_connectivities_of_other_libraries():
    import jax

    _, field, conn_a, _ = _jax_narrowing_setup()
    numpy_conn = common._connectivity(
        np.asarray([0, 1, 2, 3, 5], dtype=np.int32), codomain=conn_a.codomain, domain=conn_a.domain
    )

    with pytest.raises(AssertionError, match="Only connectivities of JAX arrays"):
        jax.jit(lambda field, conn: field.premap(conn))(field, numpy_conn)

    # not a pytree leaf, so it is not flattened
    result = jax.jit(lambda field: field.premap(numpy_conn))(field)
    np.testing.assert_array_equal(result.asnumpy(), [10.0, 20.0, 30.0])


@pytest.mark.requires_jax
def test_jax_pytree_of_concrete_connectivity_has_no_array_ops_state():
    import jax

    E, field, conn_a, _ = _jax_narrowing_setup()

    rebuilt = jax.tree.map(lambda table: table + 1, conn_a)

    assert rebuilt._array_ops_state is None
    # targets 1, 2, 3, 4, 6 of a field on V:(1,4)
    result = field.premap(rebuilt)
    assert result.domain == common.domain({E: (0, 3)})
    np.testing.assert_array_equal(result.asnumpy(), [10.0, 20.0, 30.0])


@pytest.mark.requires_jax
def test_jax_vmap_premap_over_connectivities_has_no_values():
    import jax

    _, field, conn_a, conn_b = _jax_narrowing_setup()
    stacked = jax.tree.unflatten(
        jax.tree.structure(conn_a), [jax.numpy.stack([conn_a.ndarray, conn_b.ndarray])]
    )

    result = jax.vmap(lambda conn: field.premap(conn).ndarray)(stacked)

    # each element is a table without values: not narrowed, out-of-range targets clamped
    np.testing.assert_array_equal(
        np.asarray(result), [[10.0, 10.0, 20.0, 30.0, 30.0], [10.0, 20.0, 30.0, 10.0, 30.0]]
    )


@pytest.mark.requires_jax
def test_jax_jit_premap_ignores_array_ops_state_of_another_library():
    import jax

    E, field, conn_a, _ = _jax_narrowing_setup()

    def premap(field, conn):
        return field.premap(dataclasses.replace(conn, _array_ops_state=object()))

    result = jax.jit(premap)(field, conn_a)

    assert result.domain == common.domain({E: (0, 5)})
    np.testing.assert_array_equal(result.asnumpy(), [10.0, 10.0, 20.0, 30.0, 30.0])


@pytest.mark.requires_jax
def test_jax_jit_cond_returning_field():
    import jax

    _, field, _, _ = _jax_narrowing_setup()

    result = jax.jit(
        lambda predicate, field: jax.lax.cond(predicate, lambda: field, lambda: field * 2.0)
    )(False, field)

    assert result.domain == field.domain
    np.testing.assert_array_equal(result.asnumpy(), [20.0, 40.0, 60.0])
