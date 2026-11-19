# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
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

    with pytest.raises(TypeError, match="Only fields of JAX arrays"):
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


def _raises_or_equals_eager(traced, eager):
    """An unsupported connectivity may raise, but must not compute a wrong result."""
    try:
        result = traced()
    except Exception:
        return
    expected = eager()
    assert result.domain == expected.domain
    np.testing.assert_array_equal(result.asnumpy(), expected.asnumpy())


@pytest.mark.requires_jax
def test_jax_jit_premap_with_unsupported_connectivity_is_not_wrong():
    import jax

    E, field, conn_a, _ = _jax_narrowing_setup()
    E2VDim = Dimension("E2V", kind=DimensionKind.LOCAL)
    conn_2d = common._connectivity(
        jax.numpy.asarray([[0, 1], [1, 2], [3, 3]], dtype=np.int32),
        codomain=conn_a.codomain,
        domain=common.domain({E: (0, 3), E2VDim: (0, 2)}),
    )

    def built_in_trace(conn):
        return common._connectivity(conn.ndarray + 1, codomain=conn.codomain, domain=conn.domain)

    def premap(field, conn):
        return field.premap(conn)

    cases = [
        # closed over by the trace
        (lambda: jax.jit(lambda f: f.premap(conn_a))(field), lambda: field.premap(conn_a)),
        (
            lambda: jax.jit(lambda f: f.premap(conn_2d[NamedIndex(E2VDim, 1)]))(field),
            lambda: field.premap(conn_2d[NamedIndex(E2VDim, 1)]),
        ),
        # built from another tracer, directly and behind a pytree boundary
        (
            lambda: jax.jit(lambda f, c: f.premap(built_in_trace(c)))(field, conn_a),
            lambda: field.premap(built_in_trace(conn_a)),
        ),
        (
            lambda: jax.jit(lambda f, c: jax.jit(premap)(f, built_in_trace(c)))(field, conn_a),
            lambda: field.premap(built_in_trace(conn_a)),
        ),
        (
            lambda: jax.jit(lambda f, c: jax.lax.cond(True, premap, premap, f, built_in_trace(c)))(
                field, conn_a
            ),
            lambda: field.premap(built_in_trace(conn_a)),
        ),
    ]
    for traced, eager in cases:
        _raises_or_equals_eager(traced, eager)


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

    # out-of-range targets read some element of the field under a trace, without an error
    for shift in (3, -3):
        result = jax.jit(shifted)(
            field, common._field(jax.numpy.full((2, 4), shift, dtype=np.int32), domain=domain)
        ).asnumpy()
        targets = np.arange(4) + shift
        in_range = (targets >= 0) & (targets < 4)
        np.testing.assert_array_equal(result[:, in_range], values[:, targets[in_range]])
        assert np.isin(result, values).all()


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

    with pytest.raises(TypeError, match="pytree structure"):
        jax.jit(
            lambda index, field, conn1, conn2: field.premap(
                jax.lax.switch(index, [lambda: conn1, lambda: conn2])
            )
        )(1, field, conn_a, conn_b)

    # the same table in both branches
    same_table = common._connectivity(
        conn_a.ndarray, codomain=conn_a.codomain, domain=conn_a.domain
    )
    result = jax.jit(premap_selected)(predicate, field, conn_a, same_table)
    assert result.domain == common.domain({E: (1, 4)})
    np.testing.assert_array_equal(result.asnumpy(), [10.0, 20.0, 30.0])


@pytest.mark.requires_jax
def test_jax_jit_premap_closing_over_connectivity_argument():
    import jax

    E, field, conn_a, _ = _jax_narrowing_setup()

    def in_cond(field, conn):
        return jax.lax.cond(True, lambda: field.premap(conn), lambda: field.premap(conn))

    def in_inner_jit(field, conn):
        return jax.jit(lambda field: field.premap(conn))(field)

    def in_scan(field, conn):
        result, _ = jax.lax.scan(
            lambda carry, _: (field.premap(conn), None), field.premap(conn), None, length=2
        )
        return result

    def and_again_afterwards(closing_over, field, conn):
        return closing_over(field, conn), field.premap(conn)

    for closing_over in (in_cond, in_inner_jit, in_scan):
        traced = jax.jit(functools.partial(and_again_afterwards, closing_over))
        for result in traced(field, conn_a):
            assert result.domain == common.domain({E: (1, 4)})
            np.testing.assert_array_equal(result.asnumpy(), [10.0, 20.0, 30.0])


@pytest.mark.requires_jax
def test_jax_pytree_rejects_connectivities_of_other_libraries():
    import jax

    _, field, conn_a, _ = _jax_narrowing_setup()
    numpy_conn = common._connectivity(
        np.asarray([0, 1, 2, 3, 5], dtype=np.int32), codomain=conn_a.codomain, domain=conn_a.domain
    )

    with pytest.raises(TypeError, match="Only connectivities of JAX arrays"):
        jax.jit(lambda field, conn: field.premap(conn))(field, numpy_conn)


@pytest.mark.requires_jax
@pytest.mark.parametrize("to_leaf", [lambda table: table + 1, lambda table: np.asarray(table) + 1])
def test_jax_pytree_rebuilt_connectivity_reads_its_own_values(to_leaf):
    import jax

    E, field, conn_a, _ = _jax_narrowing_setup()

    rebuilt = jax.tree.map(to_leaf, conn_a)

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

    # each element is a table without values: not narrowed, the in-range targets are read
    np.testing.assert_array_equal(np.asarray(result)[0, 1:4], [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(np.asarray(result)[1, 0:3], [10.0, 20.0, 30.0])


@pytest.mark.requires_jax
def test_jax_jit_cond_returning_field():
    import jax

    _, field, _, _ = _jax_narrowing_setup()

    result = jax.jit(
        lambda predicate, field: jax.lax.cond(predicate, lambda: field, lambda: field * 2.0)
    )(False, field)

    assert result.domain == field.domain
    np.testing.assert_array_equal(result.asnumpy(), [20.0, 40.0, 60.0])


@pytest.mark.requires_jax
def test_jax_jit_grad_through_premap_with_connectivity_argument():
    import jax

    _, field, conn_a, conn_b = _jax_narrowing_setup()

    def loss(x, conn):
        return jax.numpy.sum(common._field(x, domain=field.domain).premap(conn).ndarray ** 2)

    x = field.ndarray
    for conn in (conn_a, conn_b):
        np.testing.assert_allclose(jax.jit(jax.grad(loss))(x, conn), jax.grad(loss)(x, conn))


@pytest.mark.requires_jax
def test_jax_jit_vmap_over_fields_with_connectivity_argument():
    import jax

    _, field, conn_a, _ = _jax_narrowing_setup()
    xs = jax.numpy.stack([field.ndarray, 2.0 * field.ndarray])

    def premap_each(xs, conn):
        return jax.vmap(lambda x: common._field(x, domain=field.domain).premap(conn).ndarray)(xs)

    np.testing.assert_array_equal(
        jax.jit(premap_each)(xs, conn_a), [[10.0, 20.0, 30.0], [20.0, 40.0, 60.0]]
    )


@pytest.mark.requires_jax
def test_jax_loop_carry_with_connectivity():
    import jax

    E, field, conn_a, conn_b = _jax_narrowing_setup()

    def scan(conn, next_conn):
        carried, _ = jax.lax.scan(
            lambda c, x: (c if next_conn is None else next_conn, x), conn, jax.numpy.arange(2.0)
        )
        return field.premap(carried)

    def fori(conn, next_conn):
        carried = jax.lax.fori_loop(0, 2, lambda i, c: c if next_conn is None else next_conn, conn)
        return field.premap(carried)

    def while_(conn, next_conn):
        def body(state):
            i, c = state
            return i + 1, (c if next_conn is None else next_conn)

        _, carried = jax.lax.while_loop(lambda state: state[0] < 2, body, (0, conn))
        return field.premap(carried)

    for loop in (scan, fori, while_):
        result = jax.jit(functools.partial(loop, next_conn=None))(conn_a)
        assert result.domain == common.domain({E: (1, 4)})
        np.testing.assert_array_equal(result.asnumpy(), [10.0, 20.0, 30.0])
        with pytest.raises(TypeError):
            jax.jit(loop)(conn_a, conn_b)


@pytest.mark.requires_jax
def test_jax_jit_as_offset_connectivity_through_cond_and_scan():
    import jax

    from gt4py.next.ffront import experimental

    K = Dimension("K", kind=DimensionKind.VERTICAL)
    Koff = fbuiltins.FieldOffset("Koff", source=K, target=(K,))
    domain = common.domain({K: (0, 4)})
    field = common._field(jax.numpy.asarray([0.0, 1.0, 2.0, 3.0]), domain=domain)
    offsets = common._field(jax.numpy.asarray([1, 1, 0, -1], dtype=np.int32), domain=domain)

    def via_cond(predicate, field, offset_field):
        conn = experimental.as_offset(Koff, offset_field)
        return jax.lax.cond(
            predicate,
            lambda c: field.premap(c).ndarray,
            lambda c: 2.0 * field.premap(c).ndarray,
            conn,
        )

    def via_scan(field, offset_field):
        conn = experimental.as_offset(Koff, offset_field)
        return jax.lax.scan(
            lambda c, x: (c, x * field.premap(c).ndarray), conn, jax.numpy.asarray([1.0, 2.0])
        )[1]

    np.testing.assert_array_equal(jax.jit(via_cond)(True, field, offsets), [1.0, 2.0, 2.0, 2.0])
    np.testing.assert_array_equal(jax.jit(via_cond)(False, field, offsets), [2.0, 4.0, 4.0, 4.0])
    np.testing.assert_array_equal(
        jax.jit(via_scan)(field, offsets), [[1.0, 2.0, 2.0, 2.0], [2.0, 4.0, 4.0, 4.0]]
    )
