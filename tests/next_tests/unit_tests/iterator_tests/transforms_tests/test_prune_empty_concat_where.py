# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pytest

from gt4py.next import common
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms.prune_empty_concat_where import prune_empty_concat_where
from gt4py.next.iterator.transforms.concat_where import canonicalize_domain_argument
from gt4py.next.iterator.transforms.infer_domain import infer_expr
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas
from gt4py.next.iterator.ir_utils import domain_utils
from gt4py.next.type_system import type_specifications as ts

Vertex = common.Dimension(value="Vertex", kind=common.DimensionKind.HORIZONTAL)
K = common.Dimension(value="K", kind=common.DimensionKind.VERTICAL)

float64 = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
vertex_k_field = ts.FieldType(dims=[Vertex, K], dtype=float64)
vertex_field = ts.FieldType(dims=[Vertex], dtype=float64)
k_field = ts.FieldType(dims=[K], dtype=float64)


def _infer(testee, accessed_domain):
    testee = canonicalize_domain_argument(testee)
    testee, _ = infer_expr(
        testee,
        domain_utils.SymbolicDomain.from_expr(
            im.domain(common.GridType.UNSTRUCTURED, accessed_domain)
        ),
        offset_provider={},
    )
    return testee


@pytest.mark.parametrize(
    "accessed_domain, cond_domain, expected",
    [
        # TODO(tehrengruber): Implement in pass and enable commented out symbolic test cases below.
        # cond spans entire accessed domain of true branch value
        ({Vertex: (0, 10), K: (0, 10)}, {Vertex: (0, 10)}, "a"),
        # ({Vertex: ("v0", "v1")}, {Vertex: ("v0", "v1")}, "a"),
        # cond is empty
        ({Vertex: (0, 10)}, {Vertex: (0, 0)}, "b"),
        ({Vertex: (0, 10), K: (0, 10)}, {K: (0, 0)}, "b"),
        # ({Vertex: ("v0", "v0")}, {Vertex: ("v0", "v0")}, "b"),
        # cond disjoint from the accessed domain:
        #  entirely below it
        ({Vertex: (0, 10), K: (0, 10)}, {K: (itir.InfinityLiteral.NEGATIVE, 0)}, "b"),
        #  entirely above it
        ({Vertex: (0, 10), K: (0, 10)}, {K: (10, itir.InfinityLiteral.POSITIVE)}, "b"),
        # cond covers the accessed domain:
        #  from below
        ({Vertex: (0, 10), K: (0, 10)}, {K: (itir.InfinityLiteral.NEGATIVE, 10)}, "a"),
        #  from above
        ({Vertex: (0, 10), K: (0, 10)}, {K: (0, itir.InfinityLiteral.POSITIVE)}, "a"),
        # cond subset of accessed domain, no transformation occurs
        ({Vertex: (0, 10)}, {Vertex: (1, 2)}, None),
        ({Vertex: (0, 10), K: (0, 10)}, {Vertex: (1, 2)}, None),
        # ({Vertex: ("v0", "v1")}, {Vertex: ("v0", "v2")}, None)
        # cond subset of accessed domain, but only one half-space
        #  after canonicalization will remain
        (
            {Vertex: (0, 10)},
            {Vertex: (0, 1)},
            im.concat_where(
                im.domain(
                    common.GridType.UNSTRUCTURED, {Vertex: (1, itir.InfinityLiteral.POSITIVE)}
                ),
                "b",
                "a",
            ),
        ),
    ],
)
def test_prune_concat_where(accessed_domain, cond_domain, expected):
    field_t = ts.FieldType(dims=list(accessed_domain.keys()), dtype=float64)
    testee = im.concat_where(
        im.domain(common.GridType.UNSTRUCTURED, cond_domain),
        im.ref("a", field_t),
        im.ref("b", field_t),
    )
    testee = _infer(testee, accessed_domain)

    if expected is None:
        expected = testee
    expected = im.ensure_expr(expected)
    expected = canonicalize_domain_argument(expected)
    expected = InlineLambdas.apply(expected)

    actual = prune_empty_concat_where(testee)
    actual = InlineLambdas.apply(actual)
    assert actual == expected


def _concat_where(
    cond_range: tuple[int | itir.InfinityLiteral, int | itir.InfinityLiteral],
    true_branch_type: ts.FieldType,
    false_branch_type: ts.FieldType,
    accessed_domain: dict[common.Dimension, tuple[int, int]],
) -> itir.Expr:
    """A `concat_where` on `K` with domains inferred from `accessed_domain`."""
    testee = im.concat_where(
        im.domain(common.GridType.UNSTRUCTURED, {K: cond_range}),
        im.ref("a", true_branch_type),
        im.ref("b", false_branch_type),
    )
    return _infer(testee, accessed_domain)


def test_prune_to_branch_that_lacks_a_dimension():
    """The surviving branch is broadcast to the dimensions of the `concat_where`.

    Replacing the `concat_where` by the branch alone would turn a two-dimensional expression
    into a one-dimensional one.
    """
    testee = _concat_where(
        (itir.InfinityLiteral.NEGATIVE, 0),
        vertex_k_field,
        k_field,
        accessed_domain={Vertex: (0, 10), K: (0, 10)},
    )

    actual = prune_empty_concat_where(testee)
    assert actual == im.broadcast(im.ref("b"), (Vertex, K))
    # `RemoveBroadcast` lowers the `broadcast` using the domain of the pruned `concat_where`
    assert actual.annex.domain == domain_utils.SymbolicDomain.from_expr(
        im.domain(common.GridType.UNSTRUCTURED, {Vertex: (0, 10), K: (0, 10)})
    )


def test_prune_never_selected_branch_when_no_branch_has_the_concat_dimension():
    """A never-selected branch must be pruned even if no branch has the concat dimension.

    `a` is selected where `K < 0`, i.e. nowhere within the accessed domain `K: [0, 10)`, so the
    expression is `b`. Since neither branch has the `K` dimension, neither branch's inferred
    domain has a `K` range — the emptiness of the region selecting `a` is only visible in the
    domain of the `concat_where` itself.
    """
    testee = _concat_where(
        (itir.InfinityLiteral.NEGATIVE, 0),
        vertex_field,
        vertex_field,
        accessed_domain={Vertex: (0, 10), K: (0, 10)},
    )

    actual = prune_empty_concat_where(testee)
    assert actual == im.broadcast(im.ref("b"), (Vertex, K))
    assert actual.annex.domain == domain_utils.SymbolicDomain.from_expr(
        im.domain(common.GridType.UNSTRUCTURED, {Vertex: (0, 10), K: (0, 10)})
    )


def test_no_prune_of_selected_branch_that_lacks_the_concat_dimension():
    """A branch lacking the concat dimension can still be selected via the implicit broadcast.

    The condition `K < 0` overlaps the accessed domain `K: [-5, 10)`, so `a` — implicitly
    broadcast to `(Vertex, K)` — is selected on `K: [-5, 0)` and must not be pruned, even
    though `a`'s own inferred domain has no `K` range. Note that `b` is only selected (and
    hence accessed) on `K: [0, 10)`, which is disjoint from the condition; that must not be
    mistaken for `a` being never selected.
    """
    testee = _concat_where(
        (itir.InfinityLiteral.NEGATIVE, 0),
        vertex_field,
        vertex_k_field,
        accessed_domain={Vertex: (0, 10), K: (-5, 10)},
    )

    assert prune_empty_concat_where(testee) == testee


def test_prune_equal_branches_that_lack_the_concat_dimension():
    """Equal branches are pruned, broadcasting if they do not have the concat dimension."""
    testee = im.concat_where(
        im.domain(common.GridType.UNSTRUCTURED, {K: (2, itir.InfinityLiteral.POSITIVE)}),
        im.ref("a", vertex_field),
        im.ref("a", vertex_field),
    )
    testee = _infer(testee, {Vertex: (0, 10), K: (0, 10)})

    assert prune_empty_concat_where(testee) == im.broadcast(im.ref("a"), (Vertex, K))
