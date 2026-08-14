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
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, domain_utils
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
    "accessed_domain, cond_domain, true_branch, false_branch, expected",
    [
        # TODO(tehrengruber): Implement in pass and enable commented out symbolic test cases below.
        # cond spans entire accessed domain of true branch value
        (
            {Vertex: (0, 10), K: (0, 10)},
            {Vertex: (0, 10)},
            ("a", vertex_k_field),
            ("b", vertex_k_field),
            "a",
        ),
        # ({Vertex: ("v0", "v1")}, {Vertex: ("v0", "v1")}, ..., "a"),
        # cond is empty
        ({Vertex: (0, 10)}, {Vertex: (0, 0)}, ("a", vertex_field), ("b", vertex_field), "b"),
        (
            {Vertex: (0, 10), K: (0, 10)},
            {K: (0, 0)},
            ("a", vertex_k_field),
            ("b", vertex_k_field),
            "b",
        ),
        # ({Vertex: ("v0", "v0")}, {Vertex: ("v0", "v0")}, ..., "b"),
        # cond disjoint from the accessed domain:
        #  entirely below it
        (
            {Vertex: (0, 10), K: (0, 10)},
            {K: (itir.InfinityLiteral.NEGATIVE, 0)},
            ("a", vertex_k_field),
            ("b", vertex_k_field),
            "b",
        ),
        #  entirely above it
        (
            {Vertex: (0, 10), K: (0, 10)},
            {K: (10, itir.InfinityLiteral.POSITIVE)},
            ("a", vertex_k_field),
            ("b", vertex_k_field),
            "b",
        ),
        # cond covers the accessed domain:
        #  from below
        (
            {Vertex: (0, 10), K: (0, 10)},
            {K: (itir.InfinityLiteral.NEGATIVE, 10)},
            ("a", vertex_k_field),
            ("b", vertex_k_field),
            "a",
        ),
        #  from above
        (
            {Vertex: (0, 10), K: (0, 10)},
            {K: (0, itir.InfinityLiteral.POSITIVE)},
            ("a", vertex_k_field),
            ("b", vertex_k_field),
            "a",
        ),
        # cond subset of accessed domain, no transformation occurs
        ({Vertex: (0, 10)}, {Vertex: (1, 2)}, ("a", vertex_field), ("b", vertex_field), None),
        (
            {Vertex: (0, 10), K: (0, 10)},
            {Vertex: (1, 2)},
            ("a", vertex_k_field),
            ("b", vertex_k_field),
            None,
        ),
        # ({Vertex: ("v0", "v1")}, {Vertex: ("v0", "v2")}, ..., None)
        # cond subset of accessed domain, but only one half-space
        #  after canonicalization will remain
        (
            {Vertex: (0, 10)},
            {Vertex: (0, 1)},
            ("a", vertex_field),
            ("b", vertex_field),
            im.concat_where(
                im.domain(
                    common.GridType.UNSTRUCTURED, {Vertex: (1, itir.InfinityLiteral.POSITIVE)}
                ),
                "b",
                "a",
            ),
        ),
        # prune to a branch that lacks a dimension: the surviving branch is broadcast to the
        #  dimensions of the `concat_where` instead of turning the two-dimensional expression
        #  into a one-dimensional one
        (
            {Vertex: (0, 10), K: (0, 10)},
            {K: (itir.InfinityLiteral.NEGATIVE, 0)},
            ("a", vertex_k_field),
            ("b", k_field),
            im.broadcast(im.ref("b"), (Vertex, K)),
        ),
        # a never-selected branch is pruned even if no branch has the concat dimension: neither
        #  branch's inferred domain has a `K` range, so the emptiness of the region selecting
        #  `a` is only visible in the domain of the `concat_where` itself
        (
            {Vertex: (0, 10), K: (0, 10)},
            {K: (itir.InfinityLiteral.NEGATIVE, 0)},
            ("a", vertex_field),
            ("b", vertex_field),
            im.broadcast(im.ref("b"), (Vertex, K)),
        ),
        # no pruning of a selected branch that lacks the concat dimension: `a` — implicitly
        #  broadcast to `(Vertex, K)` — is selected on `K: [-5, 0)`; note that `b`'s accessed
        #  domain `K: [0, 10)` is disjoint from the cond, which must not be mistaken for `a`
        #  being never selected
        (
            {Vertex: (0, 10), K: (-5, 10)},
            {K: (itir.InfinityLiteral.NEGATIVE, 0)},
            ("a", vertex_field),
            ("b", vertex_k_field),
            None,
        ),
        # equal branches are pruned, broadcasting since they lack the concat dimension
        (
            {Vertex: (0, 10), K: (0, 10)},
            {K: (2, itir.InfinityLiteral.POSITIVE)},
            ("a", vertex_field),
            ("a", vertex_field),
            im.broadcast(im.ref("a"), (Vertex, K)),
        ),
    ],
)
def test_prune_concat_where(accessed_domain, cond_domain, true_branch, false_branch, expected):
    testee = im.concat_where(
        im.domain(common.GridType.UNSTRUCTURED, cond_domain),
        im.ref(*true_branch),
        im.ref(*false_branch),
    )
    testee = _infer(testee, accessed_domain)

    if expected is None:
        expected = testee
    expected = im.ensure_expr(expected)
    expected = canonicalize_domain_argument(expected)
    expected = InlineLambdas.apply(expected)

    actual = prune_empty_concat_where(testee)
    if not cpm.is_call_to(actual, "concat_where"):
        # on pruning, the domain annex is populated from the pruned `concat_where`
        #  (`RemoveBroadcast` relies on it)
        assert actual.annex.domain == domain_utils.SymbolicDomain.from_expr(
            im.domain(common.GridType.UNSTRUCTURED, accessed_domain)
        )
    actual = InlineLambdas.apply(actual)
    assert actual == expected
