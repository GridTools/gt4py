# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from gt4py.next import common, constructors
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms.prune_empty_concat_where import prune_empty_concat_where
from gt4py.next.iterator.transforms.concat_where import canonicalize_domain_argument
from gt4py.next.iterator.transforms.infer_domain import infer_expr
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, domain_utils
from gt4py.next.type_system import type_info, type_specifications as ts

Vertex = common.Dimension(value="Vertex", kind=common.DimensionKind.HORIZONTAL)
Edge = common.Dimension(value="Edge", kind=common.DimensionKind.HORIZONTAL)
V2EDim = common.Dimension(value="V2E", kind=common.DimensionKind.LOCAL)
K = common.Dimension(value="K", kind=common.DimensionKind.VERTICAL)

float64 = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
vertex_k_field = ts.FieldType(dims=[Vertex, K], dtype=float64)
vertex_field = ts.FieldType(dims=[Vertex], dtype=float64)
k_field = ts.FieldType(dims=[K], dtype=float64)
edge_field = ts.FieldType(dims=[Edge], dtype=float64)


def _infer(
    testee: itir.Expr,
    accessed_domain: dict[common.Dimension, tuple],
    offset_provider: common.OffsetProvider | None = None,
) -> itir.Expr:
    testee = canonicalize_domain_argument(testee)
    testee, _ = infer_expr(
        testee,
        domain_utils.SymbolicDomain.from_expr(
            im.domain(common.GridType.UNSTRUCTURED, accessed_domain)
        ),
        offset_provider=offset_provider or {},
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
        # (
        #     {Vertex: ("v0", "v1")},
        #     {Vertex: ("v0", "v1")},
        #     ("a", vertex_field),
        #     ("b", vertex_field),
        #     "a",
        # ),
        # cond is empty
        ({Vertex: (0, 10)}, {Vertex: (0, 0)}, ("a", vertex_field), ("b", vertex_field), "b"),
        (
            {Vertex: (0, 10), K: (0, 10)},
            {K: (0, 0)},
            ("a", vertex_k_field),
            ("b", vertex_k_field),
            "b",
        ),
        # (
        #     {Vertex: ("v0", "v0")},
        #     {Vertex: ("v0", "v0")},
        #     ("a", vertex_field),
        #     ("b", vertex_field),
        #     "b",
        # ),
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
        # (
        #     {Vertex: ("v0", "v1")},
        #     {Vertex: ("v0", "v2")},
        #     ("a", vertex_field),
        #     ("b", vertex_field),
        #     None,
        # ),
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
        #  broadcast to `(Vertex, K)` — is selected on `K: [0, 5)`, even though `a`'s own
        #  inferred domain has no `K` range that could show this
        (
            {Vertex: (0, 10), K: (0, 10)},
            {K: (itir.InfinityLiteral.NEGATIVE, 5)},
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
        # branches that are equal after the implicit broadcast is made explicit; the domains
        #  inside the surviving branch must be reinferred on the full domain of the
        #  `concat_where`, e.g. the `a` inside the `broadcast` would otherwise keep the domain
        #  `K: [2, 10)` of the branch instance it originates from
        (
            {Vertex: (0, 10), K: (0, 10)},
            {K: (2, itir.InfinityLiteral.POSITIVE)},
            ("a", k_field),
            im.broadcast(im.ref("a", k_field), (Vertex, K)),
            im.broadcast(im.ref("a"), (Vertex, K)),
        ),
    ],
)
def test_prune_concat_where(accessed_domain, cond_domain, true_branch, false_branch, expected):
    def branch_expr(branch: tuple[str, ts.FieldType] | itir.Expr) -> itir.Expr:
        return im.ref(*branch) if isinstance(branch, tuple) else branch

    testee = im.concat_where(
        im.domain(common.GridType.UNSTRUCTURED, cond_domain),
        branch_expr(true_branch),
        branch_expr(false_branch),
    )
    testee = _infer(testee, accessed_domain)

    if expected is None:
        expected = testee
    expected = im.ensure_expr(expected)
    expected = canonicalize_domain_argument(expected)
    expected = InlineLambdas.apply(expected)

    actual = prune_empty_concat_where(testee, offset_provider={})
    # on pruning, the domain annex is populated from the pruned `concat_where`
    #  (`RemoveBroadcast` relies on it)
    assert actual.annex.domain == domain_utils.SymbolicDomain.from_expr(
        im.domain(common.GridType.UNSTRUCTURED, accessed_domain)
    )
    if cpm.is_call_to(actual, "broadcast"):
        # the domains inside the pruned branch are inferred from the full domain of the
        #  `concat_where`, restricted to the respective expression's dimensions
        inner = actual.args[0]
        assert inner.annex.domain == domain_utils.SymbolicDomain.from_expr(
            im.domain(
                common.GridType.UNSTRUCTURED,
                {
                    dim: bounds
                    for dim, bounds in accessed_domain.items()
                    if dim in type_info.extract_dims(inner.type)
                },
            )
        )
    actual = InlineLambdas.apply(actual)
    assert actual == expected


def test_prune_equal_branches_containing_unstructured_shift():
    """
    Pruning equal branches reinfers the domains of the surviving branch on the full domain of
    the `concat_where`, which requires the actual offset provider when the branch contains an
    unstructured shift (regression test: the pass previously used an empty offset provider,
    failing on the `V2E` translation during the re-inference).
    """
    offset_provider = {
        "V2E": constructors.as_connectivity(
            domain={Vertex: 1, V2EDim: 2},
            codomain=Edge,
            data=np.array([[0, 1]], dtype=np.int32),
        )
    }
    stencil = im.lambda_("it")(im.deref(im.shift("V2E", 0)("it")))
    branch = im.as_fieldop(stencil)(im.ref("e", edge_field))
    accessed_domain = {Vertex: (0, 1), K: (0, 10)}

    testee = im.concat_where(
        im.domain(common.GridType.UNSTRUCTURED, {K: (2, itir.InfinityLiteral.POSITIVE)}),
        branch,
        branch,
    )
    testee = _infer(testee, accessed_domain, offset_provider)

    actual = prune_empty_concat_where(testee, offset_provider=offset_provider)

    expected = im.broadcast(
        im.as_fieldop(stencil, im.domain(common.GridType.UNSTRUCTURED, {Vertex: (0, 1)}))(
            im.ref("e")
        ),
        (Vertex, K),
    )
    assert actual == expected
    assert actual.annex.domain == domain_utils.SymbolicDomain.from_expr(
        im.domain(common.GridType.UNSTRUCTURED, accessed_domain)
    )
