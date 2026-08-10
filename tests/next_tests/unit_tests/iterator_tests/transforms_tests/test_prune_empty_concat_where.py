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


def _domain(ranges):
    return domain_utils.SymbolicDomain.from_expr(im.domain(common.GridType.UNSTRUCTURED, ranges))


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
    accessed_domain = im.domain(common.GridType.UNSTRUCTURED, accessed_domain)
    testee = im.concat_where(im.domain(common.GridType.UNSTRUCTURED, cond_domain), "a", "b")
    testee = canonicalize_domain_argument(testee)
    testee, _ = infer_expr(
        testee, domain_utils.SymbolicDomain.from_expr(accessed_domain), offset_provider={}
    )

    if expected is None:
        expected = testee
    expected = im.ensure_expr(expected)
    expected = canonicalize_domain_argument(expected)
    expected = InlineLambdas.apply(expected)

    actual = prune_empty_concat_where(testee)
    actual = InlineLambdas.apply(actual)
    assert actual == expected


def _infer_concat_where(cond_range, true_branch_type, false_branch_type):
    """A `concat_where` on `K` over the domain `Vertex: [0, 10), K: [0, 10)`."""
    cond = im.domain(common.GridType.UNSTRUCTURED, {K: cond_range})
    testee = im.concat_where(cond, im.ref("a", true_branch_type), im.ref("b", false_branch_type))
    testee = canonicalize_domain_argument(testee)
    testee, _ = infer_expr(
        testee,
        _domain({Vertex: (0, 10), K: (0, 10)}),
        offset_provider={},
    )
    return testee


@pytest.mark.parametrize(
    "cond_range",
    [
        (itir.InfinityLiteral.NEGATIVE, 0),  # entirely below the domain
        (10, itir.InfinityLiteral.POSITIVE),  # entirely above it
    ],
)
def test_prune_branch_that_lacks_the_concat_dimension(cond_range):
    """A never selected branch is pruned even when it does not have the concat dimension.

    A branch's own domain is restricted to the dimensions of the branch's type, so for such
    a branch it has no ranges at all and is vacuously non-empty. Whether a branch is selected
    therefore has to be decided from the `concat_where`'s domain.
    """
    testee = _infer_concat_where(cond_range, vertex_field, vertex_k_field)

    assert prune_empty_concat_where(testee) == im.ref("b", vertex_k_field)


def test_no_prune_when_the_surviving_branch_lacks_a_dimension():
    """Pruning must not silently drop a dimension.

    The true branch is never selected, but the false branch does not have the `Vertex`
    dimension, so replacing the `concat_where` by it would change the result.
    """
    testee = _infer_concat_where((itir.InfinityLiteral.NEGATIVE, 0), vertex_k_field, k_field)

    assert prune_empty_concat_where(testee) == testee
