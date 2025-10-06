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

Vertex = common.Dimension(value="Vertex", kind=common.DimensionKind.HORIZONTAL)
K = common.Dimension(value="K", kind=common.DimensionKind.VERTICAL)


@pytest.mark.parametrize(
    "accessed_domain, cond_domain, expected",
    [
        # TODO(tehrengruber): Implement and enable symbolic test cases.
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
