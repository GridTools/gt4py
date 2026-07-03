# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np
from gt4py.next import common
from gt4py.next.ffront import fbuiltins
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import domain_utils, ir_makers as im
from gt4py.next import common, constructors

I = common.Dimension("I")
J = common.Dimension("J")
K = common.Dimension("J", kind=common.DimensionKind.VERTICAL)
Vertex = common.Dimension("Vertex")
Edge = common.Dimension("Edge")
V2EDim = common.Dimension("V2E", kind=common.DimensionKind.LOCAL)
E2VDim = common.Dimension("E2V", kind=common.DimensionKind.LOCAL)
V2VDim = common.Dimension("V2V", kind=common.DimensionKind.LOCAL)

a_range = domain_utils.SymbolicRange(0, 10)
another_range = domain_utils.SymbolicRange(5, 15)
infinity_range = domain_utils.SymbolicRange(
    itir.InfinityLiteral.NEGATIVE, itir.InfinityLiteral.POSITIVE
)
# the two next ranges are complement of each other (which is used in tests)
right_infinity_range = domain_utils.SymbolicRange(0, itir.InfinityLiteral.POSITIVE)
left_infinity_range = domain_utils.SymbolicRange(itir.InfinityLiteral.NEGATIVE, 0)


def _make_domain(ranges: dict[common.Dimension, tuple[int, int]]):
    return domain_utils.SymbolicDomain(
        grid_type=common.GridType.CARTESIAN,
        ranges={
            dim: domain_utils.SymbolicRange(start, stop) for dim, (start, stop) in ranges.items()
        },
    )


def test_domain_op_preconditions():
    domain_a = domain_utils.SymbolicDomain(
        grid_type=common.GridType.CARTESIAN,
        ranges={I: domain_utils.SymbolicRange(0, 10)},
    )
    domain_b = domain_utils.SymbolicDomain(
        grid_type=common.GridType.CARTESIAN,
        ranges={J: domain_utils.SymbolicRange(5, 15)},
    )
    with pytest.raises(AssertionError):
        domain_utils._reduce_domains(domain_a, domain_b, range_reduce_op=domain_utils._range_union)
    domain_c = domain_utils.SymbolicDomain(
        grid_type=common.GridType.UNSTRUCTURED, ranges={I: domain_utils.SymbolicRange(0, 10)}
    )
    with pytest.raises(AssertionError):
        domain_utils._reduce_domains(domain_a, domain_c, range_reduce_op=domain_utils._range_union)


def test_domain_union():
    domain0 = _make_domain({I: (0, 10), J: (0, 10)})
    domain1 = _make_domain({I: (5, 15), J: (2, 8)})
    domain2 = _make_domain({I: (3, 7), J: (4, 20)})

    # the union is the convex hull per dimension: (min of starts, max of stops)
    expected = im.domain(common.GridType.CARTESIAN, {I: (0, 15), J: (0, 20)})
    result = domain_utils.domain_union(domain0, domain1, domain2)
    assert result.as_expr() == expected


def test_domain_union_drops_empty_domains():
    # Empty domains are the union's identity element; keeping them would over-approximate (#2205).
    non_empty = domain_utils.SymbolicDomain(
        grid_type=common.GridType.CARTESIAN, ranges={I: domain_utils.SymbolicRange(0, 10)}
    )
    empty_a = domain_utils.SymbolicDomain(
        grid_type=common.GridType.CARTESIAN, ranges={I: domain_utils.SymbolicRange(10, 10)}
    )
    empty_b = domain_utils.SymbolicDomain(
        grid_type=common.GridType.CARTESIAN, ranges={I: domain_utils.SymbolicRange(11, 11)}
    )

    # a single non-empty domain among empty ones is returned unchanged
    assert domain_utils.domain_union(empty_a, non_empty, empty_b) == non_empty
    assert domain_utils.domain_union(empty_a) == empty_a


def test_domain_union_all_empty():
    # A union of only empty domains stays empty (not the convex hull `[10, 11)`).
    empty_a = domain_utils.SymbolicDomain(
        grid_type=common.GridType.CARTESIAN, ranges={I: domain_utils.SymbolicRange(10, 10)}
    )
    empty_b = domain_utils.SymbolicDomain(
        grid_type=common.GridType.CARTESIAN, ranges={I: domain_utils.SymbolicRange(11, 11)}
    )
    result = domain_utils.domain_union(empty_a, empty_b)
    assert result.empty()


def test_unstructured_translate_empty_range():
    offset_provider = {
        "V2E": constructors.as_connectivity(
            domain={Vertex: (0, 4), V2EDim: 1},
            codomain=Edge,
            data=np.asarray([0, 1, 2, 3], dtype=fbuiltins.IndexType).reshape((4, 1)),
        )
    }
    domain = domain_utils.SymbolicDomain.from_expr(
        im.domain(common.GridType.UNSTRUCTURED, {Vertex: (2, 2)})  # empty
    )
    translated = domain.translate(
        [itir.OffsetLiteral(value="V2E"), itir.OffsetLiteral(value=0)], offset_provider
    )
    assert translated.empty()
    assert set(translated.ranges.keys()) == {Edge}


def test_domain_intersection():
    domain0 = _make_domain({I: (0, 10), J: (0, 10)})
    domain1 = _make_domain({I: (5, 15), J: (2, 8)})
    domain2 = _make_domain({I: (3, 7), J: (4, 20)})

    # the intersection is per dimension: (max of starts, min of stops)
    expected = im.domain(common.GridType.CARTESIAN, {I: (5, 7), J: (4, 8)})
    result = domain_utils.domain_intersection(domain0, domain1, domain2)
    assert result.as_expr() == expected


def test_domain_union_then_intersection():
    # Cross-operation case: the union result flows into the intersection.
    domain0 = _make_domain({I: (0, 10), J: (0, 10)})
    domain1 = _make_domain({I: (5, 15), J: (2, 8)})
    domain2 = _make_domain({I: (3, 7), J: (4, 20)})

    union_result = domain_utils.domain_union(domain0, domain1)  # {I: (0, 15), J: (0, 10)}
    result = domain_utils.domain_intersection(union_result, domain2)

    # intersection of the union `{I: (0, 15), J: (0, 10)}` with `domain2`
    expected = im.domain(common.GridType.CARTESIAN, {I: (3, 7), J: (4, 10)})
    assert result.as_expr() == expected


@pytest.mark.parametrize(
    "ranges, expected",
    [
        ({I: a_range}, None),
        ({I: infinity_range}, None),
        ({I: a_range, J: right_infinity_range}, None),
        (
            {I: right_infinity_range, J: left_infinity_range},
            {I: left_infinity_range, J: right_infinity_range},
        ),
    ],
)
def test_domain_complement(ranges, expected):
    if expected is None:
        with pytest.raises(AssertionError):
            domain_utils.domain_complement(
                domain_utils.SymbolicDomain(grid_type=common.GridType.CARTESIAN, ranges=ranges)
            )
    else:
        assert domain_utils.domain_complement(
            domain_utils.SymbolicDomain(grid_type=common.GridType.CARTESIAN, ranges=ranges)
        ) == domain_utils.SymbolicDomain(grid_type=common.GridType.CARTESIAN, ranges=expected)


@pytest.mark.parametrize(
    "testee_ranges, dimensions, expected_ranges",
    [
        ({I: a_range}, [I, J], {I: a_range, J: infinity_range}),
        ({I: a_range}, [J, I], {I: a_range, J: infinity_range}),
        ({I: a_range}, [I], {I: a_range}),
        ({I: a_range}, [J], None),
        ({I: a_range, J: another_range}, [J], None),
    ],
)
def test_promote_domain(testee_ranges, dimensions, expected_ranges):
    testee = domain_utils.SymbolicDomain(grid_type=common.GridType.CARTESIAN, ranges=testee_ranges)
    if expected_ranges is None:
        with pytest.raises(AssertionError):
            domain_utils.promote_domain(testee, dimensions)
    else:
        expected = domain_utils.SymbolicDomain(
            grid_type=common.GridType.CARTESIAN, ranges=expected_ranges
        )
        promoted = domain_utils.promote_domain(testee, dimensions)
        assert promoted == expected


def test_is_finite_symbolic_range():
    assert not domain_utils.is_finite(infinity_range)
    assert not domain_utils.is_finite(left_infinity_range)
    assert not domain_utils.is_finite(right_infinity_range)
    assert domain_utils.is_finite(a_range)


@pytest.mark.parametrize(
    "ranges, expected",
    [
        ({I: a_range, J: a_range}, True),
        ({I: a_range, J: another_range}, True),
        ({I: right_infinity_range, J: a_range}, False),
        ({I: a_range, J: right_infinity_range}, False),
    ],
)
def test_is_finite_symbolic_domain(ranges, expected):
    assert (
        domain_utils.is_finite(
            domain_utils.SymbolicDomain(grid_type=common.GridType.CARTESIAN, ranges=ranges)
        )
        == expected
    )


@pytest.mark.parametrize(
    "shift_chain, expected_end_domain",
    [
        (("V2V", 0), {Vertex: (0, 4)}),
        (("V2V", 1), {Vertex: (0, 4)}),
        (("V2V", 2), {Vertex: (0, 1)}),
        (("V2V", 3), {Vertex: (1, 4)}),
        (("V2V", 0, "V2V", 3, "V2V", 0), {Vertex: (1, 4)}),
        (("V2E", 0), {Edge: (0, 4)}),
        (("V2E", 0, "E2V", 0), {Vertex: (0, 4)}),
        (("V2V", 3, "V2E", 0), {Edge: (1, 4)}),
    ],
)
def test_unstructured_translate(shift_chain, expected_end_domain):
    offset_provider = {
        "V2V": constructors.as_connectivity(
            domain={Vertex: (0, 4), V2VDim: 5},
            codomain=Vertex,
            data=np.asarray(
                [[0, 3, 0, 1, -1], [1, 2, 0, 1, 1], [2, 1, 0, 3, 2], [3, 0, 0, 3, -1]],
                dtype=fbuiltins.IndexType,
            ),
        ),
        "V2E": constructors.as_connectivity(
            domain={Vertex: (0, 4), V2EDim: 1},
            codomain=Edge,
            data=np.asarray(
                [
                    [0, 1, 2, 3],
                ],
                dtype=fbuiltins.IndexType,
            ).reshape((4, 1)),
        ),
        "E2V": constructors.as_connectivity(
            domain={Edge: (0, 4), E2VDim: 1},
            codomain=Vertex,
            data=np.asarray(
                [
                    [0, 1, 2, 3],
                ],
                dtype=fbuiltins.IndexType,
            ).reshape((4, 1)),
        ),
    }
    shift_chain = [im.ensure_offset(o) for o in shift_chain]
    expected_end_domain = im.domain(common.GridType.UNSTRUCTURED, expected_end_domain)

    init_domain = domain_utils.SymbolicDomain.from_expr(
        im.domain(common.GridType.UNSTRUCTURED, {Vertex: (0, 4)})
    )
    end_domain = init_domain.translate(shift_chain, offset_provider).as_expr()
    assert end_domain == expected_end_domain


def test_non_contiguous_domain_warning(monkeypatch):
    monkeypatch.setattr(domain_utils, "_NON_CONTIGUOUS_DOMAIN_WARNING_SKIPPED_OFFSET_TAGS", set())

    offset_provider = {
        "V2V": constructors.as_connectivity(
            domain={Vertex: (0, 100), V2VDim: 1},
            codomain=Vertex,
            data=np.asarray([0] + [99] * 99, dtype=fbuiltins.IndexType).reshape((100, 1)),
        )
    }
    shift_chain = ("V2V", 0)
    shift_chain = [im.ensure_offset(o) for o in shift_chain]
    domain = domain_utils.SymbolicDomain.from_expr(
        im.domain(common.GridType.UNSTRUCTURED, {Vertex: (0, 2)})
    )
    with pytest.warns(
        UserWarning,
        match=r"98%.*Please consider reordering your mesh.",
    ):
        domain.translate(shift_chain, offset_provider).as_expr()


def test_oob_error():
    offset_provider = {
        "V2V": constructors.as_connectivity(
            domain={Vertex: (0, 3), V2VDim: 1},
            codomain=Vertex,
            data=np.asarray([0, -1, 1], dtype=fbuiltins.IndexType).reshape((3, 1)),
        )
    }
    shift_chain = ("V2V", 0)
    shift_chain = [im.ensure_offset(o) for o in shift_chain]
    domain = domain_utils.SymbolicDomain.from_expr(
        im.domain(common.GridType.UNSTRUCTURED, {Vertex: (0, 3)})
    )
    with pytest.warns(
        UserWarning,
        match=r"out-of-bounds",
    ):
        domain.translate(shift_chain, offset_provider).as_expr()
