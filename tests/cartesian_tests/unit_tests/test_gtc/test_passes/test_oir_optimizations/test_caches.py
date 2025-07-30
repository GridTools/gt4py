# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.cartesian.gtc.common import AxisBound, LevelMarker, LoopOrder
from gt4py.cartesian.gtc.oir import IJCache, KCache
from gt4py.cartesian.gtc.passes.oir_optimizations.caches import (
    FillFlushToLocalKCaches,
    IJCacheDetection,
    KCacheDetection,
    PruneKCacheFills,
    PruneKCacheFlushes,
)

from ...oir_utils import (
    AssignStmtFactory,
    HorizontalExecutionFactory,
    IJCacheFactory,
    IntervalFactory,
    KCacheFactory,
    LocalScalarFactory,
    StencilFactory,
    TemporaryFactory,
    VerticalLoopFactory,
    VerticalLoopSectionFactory,
)


def test_ij_cache_detection():
    testee = StencilFactory(
        vertical_loops=[
            VerticalLoopFactory(
                sections__0__horizontal_executions=[
                    HorizontalExecutionFactory(
                        body=[AssignStmtFactory(left__name="tmp1", right__offset__i=1)]
                    ),
                    HorizontalExecutionFactory(
                        body=[
                            AssignStmtFactory(
                                left__name="tmp2", right__name="tmp1", right__offset__j=1
                            )
                        ]
                    ),
                    HorizontalExecutionFactory(
                        body=[
                            AssignStmtFactory(
                                left__name="baz", right__name="tmp2", right__offset__k=1
                            ),
                            AssignStmtFactory(left__name="tmp3", right__name="baz"),
                            AssignStmtFactory(left__name="foo", right__name="tmp3"),
                            AssignStmtFactory(left__name="tmp4", right__name="tmp3"),
                        ]
                    ),
                ],
                caches=[IJCacheFactory(name="tmp3")],
            ),
            VerticalLoopFactory(
                sections__0__horizontal_executions__0__body=[
                    AssignStmtFactory(left__name="baz", right__name="tmp4")
                ]
            ),
        ],
        declarations=[
            TemporaryFactory(name="tmp1"),
            TemporaryFactory(name="tmp2"),
            TemporaryFactory(name="tmp3"),
            TemporaryFactory(name="tmp4"),
        ],
    )
    transformed = IJCacheDetection().visit(testee)
    caches = transformed.vertical_loops[0].caches
    assert len(caches) == 2
    assert {cache.name for cache in caches} == {"tmp1", "tmp3"}
    assert all(isinstance(cache, IJCache) for cache in caches)
    assert not transformed.vertical_loops[1].caches


def test_k_cache_detection_basic():
    testee = VerticalLoopFactory(
        loop_order=LoopOrder.FORWARD,
        sections__0__horizontal_executions__0__body=[
            AssignStmtFactory(left__name="foo", right__name="foo", right__offset__k=1),
            AssignStmtFactory(left__name="bar", right__name="foo", right__offset__k=-1),
            AssignStmtFactory(
                left__name="baz", right__name="baz", right__offset__i=1, right__offset__k=1
            ),
            AssignStmtFactory(
                left__name="foo", right__name="baz", right__offset__j=1, right__offset__k=-1
            ),
        ],
    )
    transformed = KCacheDetection().visit(testee)
    assert {c.name for c in transformed.caches} == {"foo"}
    assert all(isinstance(cache, KCache) for cache in transformed.caches)


def test_k_cache_detection_single_access_point():
    testee = VerticalLoopFactory(
        loop_order=LoopOrder.FORWARD,
        sections__0__horizontal_executions=[
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="foo", right__name="bar")]
            ),
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="bar", right__name="baz", right__offset__k=1)]
            ),
        ],
    )
    transformed = KCacheDetection().visit(testee)
    assert not transformed.caches


def test_prune_k_cache_fills_forward():
    testee = VerticalLoopFactory(
        loop_order=LoopOrder.FORWARD,
        sections=[
            VerticalLoopSectionFactory(
                horizontal_executions__0__body=[
                    AssignStmtFactory(left__name="foo"),
                    AssignStmtFactory(left__name="bar"),
                    AssignStmtFactory(left__name="barbaz"),
                    AssignStmtFactory(left__name="barbaz"),
                ],
                interval__end=AxisBound.from_start(1),
            ),
            VerticalLoopSectionFactory(
                horizontal_executions__0__body=[
                    AssignStmtFactory(left__name="foo", right__name="foo", right__offset__k=1),
                    AssignStmtFactory(left__name="bar", right__name="bar"),
                    AssignStmtFactory(left__name="baz", right__name="baz", right__offset__k=-1),
                    AssignStmtFactory(left__name="barbaz", right__name="bar"),
                    AssignStmtFactory(
                        left__name="barbaz", right__name="barbaz", right__offset__k=-1
                    ),
                ],
                interval__start=AxisBound.from_start(1),
                interval__end=AxisBound.from_end(-1),
            ),
            VerticalLoopSectionFactory(
                horizontal_executions__0__body=[
                    AssignStmtFactory(left__name="foo"),
                    AssignStmtFactory(left__name="bar"),
                    AssignStmtFactory(left__name="baz"),
                    AssignStmtFactory(left__name="barbaz"),
                    AssignStmtFactory(left__name="barbaz"),
                ],
                interval__start=AxisBound.from_end(-1),
            ),
        ],
        caches=[
            KCacheFactory(name="foo", fill=True),
            KCacheFactory(name="bar", fill=True),
            KCacheFactory(name="baz", fill=True),
            KCacheFactory(name="barbaz", fill=True),
        ],
    )
    transformed = PruneKCacheFills().visit(testee)
    cache_dict = {c.name: c for c in transformed.caches}
    assert cache_dict["foo"].fill
    assert cache_dict["bar"].fill
    assert not cache_dict["baz"].fill
    assert not cache_dict["barbaz"].fill


def test_prune_k_cache_fills_forward_with_reads_outside_interval():
    testee = VerticalLoopFactory(
        loop_order=LoopOrder.FORWARD,
        sections__0=VerticalLoopSectionFactory(
            horizontal_executions__0__body=[
                AssignStmtFactory(left__name="foo", right__name="foo", right__offset__k=-1)
            ],
            interval__start=AxisBound.from_start(1),
        ),
        caches=[KCacheFactory(name="foo", fill=True)],
    )
    transformed = PruneKCacheFills().visit(testee)
    cache_dict = {c.name: c for c in transformed.caches}
    assert cache_dict["foo"].fill


def test_prune_k_cache_fills_backward():
    testee = VerticalLoopFactory(
        loop_order=LoopOrder.BACKWARD,
        sections=[
            VerticalLoopSectionFactory(
                horizontal_executions__0__body=[
                    AssignStmtFactory(left__name="foo"),
                    AssignStmtFactory(left__name="bar"),
                    AssignStmtFactory(left__name="baz"),
                    AssignStmtFactory(left__name="barbaz"),
                    AssignStmtFactory(left__name="barbaz"),
                ],
                interval__start=AxisBound.from_end(-1),
            ),
            VerticalLoopSectionFactory(
                horizontal_executions__0__body=[
                    AssignStmtFactory(left__name="foo", right__name="foo", right__offset__k=-1),
                    AssignStmtFactory(left__name="bar", right__name="bar"),
                    AssignStmtFactory(left__name="baz", right__name="baz", right__offset__k=1),
                    AssignStmtFactory(left__name="barbaz", right__name="bar"),
                    AssignStmtFactory(
                        left__name="barbaz", right__name="barbaz", right__offset__k=1
                    ),
                ],
                interval__start=AxisBound.from_start(1),
                interval__end=AxisBound.from_end(-1),
            ),
            VerticalLoopSectionFactory(
                horizontal_executions__0__body=[
                    AssignStmtFactory(left__name="foo"),
                    AssignStmtFactory(left__name="bar"),
                    AssignStmtFactory(left__name="baz"),
                    AssignStmtFactory(left__name="barbaz"),
                    AssignStmtFactory(left__name="barbaz"),
                ],
                interval__end=AxisBound.from_start(1),
            ),
        ],
        caches=[
            KCacheFactory(name="foo", fill=True),
            KCacheFactory(name="bar", fill=True),
            KCacheFactory(name="baz", fill=True),
            KCacheFactory(name="barbaz", fill=True),
        ],
    )
    transformed = PruneKCacheFills().visit(testee)
    cache_dict = {c.name: c for c in transformed.caches}
    assert cache_dict["foo"].fill
    assert cache_dict["bar"].fill
    assert not cache_dict["baz"].fill
    assert not cache_dict["barbaz"].fill


def test_prune_k_cache_flushes():
    testee = StencilFactory(
        vertical_loops=[
            VerticalLoopFactory(
                loop_order=LoopOrder.FORWARD,
                sections__0__horizontal_executions__0__body=[
                    AssignStmtFactory(left__name="foo", right__name="foo", right__offset__k=1),
                    AssignStmtFactory(left__name="bar", right__name="baz", right__offset__k=1),
                    AssignStmtFactory(left__name="tmp1", right__name="tmp1", right__offset__k=1),
                    AssignStmtFactory(left__name="tmp2", right__name="tmp2", right__offset__k=1),
                ],
                caches=[
                    KCacheFactory(name="foo", flush=True),
                    KCacheFactory(name="bar", flush=True),
                    KCacheFactory(name="baz", flush=True),
                    KCacheFactory(name="tmp1", flush=True),
                    KCacheFactory(name="tmp2", flush=True),
                ],
            ),
            VerticalLoopFactory(
                loop_order=LoopOrder.FORWARD,
                sections__0__horizontal_executions__0__body=[
                    AssignStmtFactory(left__name="bar", right__name="bar", right__offset__k=1),
                    AssignStmtFactory(left__name="tmp1", right__name="tmp1", right__offset__k=1),
                ],
            ),
        ],
        declarations=[TemporaryFactory(name="tmp1"), TemporaryFactory(name="tmp2")],
    )
    transformed = PruneKCacheFlushes().visit(testee)
    cache_dict = {c.name: c for c in transformed.vertical_loops[0].caches}
    assert cache_dict["foo"].flush
    assert cache_dict["bar"].flush
    assert not cache_dict["baz"].flush
    assert cache_dict["tmp1"].flush
    assert not cache_dict["tmp2"].flush


def test_fill_to_local_k_caches_basic_forward():
    testee = StencilFactory(
        vertical_loops=[
            VerticalLoopFactory(
                loop_order=LoopOrder.FORWARD,
                sections=[
                    VerticalLoopSectionFactory(
                        interval=IntervalFactory(end=AxisBound(level=LevelMarker.END, offset=-1)),
                        horizontal_executions__0__body__0=AssignStmtFactory(
                            left__name="foo", right__name="foo", right__offset__k=1
                        ),
                    ),
                    VerticalLoopSectionFactory(
                        interval=IntervalFactory(start=AxisBound(level=LevelMarker.END, offset=-1)),
                        horizontal_executions__0__body__0=AssignStmtFactory(
                            left__name="foo", right__name="foo"
                        ),
                    ),
                ],
                caches=[KCacheFactory(name="foo", fill=True, flush=False)],
            )
        ]
    )
    transformed = FillFlushToLocalKCaches().visit(testee)
    vertical_loop = transformed.vertical_loops[0]

    assert len(vertical_loop.caches) == 1, "wrong number of caches"
    assert not vertical_loop.caches[0].fill, "filling cache was not removed"
    assert not vertical_loop.caches[0].flush, "cache suddenly flushes"

    cache_name = vertical_loop.caches[0].name
    assert cache_name != "foo", "cache name must not be the same as filling field"
    assert transformed.declarations[0].name == cache_name, "cache field not found in temporaries"

    assert len(vertical_loop.sections) == 2, "number of vertical sections has changed"

    assert len(vertical_loop.sections[0].horizontal_executions[0].body) == 2, (
        "no or too many fill stmts introduced?"
    )
    assert vertical_loop.sections[0].horizontal_executions[0].body[0].left.name == cache_name, (
        "wrong fill destination"
    )
    assert vertical_loop.sections[0].horizontal_executions[0].body[0].right.name == "foo", (
        "wrong fill source"
    )
    assert (
        vertical_loop.sections[0].horizontal_executions[0].body[0].left.offset.k
        == vertical_loop.sections[0].horizontal_executions[0].body[0].right.offset.k
        == 1
    ), "wrong fill offset"
    assert vertical_loop.sections[0].horizontal_executions[0].body[1].left.name == cache_name, (
        "wrong field name in cache access"
    )
    assert vertical_loop.sections[0].horizontal_executions[0].body[1].right.name == cache_name, (
        "wrong field name in cache access"
    )
    assert vertical_loop.sections[0].horizontal_executions[0].body[1].right.offset.k == 1, (
        "wrong offset in cache access"
    )
    assert len(vertical_loop.sections[1].horizontal_executions[0].body) == 1, (
        "too many fill stmts introduced?"
    )
    assert vertical_loop.sections[1].horizontal_executions[0].body[0].left.name == cache_name, (
        "wrong field name in cache access"
    )
    assert vertical_loop.sections[1].horizontal_executions[0].body[0].right.name == cache_name, (
        "wrong field name in cache access"
    )
    assert vertical_loop.sections[1].horizontal_executions[0].body[0].right.offset.k == 0, (
        "wrong offset in cache access"
    )


def test_fill_to_local_k_caches_basic_backward():
    testee = StencilFactory(
        vertical_loops=[
            VerticalLoopFactory(
                loop_order=LoopOrder.BACKWARD,
                sections=[
                    VerticalLoopSectionFactory(
                        interval=IntervalFactory(
                            start=AxisBound(level=LevelMarker.START, offset=1)
                        ),
                        horizontal_executions__0__body__0=AssignStmtFactory(
                            left__name="foo", right__name="foo", right__offset__k=-1
                        ),
                    ),
                    VerticalLoopSectionFactory(
                        interval=IntervalFactory(end=AxisBound(level=LevelMarker.START, offset=1)),
                        horizontal_executions__0__body__0=AssignStmtFactory(
                            left__name="foo", right__name="foo"
                        ),
                    ),
                ],
                caches=[KCacheFactory(name="foo", fill=True, flush=False)],
            )
        ]
    )
    transformed = FillFlushToLocalKCaches().visit(testee)
    vertical_loop = transformed.vertical_loops[0]

    assert len(vertical_loop.caches) == 1, "wrong number of caches"
    assert not vertical_loop.caches[0].fill, "filling cache was not removed"
    assert not vertical_loop.caches[0].flush, "cache suddenly flushes"

    cache_name = vertical_loop.caches[0].name
    assert cache_name != "foo", "cache name must not be the same as filling field"
    assert transformed.declarations[0].name == cache_name, "cache field not found in temporaries"

    assert len(vertical_loop.sections) == 2, "number of vertical sections has changed"

    assert len(vertical_loop.sections[0].horizontal_executions[0].body) == 2, (
        "no or too many fill stmts introduced?"
    )
    assert vertical_loop.sections[0].horizontal_executions[0].body[0].left.name == cache_name, (
        "wrong fill destination"
    )
    assert vertical_loop.sections[0].horizontal_executions[0].body[0].right.name == "foo", (
        "wrong fill source"
    )
    assert (
        vertical_loop.sections[0].horizontal_executions[0].body[0].left.offset.k
        == vertical_loop.sections[0].horizontal_executions[0].body[0].right.offset.k
        == -1
    ), "wrong fill offset"
    assert vertical_loop.sections[0].horizontal_executions[0].body[1].left.name == cache_name, (
        "wrong field name in cache access"
    )
    assert vertical_loop.sections[0].horizontal_executions[0].body[1].right.name == cache_name, (
        "wrong field name in cache access"
    )
    assert vertical_loop.sections[0].horizontal_executions[0].body[1].right.offset.k == -1, (
        "wrong offset in cache access"
    )
    assert len(vertical_loop.sections[1].horizontal_executions[0].body) == 1, (
        "too many fill stmts introduced?"
    )
    assert vertical_loop.sections[1].horizontal_executions[0].body[0].left.name == cache_name, (
        "wrong field name in cache access"
    )
    assert vertical_loop.sections[1].horizontal_executions[0].body[0].right.name == cache_name, (
        "wrong field name in cache access"
    )
    assert vertical_loop.sections[1].horizontal_executions[0].body[0].right.offset.k == 0, (
        "wrong offset in cache access"
    )


def test_fill_to_local_k_caches_section_splitting_forward():
    testee = StencilFactory(
        vertical_loops=[
            VerticalLoopFactory(
                loop_order=LoopOrder.FORWARD,
                sections=[
                    VerticalLoopSectionFactory(
                        interval=IntervalFactory(end=AxisBound(level=LevelMarker.END, offset=-1)),
                        horizontal_executions=[
                            HorizontalExecutionFactory(
                                body=[
                                    AssignStmtFactory(
                                        left__name="foo", right__name="foo", right__offset__k=0
                                    ),
                                    AssignStmtFactory(
                                        left__name="foo", right__name="foo", right__offset__k=1
                                    ),
                                ],
                                declarations=[LocalScalarFactory()],
                            )
                        ],
                    ),
                    VerticalLoopSectionFactory(
                        interval=IntervalFactory(start=AxisBound(level=LevelMarker.END, offset=-1)),
                        horizontal_executions__0__body__0=AssignStmtFactory(
                            left__name="foo", right__name="foo"
                        ),
                    ),
                ],
                caches=[KCacheFactory(name="foo", fill=True, flush=False)],
            )
        ]
    )
    transformed = FillFlushToLocalKCaches().visit(testee)
    vertical_loop = transformed.vertical_loops[0]
    assert len(vertical_loop.sections) == 3, "wrong number of vertical sections"
    assert (
        vertical_loop.sections[0].interval.start.level
        == vertical_loop.sections[0].interval.end.level
        == vertical_loop.sections[1].interval.start.level
        == LevelMarker.START
        and vertical_loop.sections[1].interval.end.level
        == vertical_loop.sections[2].interval.start.level
        == vertical_loop.sections[2].interval.end.level
        == LevelMarker.END
    ), "wrong interval levels in split sections"
    assert (
        vertical_loop.sections[0].interval.start.offset == 0
        and vertical_loop.sections[0].interval.end.offset
        == vertical_loop.sections[1].interval.start.offset
        == 1
        and vertical_loop.sections[1].interval.end.offset
        == vertical_loop.sections[2].interval.start.offset
        == -1
        and vertical_loop.sections[2].interval.end.offset == 0
    ), "wrong interval offsets in split sections"
    assert len(vertical_loop.sections[0].horizontal_executions[0].body) == 4, (
        "wrong number of fill stmts"
    )
    assert len(vertical_loop.sections[1].horizontal_executions[0].body) == 3, (
        "wrong number of fill stmts"
    )
    assert len(vertical_loop.sections[2].horizontal_executions[0].body) == 1, (
        "wrong number of fill stmts"
    )


def test_fill_to_local_k_caches_section_splitting_backward():
    testee = StencilFactory(
        vertical_loops=[
            VerticalLoopFactory(
                loop_order=LoopOrder.BACKWARD,
                sections=[
                    VerticalLoopSectionFactory(
                        interval=IntervalFactory(
                            start=AxisBound(level=LevelMarker.START, offset=1)
                        ),
                        horizontal_executions__0__body=[
                            AssignStmtFactory(
                                left__name="foo", right__name="foo", right__offset__k=0
                            ),
                            AssignStmtFactory(
                                left__name="foo", right__name="foo", right__offset__k=-1
                            ),
                        ],
                    ),
                    VerticalLoopSectionFactory(
                        interval=IntervalFactory(end=AxisBound(level=LevelMarker.START, offset=1)),
                        horizontal_executions__0__body__0=AssignStmtFactory(
                            left__name="foo", right__name="foo"
                        ),
                    ),
                ],
                caches=[KCacheFactory(name="foo", fill=True, flush=False)],
            )
        ]
    )
    transformed = FillFlushToLocalKCaches().visit(testee)
    vertical_loop = transformed.vertical_loops[0]
    assert len(vertical_loop.sections) == 3, "wrong number of vertical sections"
    assert (
        vertical_loop.sections[0].interval.start.level
        == vertical_loop.sections[0].interval.end.level
        == vertical_loop.sections[1].interval.end.level
        == LevelMarker.END
        and vertical_loop.sections[1].interval.start.level
        == vertical_loop.sections[2].interval.start.level
        == vertical_loop.sections[2].interval.end.level
        == LevelMarker.START
    ), "wrong interval levels in split sections"
    assert (
        vertical_loop.sections[0].interval.end.offset == 0
        and vertical_loop.sections[0].interval.start.offset
        == vertical_loop.sections[1].interval.end.offset
        == -1
        and vertical_loop.sections[1].interval.start.offset
        == vertical_loop.sections[2].interval.end.offset
        == 1
        and vertical_loop.sections[2].interval.start.offset == 0
    ), "wrong interval offsets in split sections"
    assert len(vertical_loop.sections[0].horizontal_executions[0].body) == 4, (
        "wrong number of fill stmts"
    )
    assert len(vertical_loop.sections[1].horizontal_executions[0].body) == 3, (
        "wrong number of fill stmts"
    )
    assert len(vertical_loop.sections[2].horizontal_executions[0].body) == 1, (
        "wrong number of fill stmts"
    )


def test_flush_to_local_k_caches_basic():
    testee = StencilFactory(
        vertical_loops=[
            VerticalLoopFactory(
                loop_order=LoopOrder.FORWARD,
                sections=[
                    VerticalLoopSectionFactory(
                        interval=IntervalFactory(end=AxisBound(level=LevelMarker.START, offset=1)),
                        horizontal_executions__0__body__0=AssignStmtFactory(
                            left__name="foo", right__name="foo"
                        ),
                    ),
                    VerticalLoopSectionFactory(
                        interval=IntervalFactory(
                            start=AxisBound(level=LevelMarker.START, offset=1)
                        ),
                        horizontal_executions__0__body__0=AssignStmtFactory(
                            left__name="foo", right__name="foo", right__offset__k=-1
                        ),
                    ),
                ],
                caches=[KCacheFactory(name="foo", fill=False, flush=True)],
            )
        ]
    )
    transformed = FillFlushToLocalKCaches().visit(testee)
    vertical_loop = transformed.vertical_loops[0]

    assert len(vertical_loop.caches) == 1, "wrong number of caches"
    assert not vertical_loop.caches[0].fill, "cache suddenly fills"
    assert not vertical_loop.caches[0].flush, "flushing cache was not removed"

    cache_name = vertical_loop.caches[0].name
    assert cache_name != "foo", "cache name must not be the same as flushing field"
    assert transformed.declarations[0].name == cache_name, "cache field not found in temporaries"

    assert len(vertical_loop.sections) == 2, "number of vertical sections has changed"

    assert len(vertical_loop.sections[0].horizontal_executions[0].body) == 2, (
        "no or too many flush stmts introduced?"
    )
    assert vertical_loop.sections[0].horizontal_executions[0].body[0].left.name == cache_name, (
        "wrong field name in cache access"
    )
    assert vertical_loop.sections[0].horizontal_executions[0].body[0].right.name == cache_name, (
        "wrong field name in cache access"
    )
    assert vertical_loop.sections[0].horizontal_executions[0].body[0].right.offset.k == 0, (
        "wrong offset in cache access"
    )
    assert vertical_loop.sections[0].horizontal_executions[0].body[1].left.name == "foo", (
        "wrong flush source"
    )
    assert vertical_loop.sections[0].horizontal_executions[0].body[1].right.name == cache_name, (
        "wrong flush destination"
    )
    assert (
        vertical_loop.sections[0].horizontal_executions[0].body[1].left.offset.k
        == vertical_loop.sections[0].horizontal_executions[0].body[1].right.offset.k
        == 0
    ), "wrong flush offset"
    assert len(vertical_loop.sections[1].horizontal_executions[0].body) == 2, (
        "no or too many flush stmts introduced?"
    )
    assert vertical_loop.sections[1].horizontal_executions[0].body[0].left.name == cache_name, (
        "wrong field name in cache access"
    )
    assert vertical_loop.sections[1].horizontal_executions[0].body[0].right.name == cache_name, (
        "wrong field name in cache access"
    )
    assert vertical_loop.sections[1].horizontal_executions[0].body[0].right.offset.k == -1, (
        "wrong offset in cache access"
    )
    assert vertical_loop.sections[1].horizontal_executions[0].body[1].left.name == "foo", (
        "wrong flush source"
    )
    assert vertical_loop.sections[1].horizontal_executions[0].body[1].right.name == cache_name, (
        "wrong flush destination"
    )
    assert vertical_loop.sections[1].horizontal_executions[0].body[1].right.offset.k == 0, (
        "wrong flush offset"
    )


def test_fill_flush_to_local_k_caches_basic_forward():
    testee = StencilFactory(
        vertical_loops=[
            VerticalLoopFactory(
                loop_order=LoopOrder.FORWARD,
                sections__0__horizontal_executions__0__body=[
                    AssignStmtFactory(left__name="foo", right__name="foo")
                ],
                caches=[KCacheFactory(name="foo", fill=True, flush=True)],
            )
        ]
    )
    transformed = FillFlushToLocalKCaches().visit(testee)
    vertical_loop = transformed.vertical_loops[0]

    assert len(vertical_loop.caches) == 1, "wrong number of caches"
    assert not vertical_loop.caches[0].fill, "filling cache was not removed"
    assert not vertical_loop.caches[0].flush, "flushing cache was not removed"

    cache_name = vertical_loop.caches[0].name
    assert cache_name != "foo", "cache name must not be the same as filling field"
    assert transformed.declarations[0].name == cache_name, "cache field not found in temporaries"

    assert len(vertical_loop.sections) == 1, "number of vertical sections has changed"

    body = vertical_loop.sections[0].horizontal_executions[0].body
    assert len(body) == 3, "no or too many fill/flush stmts introduced?"
    assert body[0].left.name == cache_name, "wrong fill destination"
    assert body[0].right.name == "foo", "wrong fill source"
    assert body[0].left.offset.k == body[0].right.offset.k == 0, "wrong fill offset"
    assert body[1].left.name == cache_name, "wrong field name in cache access"
    assert body[1].right.name == cache_name, "wrong field name in cache access"
    assert body[1].left.offset.k == body[1].right.offset.k == 0, "wrong offset in cache access"
    assert body[2].left.name == "foo", "wrong flush destination"
    assert body[2].right.name == cache_name, "wrong flush source"
    assert body[2].left.offset.k == body[2].right.offset.k == 0, "wrong flush offset"
