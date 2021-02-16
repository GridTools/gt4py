# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gtc.common import LoopOrder
from gtc.oir import IJCache, KCache
from gtc.passes.oir_optimizations.caches import (
    IJCacheDetection,
    KCacheDetection,
    PruneKCacheFills,
    PruneKCacheFlushes,
)

from ...oir_utils import (
    AssignStmtFactory,
    HorizontalExecutionFactory,
    IJCacheFactory,
    KCacheFactory,
    StencilFactory,
    TemporaryFactory,
    VerticalLoopFactory,
)


def test_ij_cache_detection():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="tmp1", right__offset__i=1)]
            ),
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="tmp2", right__name="tmp1", right__offset__j=1)]
            ),
            HorizontalExecutionFactory(
                body=[
                    AssignStmtFactory(left__name="baz", right__name="tmp2", right__offset__k=1),
                    AssignStmtFactory(left__name="tmp3", right__name="baz"),
                    AssignStmtFactory(left__name="foo", right__name="tmp3"),
                ]
            ),
        ],
        vertical_loops__0__caches=[IJCacheFactory(name="tmp3")],
        declarations=[
            TemporaryFactory(name="tmp1"),
            TemporaryFactory(name="tmp2"),
            TemporaryFactory(name="tmp3"),
        ],
    )
    transformed = IJCacheDetection().visit(testee)
    caches = transformed.vertical_loops[0].caches
    assert len(caches) == 2
    assert {cache.name for cache in caches} == {"tmp1", "tmp3"}
    assert all(isinstance(cache, IJCache) for cache in caches)


def test_k_cache_detection_basic():
    testee = VerticalLoopFactory(
        loop_order=LoopOrder.FORWARD,
        sections__0__horizontal_executions__0__body=[
            AssignStmtFactory(
                left__name="foo",
                right__name="foo",
                right__offset__k=1,
            ),
            AssignStmtFactory(
                left__name="bar",
                right__name="foo",
                right__offset__k=-1,
            ),
            AssignStmtFactory(
                left__name="baz",
                right__name="baz",
                right__offset__i=1,
                right__offset__k=1,
            ),
            AssignStmtFactory(
                left__name="foo",
                right__name="baz",
                right__offset__j=1,
                right__offset__k=-1,
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
        sections__0__horizontal_executions__0__body=[
            AssignStmtFactory(left__name="foo", right__name="foo", right__offset__k=1),
            AssignStmtFactory(left__name="bar", right__name="bar"),
            AssignStmtFactory(left__name="baz", right__name="baz", right__offset__k=-1),
            AssignStmtFactory(left__name="barbaz", right__name="bar"),
            AssignStmtFactory(left__name="barbaz", right__name="barbaz", right__offset__k=-1),
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


def test_prune_k_cache_fills_backward():
    testee = VerticalLoopFactory(
        loop_order=LoopOrder.BACKWARD,
        sections__0__horizontal_executions__0__body=[
            AssignStmtFactory(left__name="foo", right__name="foo", right__offset__k=-1),
            AssignStmtFactory(left__name="bar", right__name="bar", right__offset__k=0),
            AssignStmtFactory(left__name="baz", right__name="baz", right__offset__k=1),
            AssignStmtFactory(left__name="barbaz", right__name="bar"),
            AssignStmtFactory(left__name="barbaz", right__name="barbaz", right__offset__k=1),
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
        declarations=[
            TemporaryFactory(name="tmp1"),
            TemporaryFactory(name="tmp2"),
        ],
    )
    transformed = PruneKCacheFlushes().visit(testee)
    cache_dict = {c.name: c for c in transformed.vertical_loops[0].caches}
    assert cache_dict["foo"].flush
    assert cache_dict["bar"].flush
    assert not cache_dict["baz"].flush
    assert cache_dict["tmp1"].flush
    assert not cache_dict["tmp2"].flush
