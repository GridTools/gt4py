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
    AssignStmtBuilder,
    FieldDeclBuilder,
    HorizontalExecutionBuilder,
    IJCacheBuilder,
    KCacheBuilder,
    StencilBuilder,
    TemporaryBuilder,
    VerticalLoopBuilder,
    VerticalLoopSectionBuilder,
)


def test_ij_cache_detection():
    testee = (
        VerticalLoopBuilder()
        .add_section(
            VerticalLoopSectionBuilder()
            .add_horizontal_execution(
                HorizontalExecutionBuilder()
                .add_stmt(AssignStmtBuilder("tmp1", "bar", (1, 0, 0)).build())
                .build()
            )
            .add_horizontal_execution(
                HorizontalExecutionBuilder()
                .add_stmt(AssignStmtBuilder("tmp2", "tmp1", (0, 1, 0)).build())
                .build()
            )
            .add_horizontal_execution(
                HorizontalExecutionBuilder()
                .add_stmt(AssignStmtBuilder("baz", "tmp2", (0, 0, 1)).build())
                .add_stmt(AssignStmtBuilder("tmp3", "baz").build())
                .add_stmt(AssignStmtBuilder("foo", "tmp3").build())
                .build()
            )
            .build()
        )
        .add_declaration(TemporaryBuilder(name="tmp1").build())
        .add_declaration(TemporaryBuilder(name="tmp2").build())
        .add_declaration(TemporaryBuilder(name="tmp3").build())
        .add_cache(IJCacheBuilder("tmp3").build())
        .build()
    )
    transformed = IJCacheDetection().visit(testee)
    assert len(transformed.caches) == 2
    assert {cache.name for cache in transformed.caches} == {"tmp1", "tmp3"}
    assert all(isinstance(cache, IJCache) for cache in transformed.caches)


def test_k_cache_detection_basic():
    testee = (
        VerticalLoopBuilder()
        .loop_order(LoopOrder.FORWARD)
        .add_section(
            VerticalLoopSectionBuilder()
            .add_horizontal_execution(
                HorizontalExecutionBuilder()
                .add_stmt(AssignStmtBuilder("foo", "foo", (0, 0, 1)).build())
                .add_stmt(AssignStmtBuilder("bar", "foo", (0, 0, -1)).build())
                .add_stmt(AssignStmtBuilder("baz", "baz", (1, 0, 1)).build())
                .add_stmt(AssignStmtBuilder("foo", "baz", (0, 1, -1)).build())
                .build()
            )
            .build()
        )
        .build()
    )
    transformed = KCacheDetection().visit(testee)
    assert {c.name for c in transformed.caches} == {"foo"}
    assert all(isinstance(cache, KCache) for cache in transformed.caches)


def test_k_cache_detection_single_access_point():
    testee = (
        VerticalLoopBuilder()
        .loop_order(LoopOrder.FORWARD)
        .add_section(
            VerticalLoopSectionBuilder()
            .add_horizontal_execution(
                HorizontalExecutionBuilder()
                .add_stmt(AssignStmtBuilder("foo", "bar").build())
                .build()
            )
            .add_horizontal_execution(
                HorizontalExecutionBuilder()
                .add_stmt(AssignStmtBuilder("bar", "baz", (0, 0, 1)).build())
                .build()
            )
            .build()
        )
        .build()
    )
    transformed = KCacheDetection().visit(testee)
    assert not transformed.caches


def test_prune_k_cache_fills_forward():
    testee = (
        VerticalLoopBuilder()
        .loop_order(LoopOrder.FORWARD)
        .add_section(
            VerticalLoopSectionBuilder()
            .add_horizontal_execution(
                HorizontalExecutionBuilder()
                .add_stmt(AssignStmtBuilder("foo", "foo", (0, 0, 1)).build())
                .add_stmt(AssignStmtBuilder("bar", "bar", (0, 0, 0)).build())
                .add_stmt(AssignStmtBuilder("baz", "baz", (0, 0, -1)).build())
                .add_stmt(AssignStmtBuilder("barbaz", "bar").build())
                .add_stmt(AssignStmtBuilder("barbaz", "barbaz", (0, 0, -1)).build())
                .build()
            )
            .build()
        )
        .add_cache(KCacheBuilder("foo", fill=True).build())
        .add_cache(KCacheBuilder("bar", fill=True).build())
        .add_cache(KCacheBuilder("baz", fill=True).build())
        .add_cache(KCacheBuilder("barbaz", fill=True).build())
        .build()
    )
    transformed = PruneKCacheFills().visit(testee)
    cache_dict = {c.name: c for c in transformed.caches}
    assert cache_dict["foo"].fill
    assert cache_dict["bar"].fill
    assert not cache_dict["baz"].fill
    assert not cache_dict["barbaz"].fill


def test_prune_k_cache_fills_backward():
    testee = (
        VerticalLoopBuilder()
        .loop_order(LoopOrder.BACKWARD)
        .add_section(
            VerticalLoopSectionBuilder()
            .add_horizontal_execution(
                HorizontalExecutionBuilder()
                .add_stmt(AssignStmtBuilder("foo", "foo", (0, 0, -1)).build())
                .add_stmt(AssignStmtBuilder("bar", "bar", (0, 0, 0)).build())
                .add_stmt(AssignStmtBuilder("baz", "baz", (0, 0, 1)).build())
                .add_stmt(AssignStmtBuilder("barbaz", "bar").build())
                .add_stmt(AssignStmtBuilder("barbaz", "barbaz", (0, 0, 1)).build())
                .build()
            )
            .build()
        )
        .add_cache(KCacheBuilder("foo", fill=True).build())
        .add_cache(KCacheBuilder("bar", fill=True).build())
        .add_cache(KCacheBuilder("baz", fill=True).build())
        .add_cache(KCacheBuilder("barbaz", fill=True).build())
        .build()
    )
    transformed = PruneKCacheFills().visit(testee)
    cache_dict = {c.name: c for c in transformed.caches}
    assert cache_dict["foo"].fill
    assert cache_dict["bar"].fill
    assert not cache_dict["baz"].fill
    assert not cache_dict["barbaz"].fill


def test_prune_k_cache_flushes():
    testee = (
        StencilBuilder()
        .add_param(FieldDeclBuilder("foo").build())
        .add_param(FieldDeclBuilder("bar").build())
        .add_param(FieldDeclBuilder("baz").build())
        .add_vertical_loop(
            VerticalLoopBuilder()
            .loop_order(LoopOrder.FORWARD)
            .add_section(
                VerticalLoopSectionBuilder()
                .add_horizontal_execution(
                    HorizontalExecutionBuilder()
                    .add_stmt(AssignStmtBuilder("foo", "foo", (0, 0, 1)).build())
                    .add_stmt(AssignStmtBuilder("bar", "baz", (0, 0, 1)).build())
                    .add_stmt(AssignStmtBuilder("tmp1", "tmp1", (0, 0, 1)).build())
                    .add_stmt(AssignStmtBuilder("tmp2", "tmp2", (0, 0, 1)).build())
                    .build()
                )
                .build()
            )
            .add_cache(KCacheBuilder("foo", flush=True).build())
            .add_cache(KCacheBuilder("bar", flush=True).build())
            .add_cache(KCacheBuilder("baz", flush=True).build())
            .add_cache(KCacheBuilder("tmp1", flush=True).build())
            .add_cache(KCacheBuilder("tmp2", flush=True).build())
            .add_declaration(TemporaryBuilder("tmp1").build())
            .add_declaration(TemporaryBuilder("tmp2").build())
            .build()
        )
        .add_vertical_loop(
            VerticalLoopBuilder()
            .loop_order(LoopOrder.FORWARD)
            .add_section(
                VerticalLoopSectionBuilder()
                .add_horizontal_execution(
                    HorizontalExecutionBuilder()
                    .add_stmt(AssignStmtBuilder("bar", "bar", (0, 0, 1)).build())
                    .add_stmt(AssignStmtBuilder("tmp1", "tmp1", (0, 0, 1)).build())
                    .build()
                )
                .build()
            )
            .build()
        )
        .build()
    )
    transformed = PruneKCacheFlushes().visit(testee)
    cache_dict = {c.name: c for c in transformed.vertical_loops[0].caches}
    assert cache_dict["foo"].flush
    assert cache_dict["bar"].flush
    assert not cache_dict["baz"].flush
    assert cache_dict["tmp1"].flush
    assert not cache_dict["tmp2"].flush
