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

from typing import Any

from eve import NodeTranslator
from gtc import common, oir

from .utils import AccessCollector


class IJCacheDetection(NodeTranslator):
    def visit_VerticalLoop(self, node: oir.VerticalLoop, **kwargs: Any) -> oir.VerticalLoop:
        if node.loop_order != common.LoopOrder.PARALLEL:
            return self.generic_visit(node, **kwargs)
        accesses = AccessCollector.apply(node).accesses
        # TODO: ij-Caches for non-temporaries?
        cacheable = {
            field
            for field, offsets in accesses.items()
            if field in {d.name for d in node.declarations}
            and field not in {c.name for c in node.caches}
            and all(o[2] == 0 for o in offsets)
        }
        caches = self.visit(node.caches, **kwargs) + [
            oir.IJCache(name=field) for field in cacheable
        ]
        return oir.VerticalLoop(
            interval=self.visit(node.interval, **kwargs),
            horizontal_executions=self.visit(node.horizontal_executions, **kwargs),
            loop_order=self.visit(node.loop_order, **kwargs),
            declarations=self.visit(node.declarations, **kwargs),
            caches=caches,
        )


class KCacheDetection(NodeTranslator):
    def visit_VerticalLoop(self, node: oir.VerticalLoop, **kwargs: Any) -> oir.VerticalLoop:
        if node.loop_order == common.LoopOrder.PARALLEL:
            return self.generic_visit(node, **kwargs)
        accesses = AccessCollector.apply(node).accesses
        # TODO: k-caches with non-zero ij offsets?
        cacheable = {
            field
            for field, offsets in accesses.items()
            if field not in {c.name for c in node.caches}
            and len(offsets) > 1
            and all(o[:2] == (0, 0) for o in offsets)
        }
        caches = self.visit(node.caches, **kwargs) + [
            oir.KCache(name=field, fill=True, flush=True) for field in cacheable
        ]
        return oir.VerticalLoop(
            interval=self.visit(node.interval, **kwargs),
            horizontal_executions=self.visit(node.horizontal_executions, **kwargs),
            loop_order=self.visit(node.loop_order, **kwargs),
            declarations=self.visit(node.declarations, **kwargs),
            caches=caches,
        )
