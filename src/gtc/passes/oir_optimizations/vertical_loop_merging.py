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

from typing import Any, Optional, Tuple

from eve import NodeTranslator
from gtc import oir


class AdjacentLoopMerging(NodeTranslator):
    @staticmethod
    def _mergeable(
        a: oir.VerticalLoop, b: oir.VerticalLoop
    ) -> Optional[Tuple[oir.VerticalLoop, oir.VerticalLoop]]:
        if a.loop_order != b.loop_order:
            return None
        a_end = a.sections[-1].interval.end
        b_start = b.sections[0].interval.start
        if a_end.level == b_start.level and a_end.offset == b_start.offset:
            return a, b
        a_start = a.sections[0].interval.start
        b_end = b.sections[-1].interval.end
        if a_start.level == b_end.level and a_start.offset == b_end.offset:
            return b, a
        return None

    @staticmethod
    def _merge(a: oir.VerticalLoop, b: oir.VerticalLoop) -> oir.VerticalLoop:
        assert (
            a.sections[-1].interval.end.level == b.sections[0].interval.start.level
            and a.sections[-1].interval.end.offset == b.sections[0].interval.start.offset
        )

        sections = a.sections + b.sections
        declarations = a.declarations + [
            bd for bd in b.declarations if bd.name not in {ad.name for ad in a.declarations}
        ]
        return oir.VerticalLoop(
            loop_order=a.loop_order,
            sections=sections,
            declarations=declarations,
            caches=[],  # TODO: add support for cache merging?
        )

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        vertical_loops = [self.visit(node.vertical_loops[0], **kwargs)]
        for vertical_loop in node.vertical_loops[1:]:
            vertical_loop = self.visit(vertical_loop, **kwargs)
            mergeable = self._mergeable(vertical_loops[-1], vertical_loop)
            if mergeable:
                vertical_loops[-1] = self._merge(*mergeable)
            else:
                vertical_loops.append(vertical_loop)

        return oir.Stencil(name=node.name, params=node.params, vertical_loops=vertical_loops)
