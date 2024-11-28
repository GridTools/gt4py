# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from typing import Any

from gt4py.cartesian.gtc import common, oir
from gt4py.eve import NodeTranslator


class AdjacentLoopMerging(NodeTranslator):
    @staticmethod
    def _mergeable(a: oir.VerticalLoop, b: oir.VerticalLoop) -> bool:
        if a.loop_order != b.loop_order:
            return False
        a_interval = a.sections[-1].interval
        b_interval = b.sections[0].interval
        if a.loop_order == common.LoopOrder.BACKWARD:
            a_lim = a_interval.start
            b_lim = b_interval.end
        else:
            a_lim = a_interval.end
            b_lim = b_interval.start
        return a_lim.level == b_lim.level and a_lim.offset == b_lim.offset

    @staticmethod
    def _merge(a: oir.VerticalLoop, b: oir.VerticalLoop) -> oir.VerticalLoop:
        sections = a.sections + b.sections
        if a.caches or b.caches:
            warnings.warn(
                "AdjacentLoopMerging pass removed previously declared caches", stacklevel=2
            )
        return oir.VerticalLoop(loop_order=a.loop_order, sections=sections, caches=[])

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        if not node.vertical_loops:
            return self.generic_visit(node, **kwargs)
        vertical_loops = [self.visit(node.vertical_loops[0], **kwargs)]
        for vertical_loop in node.vertical_loops[1:]:
            vertical_loop = self.visit(vertical_loop, **kwargs)
            mergeable = self._mergeable(vertical_loops[-1], vertical_loop)
            if mergeable:
                vertical_loops[-1] = self._merge(vertical_loops[-1], vertical_loop)
            else:
                vertical_loops.append(vertical_loop)

        return oir.Stencil(
            name=node.name,
            params=node.params,
            vertical_loops=vertical_loops,
            declarations=node.declarations,
            loc=node.loc,
        )
