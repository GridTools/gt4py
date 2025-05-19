# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections import defaultdict
from typing import Any, Dict

from gt4py.cartesian.gtc.cuir import cuir
from gt4py.eve import NodeTranslator


class CacheExtents(NodeTranslator):
    def visit_IJCacheDecl(
        self, node: cuir.IJCacheDecl, *, ij_extents: Dict[str, cuir.KExtent], **kwargs: Any
    ) -> cuir.IJCacheDecl:
        return cuir.IJCacheDecl(name=node.name, dtype=node.dtype, extent=ij_extents[node.name])

    def visit_KCacheDecl(
        self, node: cuir.KCacheDecl, *, k_extents: Dict[str, cuir.KExtent], **kwargs: Any
    ) -> cuir.KCacheDecl:
        return cuir.KCacheDecl(name=node.name, dtype=node.dtype, extent=k_extents[node.name])

    def visit_VerticalLoop(self, node: cuir.VerticalLoop) -> cuir.VerticalLoop:
        ij_extents: Dict[str, cuir.IJExtent] = defaultdict(cuir.IJExtent.zero)
        for horizontal_execution in node.walk_values().if_isinstance(cuir.HorizontalExecution):
            ij_access_extents = (
                horizontal_execution.walk_values()
                .if_isinstance(cuir.IJCacheAccess)
                .reduceby(
                    lambda acc, x: acc.union(cuir.IJExtent.from_offset(x.offset)),
                    "name",
                    init=cuir.IJExtent.zero(),
                )
            )
            for field, ij_access_extent in ij_access_extents:
                ij_extents[field] = ij_extents[field].union(
                    ij_access_extent + horizontal_execution.extent
                )

        k_extents = (
            node.walk_values()
            .if_isinstance(cuir.KCacheAccess)
            .reduceby(
                lambda acc, x: acc.union(cuir.KExtent.from_offset(x.offset)),
                "name",
                init=cuir.KExtent.zero(),
                as_dict=True,
            )
        )

        return self.generic_visit(node, ij_extents=ij_extents, k_extents=k_extents)
