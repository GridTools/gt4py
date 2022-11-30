# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

from collections import defaultdict
from typing import Any, Dict

from eve import NodeTranslator

from . import cuir


class CacheExtents(NodeTranslator):
    def visit_IJCacheDecl(
        self,
        node: cuir.IJCacheDecl,
        *,
        ij_extents: Dict[str, cuir.KExtent],
        **kwargs: Any,
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
