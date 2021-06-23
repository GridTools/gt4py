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

import copy
from collections import defaultdict
from typing import Any, Dict

from eve import NodeTranslator

from . import cuir


class ComputeExtents(NodeTranslator):
    def visit_Program(self, node: cuir.Program) -> cuir.Program:
        extents_map: Dict[str, cuir.IJExtent] = dict()
        kernels = [self.visit(kernel, extents_map=extents_map) for kernel in reversed(node.kernels)]
        return cuir.Program(
            name=node.name,
            params=node.params,
            temporaries=node.temporaries,
            kernels=list(reversed(kernels)),
        )

    def visit_Kernel(self, node: cuir.Kernel, extents_map: Dict[str, cuir.IJExtent]) -> cuir.Kernel:
        return cuir.Kernel(
            vertical_loops=list(
                reversed(
                    [
                        self.visit(vertical_loop, extents_map=extents_map)
                        for vertical_loop in reversed(node.vertical_loops)
                    ]
                )
            )
        )

    def visit_VerticalLoop(
        self, node: cuir.VerticalLoop, extents_map: Dict[str, cuir.IJExtent]
    ) -> cuir.VerticalLoop:
        extents_maps = [copy.copy(extents_map) for _ in node.sections]
        sections = [
            self.visit(section, extents_map=em) for section, em in zip(node.sections, extents_maps)
        ]
        for em in extents_maps:
            for field, extent in em.items():
                extents_map[field] = extent.union(extents_map.get(field, cuir.IJExtent.zero()))
        return cuir.VerticalLoop(
            loop_order=node.loop_order,
            sections=sections,
            ij_caches=node.ij_caches,
            k_caches=node.k_caches,
        )

    def visit_VerticalLoopSection(
        self, node: cuir.VerticalLoopSection, extents_map: Dict[str, cuir.IJExtent]
    ) -> cuir.VerticalLoopSection:
        horizontal_executions = []
        for horizontal_execution in reversed(node.horizontal_executions):
            writes = (
                node.iter_tree()
                .if_isinstance(cuir.AssignStmt)
                .getattr("left")
                .if_isinstance(cuir.FieldAccess, cuir.IJCacheAccess)
                .getattr("name")
                .to_set()
            )
            extent: cuir.IJExtent = cuir.IJExtent.zero().union(
                *(extents_map.get(write, cuir.IJExtent.zero()) for write in writes),
            )

            horizontal_executions.append(
                cuir.HorizontalExecution(
                    body=horizontal_execution.body,
                    declarations=horizontal_execution.declarations,
                    extent=extent,
                )
            )

            accesses_map: Dict[str, cuir.IJExtent] = (
                horizontal_execution.iter_tree()
                .if_isinstance(cuir.FieldAccess, cuir.IJCacheAccess)
                .reduceby(
                    lambda acc, x: acc.union(extent + cuir.IJExtent.from_offset(x.offset)),
                    "name",
                    init=cuir.IJExtent.zero(),
                    as_dict=True,
                )
            )
            for field in set(extents_map.keys()) | set(accesses_map.keys()):
                extents_map[field] = extents_map.get(field, cuir.IJExtent.zero()).union(
                    accesses_map.get(field, cuir.IJExtent.zero())
                )

        return cuir.VerticalLoopSection(
            start=node.start,
            end=node.end,
            horizontal_executions=list(reversed(horizontal_executions)),
        )


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
        for horizontal_execution in node.iter_tree().if_isinstance(cuir.HorizontalExecution):
            ij_access_extents = (
                horizontal_execution.iter_tree()
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

        k_extents: Dict[str, cuir.KCacheAccess] = (
            node.iter_tree()
            .if_isinstance(cuir.KCacheAccess)
            .reduceby(
                lambda acc, x: acc.union(cuir.KExtent.from_offset(x.offset)),
                "name",
                init=cuir.KExtent.zero(),
                as_dict=True,
            )
        )

        return self.generic_visit(node, ij_extents=ij_extents, k_extents=k_extents)
