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

from typing import Any, Dict

from eve import NodeTranslator

from . import cuir


def _extents_map(node: cuir.LocNode) -> Dict[str, cuir.Extent]:
    return (
        node.iter_tree()
        .if_isinstance(cuir.FieldAccess)
        .reduceby(
            lambda ext, acc: ext.union(
                cuir.Extent(i=(acc.offset.i, acc.offset.i), j=(acc.offset.j, acc.offset.j)),
            ),
            "name",
            init=cuir.Extent.zero(),
            as_dict=True,
        )
    )


class ComputeExtents(NodeTranslator):
    def visit_VerticalLoopSection(
        self, node: cuir.VerticalLoopSection, **kwargs: Any
    ) -> cuir.VerticalLoopSection:
        horizontal_executions = []
        extents_map: Dict[str, cuir.Extent] = dict()
        for horizontal_execution in reversed(node.horizontal_executions):
            writes = (
                node.iter_tree()
                .if_isinstance(cuir.AssignStmt)
                .getattr("left")
                .if_isinstance(cuir.FieldAccess, cuir.IJCacheAccess)
                .getattr("name")
                .to_set()
            )
            extent = cuir.Extent.zero().union(
                *(extents_map.get(write, cuir.Extent.zero()) for write in writes),
            )

            horizontal_executions.append(
                cuir.HorizontalExecution(
                    body=horizontal_execution.body,
                    mask=horizontal_execution.mask,
                    declarations=horizontal_execution.declarations,
                    extent=extent,
                )
            )

            accesses_map = {k: extent + v for k, v in _extents_map(horizontal_execution).items()}
            extents_map = {
                k: extents_map.get(k, cuir.Extent.zero()).union(
                    accesses_map.get(k, cuir.Extent.zero())
                )
                for k in set(extents_map.keys()) | set(accesses_map.keys())
            }

        return cuir.VerticalLoopSection(
            start=node.start,
            end=node.end,
            horizontal_executions=list(reversed(horizontal_executions)),
        )
