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

import collections
from typing import Any, Dict, Set, Union

from eve import NodeTranslator
from gtc import oir


class TemporariesToScalars(NodeTranslator):
    """Replaces temporary fields by scalars.

    1. Finds temporaries that are only accessed within a single HorizontalExecution.
    2. Replaces corresponding FieldAccess nodes by ScalarAccess nodes.
    3. Removes matching temporaries from VerticalLoop declarations.
    4. Add matching temporaries to HorizontalExecution declarations.
    """

    def visit_FieldAccess(
        self, node: oir.FieldAccess, *, local_tmps: Set[str], **kwargs: Any
    ) -> Union[oir.FieldAccess, oir.ScalarAccess]:
        if node.name in local_tmps:
            assert (
                node.offset.i == node.offset.j == node.offset.k == 0
            ), "Non-zero offset in local temporary?!"
            return oir.ScalarAccess(name=node.name, dtype=node.dtype)
        return self.generic_visit(node, **kwargs)

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        local_tmps: Set[str],
        symtable: Dict[str, Any],
        **kwargs: Any,
    ) -> oir.HorizontalExecution:
        declarations = node.declarations + [
            oir.LocalScalar(name=tmp, dtype=symtable[tmp].dtype, loc=symtable[tmp].loc)
            for tmp in node.iter_tree()
            .if_isinstance(oir.FieldAccess)
            .getattr("name")
            .if_in(local_tmps)
            .to_set()
        ]
        return oir.HorizontalExecution(
            body=self.visit(node.body, local_tmps=local_tmps, **kwargs),
            mask=self.visit(node.mask, local_tmps=local_tmps, **kwargs),
            declarations=declarations,
        )

    def visit_VerticalLoopSection(
        self,
        node: oir.VerticalLoop,
        local_tmps: Set[str],
        **kwargs: Any,
    ) -> oir.VerticalLoopSection:
        return oir.VerticalLoopSection(
            interval=node.interval,
            horizontal_executions=self.visit(
                node.horizontal_executions, local_tmps=local_tmps, **kwargs
            ),
        )

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        temporaries = {
            symbol for symbol, value in node.symtable_.items() if isinstance(value, oir.Temporary)
        }
        horizontal_executions = node.iter_tree().if_isinstance(oir.HorizontalExecution)
        counts: collections.Counter = sum(
            (
                collections.Counter(
                    horizontal_execution.iter_tree()
                    .if_isinstance(oir.FieldAccess)
                    .getattr("name")
                    .if_in(temporaries)
                    .to_set()
                )
                for horizontal_execution in horizontal_executions
            ),
            collections.Counter(),
        )
        local_tmps = {tmp for tmp, count in counts.items() if count == 1}
        return oir.Stencil(
            name=node.name,
            params=node.params,
            vertical_loops=self.visit(
                node.vertical_loops, local_tmps=local_tmps, symtable=node.symtable_, **kwargs
            ),
            declarations=[d for d in node.declarations if d.name not in local_tmps],
        )
