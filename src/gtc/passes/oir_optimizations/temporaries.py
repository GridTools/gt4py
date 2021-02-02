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

from .utils import AccessCollector


class TemporariesToScalarsBase(NodeTranslator):
    def visit_FieldAccess(
        self, node: oir.FieldAccess, *, tmps_to_replace: Set[str], **kwargs: Any
    ) -> Union[oir.FieldAccess, oir.ScalarAccess]:
        if node.name in tmps_to_replace:
            assert (
                node.offset.i == node.offset.j == node.offset.k == 0
            ), "Non-zero offset in temporary that is replaced?!"
            return oir.ScalarAccess(name=node.name, dtype=node.dtype)
        return self.generic_visit(node, **kwargs)

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        tmps_to_replace: Set[str],
        symtable: Dict[str, Any],
        **kwargs: Any,
    ) -> oir.HorizontalExecution:
        declarations = node.declarations + [
            oir.LocalScalar(name=tmp, dtype=symtable[tmp].dtype, loc=symtable[tmp].loc)
            for tmp in node.iter_tree()
            .if_isinstance(oir.FieldAccess)
            .getattr("name")
            .if_in(tmps_to_replace)
            .to_set()
        ]
        return oir.HorizontalExecution(
            body=self.visit(node.body, tmps_to_replace=tmps_to_replace, **kwargs),
            mask=self.visit(node.mask, tmps_to_replace=tmps_to_replace, **kwargs),
            declarations=declarations,
        )

    def visit_VerticalLoop(
        self,
        node: oir.VerticalLoop,
        tmps_to_replace: Set[str],
        **kwargs: Any,
    ) -> oir.VerticalLoop:
        return oir.VerticalLoop(
            loop_order=node.loop_order,
            sections=self.visit(node.sections, tmps_to_replace=tmps_to_replace, **kwargs),
            declarations=[d for d in node.declarations if d.name not in tmps_to_replace],
            caches=[c for c in node.caches if c.name not in tmps_to_replace],
        )


class LocalTemporariesToScalars(TemporariesToScalarsBase):
    """Replaces temporary fields by scalars.

    1. Finds temporaries that are only accessed within a single HorizontalExecution.
    2. Replaces corresponding FieldAccess nodes by ScalarAccess nodes.
    3. Removes matching temporaries from VerticalLoop declarations.
    4. Add matching temporaries to HorizontalExecution declarations.
    """

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
        return self.generic_visit(
            node, tmps_to_replace=local_tmps, symtable=node.symtable_, **kwargs
        )


class WriteBeforeReadTemporariesToScalars(TemporariesToScalarsBase):
    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        write_before_read_tmps = {
            symbol for symbol, value in node.symtable_.items() if isinstance(value, oir.Temporary)
        }
        horizontal_executions = node.iter_tree().if_isinstance(oir.HorizontalExecution)

        for horizontal_execution in horizontal_executions:
            accesses = AccessCollector.apply(horizontal_execution)
            offsets = accesses.offsets()
            ordered_accesses = accesses.ordered_accesses()

            def write_before_read(tmp: str) -> bool:
                if tmp not in offsets:
                    return True
                if offsets[tmp] != {(0, 0, 0)}:
                    return False
                return next(o.is_write for o in ordered_accesses if o.field == tmp)

            write_before_read_tmps = {
                tmp for tmp in write_before_read_tmps if write_before_read(tmp)
            }

        return self.generic_visit(
            node, tmps_to_replace=write_before_read_tmps, symtable=node.symtable_, **kwargs
        )
