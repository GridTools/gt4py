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

from eve import NodeTranslator, NodeVisitor
from gtc import oir


class TemporaryDisposal(NodeTranslator):
    """Replaces temporary fields by scalars.

    1. Finds temporaries that are only accessed within a single HorizontalExecution.
    2. Replaces corresponding FieldAccess nodes by ScalarAccess nodes.
    3. Removes matching temporaries from VerticalLoop declarations.
    4. Add matching temporaries to HorizontalExecution declarations.
    """

    class LocalTemporaryFinder(NodeVisitor):
        """Finds a map from local temporaries to horizontal executions."""

        def visit_FieldAccess(
            self,
            node: oir.FieldAccess,
            *,
            access_map: Dict[str, Set[int]],
            hexec_id: int,
            **kwargs: Any,
        ) -> None:
            access_map[node.name].add(hexec_id)

        def visit_HorizontalExecution(self, node: oir.HorizontalExecution, **kwargs: Any) -> None:
            self.generic_visit(node, hexec_id=id(node), **kwargs)

        def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> Dict[str, int]:
            access_map: Dict[str, Set[int]] = collections.defaultdict(set)
            self.generic_visit(node, access_map=access_map, **kwargs)
            return {
                field: next(iter(accesses))
                for field, accesses in access_map.items()
                if len(accesses) == 1
                and field in node.symtable_
                and isinstance(node.symtable_[field], oir.Temporary)
            }

    def visit_FieldAccess(
        self, node: oir.FieldAccess, *, local_tmps: Set[str], **kwargs: Any
    ) -> Union[oir.FieldAccess, oir.ScalarAccess]:
        if node.name in local_tmps:
            return oir.ScalarAccess(name=node.name, dtype=node.dtype)
        return self.generic_visit(node, **kwargs)

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        local_tmps: Dict[str, int],
        symtable: Dict[str, Any],
        **kwargs: Any,
    ) -> oir.HorizontalExecution:
        result = self.generic_visit(node, local_tmps=local_tmps, **kwargs)
        result.declarations += [
            oir.LocalScalar(name=name, dtype=symtable[name].dtype, loc=symtable[name].loc)
            for name, hexec_id in local_tmps.items()
            if hexec_id == id(node)
        ]
        return result

    def visit_VerticalLoop(
        self,
        node: oir.VerticalLoop,
        local_tmps: Dict[str, int],
        symtable: Dict[str, Any],
        **kwargs: Any,
    ) -> oir.VerticalLoop:
        result = self.generic_visit(node, local_tmps=local_tmps, symtable=symtable, **kwargs)
        result.declarations = [d for d in result.declarations if d.name not in local_tmps]
        return result

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        local_tmps = self.LocalTemporaryFinder().visit(node)
        result = self.generic_visit(node, local_tmps=local_tmps, symtable=node.symtable_, **kwargs)
        result.collect_symbols()
        return result
