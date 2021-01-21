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
from typing import Any, Dict, List, Set, Tuple, Union

from eve import NodeTranslator, NodeVisitor
from gtc import oir


class TemporaryDisposal(NodeTranslator):
    """Replaces temporary fields by scalars.

    All temporary fields that are local to a horizontal execution and accesed
    only without offsets are replaced by scalars.
    """

    class LocalTemporaryFinder(NodeVisitor):
        def visit_FieldAccess(
            self,
            node: oir.FieldAccess,
            *,
            access_map: Dict[str, List[Tuple[oir.VerticalLoop, oir.HorizontalExecution, bool]]],
            vertical_loop: oir.VerticalLoop,
            horizontal_execution: oir.HorizontalExecution,
            **kwargs: Any,
        ) -> None:
            data = (
                vertical_loop,
                horizontal_execution,
                node.offset.i == node.offset.j == node.offset.k,
            )
            if data not in access_map[node.name]:
                access_map[node.name].append(data)

        def visit_HorizontalExecution(
            self,
            node: oir.HorizontalExecution,
            **kwargs: Any,
        ) -> None:
            self.generic_visit(node, horizontal_execution=node, **kwargs)

        def visit_VerticalLoop(self, node: oir.VerticalLoop, **kwargs: Any) -> None:
            self.generic_visit(node, vertical_loop=node, **kwargs)

        def visit_Stencil(
            self, node: oir.Stencil, **kwargs: Any
        ) -> Dict[str, Tuple[oir.VerticalLoop, oir.HorizontalExecution]]:
            access_map: Dict[
                str, List[Tuple[oir.VerticalLoop, oir.HorizontalExecution, bool]]
            ] = collections.defaultdict(list)
            self.generic_visit(node, access_map=access_map, **kwargs)
            return {
                field: accesses[0][:2]
                for field, accesses in access_map.items()
                if field in node.symtable_
                and isinstance(node.symtable_[field], oir.Temporary)
                and len(accesses) == 1
                and accesses[0][2]
            }

    class AccessReplacer(NodeTranslator):
        def visit_FieldAccess(
            self, node: oir.FieldAccess, *, local_tmps: Set[str], **kwargs: Any
        ) -> Union[oir.FieldAccess, oir.ScalarAccess]:
            if node.name in local_tmps:
                return oir.ScalarAccess(name=node.name, dtype=node.dtype)
            return self.generic_visit(node, **kwargs)

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        result = self.generic_visit(node, **kwargs)
        local_tmps = self.LocalTemporaryFinder().visit(result)
        for local_tmp, (vertical_loop, horizontal_execution) in local_tmps.items():
            decl = result.symtable_[local_tmp]
            vertical_loop.declarations.remove(decl)
            horizontal_execution.declarations.append(
                oir.LocalScalar(name=decl.name, dtype=decl.dtype, loc=decl.loc)
            )

        result = self.AccessReplacer().visit(result, local_tmps=local_tmps)
        result.collect_symbols()
        return result
