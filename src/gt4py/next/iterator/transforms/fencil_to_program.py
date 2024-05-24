# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

from gt4py import eve
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import global_tmps


class FencilToProgram(eve.NodeTranslator):
    @classmethod
    def apply(cls, node: itir.FencilDefinition | global_tmps.FencilWithTemporaries) -> itir.Program:
        return cls().visit(node)

    def visit_StencilClosure(self, node: itir.StencilClosure) -> itir.SetAt:
        as_fieldop = im.call(im.call("as_fieldop")(node.stencil, node.domain))(*node.inputs)
        return itir.SetAt(expr=as_fieldop, domain=node.domain, target=node.output)

    def visit_FencilDefinition(self, node: itir.FencilDefinition) -> itir.Program:
        return itir.Program(
            id=node.id,
            function_definitions=node.function_definitions,
            params=node.params,
            declarations=[],
            body=self.visit(node.closures),
            implicit_domain=node.implicit_domain,
        )

    def visit_FencilWithTemporaries(self, node: global_tmps.FencilWithTemporaries) -> itir.Program:
        return itir.Program(
            id=node.fencil.id,
            function_definitions=node.fencil.function_definitions,
            params=node.params,
            declarations=node.tmps,
            body=self.visit(node.fencil.closures),
            implicit_domain=node.fencil.implicit_domain,
        )
