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


class FencilToProgram(eve.NodeTranslator):
    @classmethod
    def apply(cls, node: itir.FencilDefinition) -> itir.Program:
        return cls().visit(node)

    def visit_StencilClosure(self, node: itir.StencilClosure) -> itir.Assign:
        apply_stencil = im.call(im.call("apply_stencil")(node.stencil, node.domain))(*node.inputs)
        return itir.Assign(target=node.output, expr=apply_stencil)

    def visit_FencilDefinition(self, node: itir.FencilDefinition) -> itir.Program:
        return itir.Program(
            id=node.id,
            function_definitions=node.function_definitions,
            params=node.params,
            declarations=[],
            body=self.visit(node.closures),
        )

    def visit_FencilWithTemporaries(self, node) -> itir.Program:
        return itir.Program(
            id=node.fencil.id,
            function_definitions=node.fencil.function_definitions,
            params=node.params,
            declarations=node.tmps,
            body=self.visit(node.fencil.closures),
        )
