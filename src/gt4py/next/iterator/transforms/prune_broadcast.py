# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)


@dataclasses.dataclass
class PruneBroadcast(PreserveLocationVisitor, NodeTranslator):
    PRESERVED_ANNEX_ATTRS = ("domain",)

    @classmethod
    def apply(cls, node: itir.Node):
        return cls().visit(node)

    def visit_FunCall(self, node: itir.FunCall) -> itir.FunCall:
        node = self.generic_visit(node)

        if cpm.is_call_to(node, "broadcast"):
            expr = self.visit(node.args[0])
            node = im.as_fieldop("deref", domain_utils.SymbolicDomain.as_expr(node.annex.domain))(
                expr
            )
        return node
