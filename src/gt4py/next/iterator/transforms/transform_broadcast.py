# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.program_processors.codegens.gtfn.itir_to_gtfn_ir import _get_domains


@dataclasses.dataclass
class TransformBroadcast(PreserveLocationVisitor, NodeTranslator):
    # PRESERVED_ANNEX_ATTRS = (
    #     "type",
    #     "domain",
    # )
    domain: common.Domain

    # @classmethod
    # def apply(cls, node: ir.Node):
    #     return cls().visit(node)

    @classmethod
    def apply(cls, program: itir.Program):
        # TODO: move _get_domains?
        return cls(domain=_get_domains(program.body)).visit(program)

    def visit_FunCall(self, node: itir.FunCall) -> itir.FunCall:
        node = self.generic_visit(node)

        if cpm.is_call_to(node, "broadcast"):
            expr = self.visit(node.args[0])
            node = im.as_fieldop(im.ref("deref"), next(iter(self.domain)))(expr)
        return node
