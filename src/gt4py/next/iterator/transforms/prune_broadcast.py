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
from gt4py.next.iterator.type_system import inference
from gt4py.next.type_system import type_specifications as ts


@dataclasses.dataclass
class PruneBroadcast(PreserveLocationVisitor, NodeTranslator):
    PRESERVED_ANNEX_ATTRS = ("domain",)

    @classmethod
    def apply(cls, node: itir.Node):
        return cls().visit(node)

    def visit_FunCall(self, node: itir.FunCall) -> itir.FunCall:
        node = self.generic_visit(node)

        # TODO(tehrengruber): write test to document why domain restriction is needed.
        # For this case the domain inference writes a (Vertex, K) domain to val, but the type
        # inference infers a Vertex only field, which leads to incompatible types after the
        # broadcast is rewritten.
        # let val = broadcast(1, (Vertex,))
        #   as_fieldop(deref, unstructured_domain(Vertex, K))
        # end
        # Even with the fix below the temporary extraction fails, after the broadcast is rewritten
        # if it overwrites the smaller domain we created here. Therefore the domain must not be
        # overwritten. Document this in the domain inference.
        if cpm.is_call_to(node, "broadcast"):
            expr = self.visit(node.args[0])
            dims_expr = node.args[1]
            # reinference is fine even if node has not been inferred previously as long as we have
            # axis literals only, but no refs to axis literals
            inference.reinfer(dims_expr)
            assert isinstance(dims_expr.type, ts.TupleType) and all(
                isinstance(el, ts.DimensionType) for el in dims_expr.type.types
            )
            dims = [dt.dim for dt in dims_expr.type.types]
            domain: domain_utils.SymbolicDomain = node.annex.domain
            restricted_domain = domain_utils.SymbolicDomain(
                grid_type=domain.grid_type,
                ranges={d: r for d, r in domain.ranges.items() if d in dims},
            )
            node = im.as_fieldop("deref", domain_utils.SymbolicDomain.as_expr(restricted_domain))(
                expr
            )
        return node
