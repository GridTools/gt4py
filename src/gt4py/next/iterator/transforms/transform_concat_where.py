# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.iterator.transforms import symbol_ref_utils


class TransformConcatWhere(PreserveLocationVisitor, NodeTranslator):
    @classmethod
    def apply(cls, node: ir.Node):
        return cls().visit(node)

    def visit_FunCall(self, node: ir.FunCall) -> ir.FunCall:
        node = self.generic_visit(node)
        if cpm.is_call_to(node, "concat_where"):
            cond_expr, field_a, field_b = node.args
            cond = domain_utils.SymbolicDomain.from_expr(cond_expr).ranges.keys()
            dims = [im.call("index")(ir.AxisLiteral(value=k.value, kind=k.kind)) for k in cond]
            refs = symbol_ref_utils.collect_symbol_refs(cond_expr)

            return im.as_fieldop(
                im.lambda_("__tcw_pos", "__tcw_arg0", "__tcw_arg1", *refs)(
                    im.let(*zip(refs, map(im.deref, refs), strict=True))(
                        im.if_(
                            im.call("in_")(im.deref("__tcw_pos"), cond_expr),
                            im.deref("__tcw_arg0"),
                            im.deref("__tcw_arg1"),
                        )
                    )
                ),
            )(im.make_tuple(*dims), field_a, field_b, *refs)

        return node
