# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next import common
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im, \
    domain_utils
from gt4py.next.iterator.ir_utils.domain_utils import SymbolicDomain
from gt4py.next.iterator import ir as itir


class NestConcatWheres(PreserveLocationVisitor, NodeTranslator):
    @classmethod
    def apply(cls, node: ir.Node):
        return cls().visit(node)

    def visit_FunCall(self, node: ir.FunCall) -> ir.FunCall:
        node = self.generic_visit(node)

        # TODO: do not duplicate exprs
        if cpm.is_call_to(node, "concat_where"):
            cond_expr, field_a, field_b = node.args
            # TODO: don't duplicate exprs here
            if cpm.is_call_to(cond_expr, "and_"):
                conds = cond_expr.args
                return self.visit(im.concat_where(
                    conds[0], im.concat_where(conds[1], field_a, field_b), field_b
                ))
            if cpm.is_call_to(cond_expr, "or_"):
                conds = cond_expr.args
                return self.visit(im.concat_where(
                    conds[0], field_a, im.concat_where(conds[1], field_a, field_b)
                ))
            if cpm.is_call_to(cond_expr, "eq"):
                cond1 = im.less(cond_expr.args[0], cond_expr.args[1])
                cond2 = im.greater(cond_expr.args[0], cond_expr.args[1])
                return self.visit(im.concat_where(cond1, field_b, im.concat_where(cond2, field_b, field_a)))

            # concat_where([1, 2[, a, b) -> concat_where([-inf, 1] | [2, inf[, b, a)
            if cpm.is_call_to(cond_expr, ("cartesian_domain", "unstructured_domain")):
                domain = SymbolicDomain.from_expr(cond_expr)
                if len(domain.ranges) == 1:
                    dim, range_ = next(iter(domain.ranges.items()))
                    if domain_utils.is_finite(range_):
                        complement = _range_complement(range_)
                        new_domains = [im.domain(
                            domain.grid_type,
                            {dim: (cr.start, cr.stop)}
                        ) for cr in complement]
                        # TODO: fp transform
                        return self.visit(im.concat_where(im.call("or_")(*new_domains), field_b, field_a))
                else:
                    # TODO(tehrengruber): Implement. Note that this case can not be triggered by
                    #  the frontend.
                    raise NotImplementedError()

        return node


def _range_complement(range_: domain_utils.SymbolicRange) -> tuple[domain_utils.SymbolicRange, domain_utils.SymbolicRange]:
    # `[a, b[` -> `[-inf, a[` ∪ `[b, inf[`
    assert not any(isinstance(b, itir.InfinityLiteral) for b in [range_.start, range_.stop])
    return (
        domain_utils.SymbolicRange(itir.InfinityLiteral.NEGATIVE, range_.start),
        domain_utils.SymbolicRange(range_.stop, itir.InfinityLiteral.POSITIVE)
    )