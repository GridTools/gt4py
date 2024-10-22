# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.type_system import type_specifications as ts


def _extract_domain(node: ir.Node) -> list[tuple[gtx_common.Domain, ir.Expr. ir.Expr]]:
    """
    Visits the domain of a field operator and returns a list of dimensions and
    the corresponding lower and upper bounds. The returned lower bound is inclusive,
    the upper bound is exclusive: [lower_bound, upper_bound[
    """
    assert cpm.is_call_to(node, ("cartesian_domain", "unstructured_domain"))

    domain = []
    for named_range in node.args:
        assert cpm.is_call_to(named_range, "named_range")
        assert len(named_range.args) == 3
        axis = named_range.args[0]
        assert isinstance(axis, ir.AxisLiteral)
        lower_bound, upper_bound = named_range.args[1:3]
        dim = gtx_common.Dimension(axis.value, axis.kind)
        domain.append(dim, lower_bound, upper_bound)

    return domain


class NormalizeFieldopDomain(PreserveLocationVisitor, NodeTranslator):
    """
    Removes lower bound from field operator domain and applies extra shift on field arguments.
    """

    def visit_FunCall(self, node: ir.FunCall, domain_shift: dict[gtx_common.Dimension, ir.Expr]) -> ir.Node:
        
        if not cpm.is_applied_as_fieldop(node):
            return self.generic_visit(node, domain_shift)
        
        assert isinstance(node.type, ts.FieldType)

        fun_node = node.fun
        assert len(fun_node.args) == 2
        domain_expr = fun_node.args[1]

        domain = _extract_domain(domain_expr)
        new_domain_shift = {
            dim: (im.plus(lower_bound, domain_shift.get(dim, 0)[0]), im.plus(upper_bound, domain_shift.get(dim, 0)[1]))
            for (dim, lower_bound, upper_bound) in domain
        }

        node.args = self.visit(node.args, domain_shift=new_domain_shift)

        if not cpm.is_call_to(node, "cast_"):
            return node

        value, type_constructor = node.args

        assert (
            value.type
            and isinstance(type_constructor, ir.SymRef)
            and (type_constructor.id in ir.TYPEBUILTINS)
        )
        dtype = ts.ScalarType(kind=getattr(ts.ScalarKind, type_constructor.id.upper()))

        if value.type == dtype:
            return value

        return node

    @classmethod
    def apply(cls, node: ir.Node) -> ir.Node:
        return cls().visit(node)
