# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next import utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.iterator.transforms import symbol_ref_utils
from gt4py.next.iterator.type_system import inference as type_inference
from gt4py.next.type_system import type_specifications as ts


def _in(pos: itir.Expr, domain: itir.Expr) -> itir.Expr:
    """
    Given a position and a domain return an expression that evaluates to `True` if the position is inside the domain.

    `in_({i, j, k}, u⟨ Iₕ: [i0, i1[, Iₕ: [j0, j1[, Iₕ: [k0, k1[ ⟩`
    -> `i0 <= i < i1 & j0 <= j < j1 & k0 <= k < k1`
    """
    ret = []
    for i, v in enumerate(domain_utils.SymbolicDomain.from_expr(domain).ranges.values()):
        ret.append(
            im.and_(
                im.less_equal(v.start, im.tuple_get(i, pos)),
                im.less(im.tuple_get(i, pos), v.stop),
            )
        )
    return functools.reduce(im.and_, ret)


class _TransformToAsFieldop(PreserveLocationVisitor, NodeTranslator):
    PRESERVED_ANNEX_ATTRS = (
        "type",
        "domain",
    )

    @classmethod
    def apply(cls, node: itir.Node):
        """
        Transform `concat_where` expressions into equivalent `as_fieldop` expressions.

        Note that (backward) domain inference may not be executed after this pass as it can not
        correctly infer the accessed domains when the value selection is represented as an `if_`
        inside the `as_fieldop.
        """
        node = cls().visit(node)
        node = type_inference.SanitizeTypes().visit(node)
        return node

    def visit_FunCall(self, node: itir.FunCall) -> itir.FunCall:
        node = self.generic_visit(node)
        if cpm.is_call_to(node, "concat_where"):
            cond, true_branch, false_branch = node.args
            assert isinstance(cond.type, ts.DomainType)
            position = [im.index(dim) for dim in cond.type.dims]
            refs = symbol_ref_utils.collect_symbol_refs(cond)

            domains: tuple[domain_utils.SymbolicDomain, ...] = utils.flatten_nested_tuple(
                node.annex.domain
            )
            assert all(domain == domains[0] for domain in domains), (
                "At this point all `concat_where` arguments should be posed on the same domain."
            )
            assert isinstance(domains[0], domain_utils.SymbolicDomain)
            domain_expr = domains[0].as_expr()

            return im.as_fieldop(
                im.lambda_("__tcw_pos", "__tcw_arg0", "__tcw_arg1", *refs)(
                    im.let(*zip(refs, map(im.deref, refs), strict=True))(
                        im.if_(
                            _in(im.deref("__tcw_pos"), cond),
                            im.deref("__tcw_arg0"),
                            im.deref("__tcw_arg1"),
                        )
                    )
                ),
                domain_expr,
            )(im.make_tuple(*position), true_branch, false_branch, *refs)

        return node


transform_to_as_fieldop = _TransformToAsFieldop.apply
