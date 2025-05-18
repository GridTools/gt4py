# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Optional

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next import common, utils
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.iterator.transforms import infer_domain, symbol_ref_utils
from gt4py.next.iterator.type_system import inference as type_inference
from gt4py.next.type_system import type_specifications as ts


class _TransformTupleConcatWhere(PreserveLocationVisitor, NodeTranslator):
    PRESERVED_ANNEX_ATTRS = (
        "type",
        "domain",
    )

    @classmethod
    def apply(cls, node: ir.Node, offset_provider_type: common.OffsetProviderType):
        node = type_inference.infer(
            node,
            offset_provider_type=offset_provider_type,
        )
        return cls().visit(node)

    def visit_FunCall(self, node: ir.FunCall) -> ir.FunCall:
        node = self.generic_visit(node)

        # `concat_where(cond, {a, b}, {c, d})`
        # -> `{concat_where(cond, a, c), concat_where(cond, a, c)}`
        if cpm.is_call_to(node, "concat_where") and isinstance(node.args[1].type, ts.TupleType):
            cond, true_branch, false_branch = node.args
            new_els = []
            for i in range(len(true_branch.type.types)):
                new_els.append(
                    im.concat_where(cond, im.tuple_get(i, "__tb"), im.tuple_get(i, "__fb"))
                )

            new_node = im.let(("__tb", true_branch), ("__fb", false_branch))(
                im.make_tuple(*new_els)
            )
            # restore domain information
            new_node, _ = infer_domain.infer_expr(
                new_node,
                node.annex.domain,
                keep_existing_domains=True,
                # offset provider not needed as all as_fieldop already have a domain
                offset_provider={},
            )
            return new_node

        return node


expand_tuple = _TransformTupleConcatWhere.apply


class _ExpandConcatWhere(PreserveLocationVisitor, NodeTranslator):
    PRESERVED_ANNEX_ATTRS = (
        "type",
        "domain",
    )

    @classmethod
    def apply(cls, node: ir.Node):
        node = cls().visit(node)
        node = type_inference.SanitizeTypes().visit(node)
        return node

    def visit_FunCall(self, node: ir.FunCall) -> ir.FunCall:
        node = self.generic_visit(node)
        if cpm.is_call_to(node, "concat_where"):
            cond, true_branch, false_branch = node.args
            assert isinstance(cond.type, ts.DomainType)
            position = [im.index(dim) for dim in cond.type.dims]
            refs = symbol_ref_utils.collect_symbol_refs(cond)

            domains = utils.flatten_nested_tuple(node.annex.domain)
            assert all(
                domain == domains[0] for domain in domains
            ), "At this point all `concat_where` arguments should be posed on the same domain."
            assert isinstance(domains[0], domain_utils.SymbolicDomain)
            domain_expr = domains[0].as_expr()

            return im.as_fieldop(
                im.lambda_("__tcw_pos", "__tcw_arg0", "__tcw_arg1", *refs)(
                    im.let(*zip(refs, map(im.deref, refs), strict=True))(
                        im.if_(
                            im.call("in_")(im.deref("__tcw_pos"), cond),
                            im.deref("__tcw_arg0"),
                            im.deref("__tcw_arg1"),
                        )
                    )
                ),
                domain_expr,
            )(im.make_tuple(*position), true_branch, false_branch, *refs)

        return node


expand = _ExpandConcatWhere.apply
