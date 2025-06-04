# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.type_system import inference as type_inference
from gt4py.next.type_system import type_specifications as ts


class _ExpandTupleArgs(PreserveLocationVisitor, NodeTranslator):
    PRESERVED_ANNEX_ATTRS = (
        "type",
        "domain",
    )

    @classmethod
    def apply(
        cls,
        node: itir.Node,
        *,
        offset_provider_type: common.OffsetProviderType,
        allow_undeclared_symbols: bool = False,
    ) -> itir.Node:
        node = type_inference.infer(
            node,
            offset_provider_type=offset_provider_type,
            allow_undeclared_symbols=allow_undeclared_symbols,
        )
        return cls().visit(node)

    def visit_FunCall(self, node: itir.FunCall) -> itir.FunCall:
        node = self.generic_visit(node)

        # `concat_where(cond, {a, b}, {c, d})`
        # -> `{concat_where(cond, a, c), concat_where(cond, a, c)}`
        if cpm.is_call_to(node, "concat_where") and isinstance(type_inference.reinfer(node.args[1]).type, ts.TupleType):
            cond, true_branch, false_branch = node.args
            new_els = []
            assert isinstance(true_branch.type, ts.TupleType)
            for i in range(len(true_branch.type.types)):
                new_els.append(
                    im.concat_where(cond, im.tuple_get(i, "__tb"), im.tuple_get(i, "__fb"))
                )

            new_node = im.let(("__tb", true_branch), ("__fb", false_branch))(
                im.make_tuple(*new_els)
            )
            return new_node

        return node


expand_tuple_args = _ExpandTupleArgs.apply
