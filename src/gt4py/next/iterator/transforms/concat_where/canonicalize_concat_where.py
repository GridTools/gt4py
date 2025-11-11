# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import TypeVar

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im


PRG = TypeVar("PRG", bound=itir.Program | itir.Expr)


class _CanonicalizeConcatWhere(PreserveLocationVisitor, NodeTranslator):
    PRESERVED_ANNEX_ATTRS = (
        "type",
        "domain",
    )

    @classmethod
    def apply(cls, node: PRG) -> PRG:
        return cls().visit(node)

    def visit_FunCall(self, node: itir.FunCall) -> itir.Expr:
        node = self.generic_visit(node)
        # `concat_where((c1, v1), (c2, v2), (c3, v3),..., default)`
        # -> `{concat_where(c1, v1, concat_where(c2, v2, concat_where(c3, v3, default))}`

        if not cpm.is_call_to(node, "concat_where") or not cpm.is_call_to(
            node.args[0], "make_tuple"
        ):
            return node

        *pairs, default = node.args

        if len(pairs) == 0:
            return node

        result = default
        for pair in reversed(pairs):
            cond, value = pair.args if isinstance(pair, itir.FunCall) else pair
            result = im.concat_where(cond, value, result)

        return result


canonicalize_concat_where = _CanonicalizeConcatWhere.apply
