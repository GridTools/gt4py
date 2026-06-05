# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Optional

from gt4py.eve import PreserveLocationVisitor
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.transforms.fixed_point_transformation import FixedPointTransformation
from gt4py.next.iterator.type_system import inference as type_inference
from gt4py.next.type_system import type_specifications as ts


class _ExpandTupleArgs(PreserveLocationVisitor, FixedPointTransformation):
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

    def transform(self, node: itir.Node, **kwargs) -> Optional[itir.Node]:
        # `concat_where(cond, {a, b}, {c, d})`
        # -> `{concat_where(cond, a, c), concat_where(cond, a, c)}`
        if not cpm.is_call_to(node, "concat_where") or not isinstance(
            type_inference.reinfer(node.args[1]).type, ts.TupleType
        ):
            return None

        cond, true_branch, false_branch = node.args
        new_els = []
        assert isinstance(true_branch.type, ts.TupleType)
        for i in range(len(true_branch.type.types)):
            new_els.append(
                self.fp_transform(
                    im.concat_where(
                        cond,
                        im.tuple_get(i, im.ref("__tb", true_branch.type)),
                        im.tuple_get(i, im.ref("__fb", false_branch.type)),
                    ),
                    **kwargs,
                )
            )

        new_node = im.let(("__tb", true_branch), ("__fb", false_branch))(im.make_tuple(*new_els))
        return new_node


expand_tuple_args = _ExpandTupleArgs.apply
