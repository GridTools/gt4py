# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses

from gt4py import eve
from gt4py.next import utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.type_system import inference as itir_inference
from gt4py.next.type_system import type_specifications as ts


def _unroll(
    f: itir.Expr,
    tup_types: list[ts.TupleType],
    tup_exprs: list[itir.Expr],
) -> itir.Expr:
    """Recursively expand ``tree_map(f)(tup0, tup1, ...)`` into ``make_tuple`` / ``tuple_get``."""
    n = len(tup_types[0].types)

    elements: list[itir.Expr] = []
    for i in range(n):
        child_types = [t.types[i] for t in tup_types]
        child_exprs = [im.tuple_get(i, e) for e in tup_exprs]

        if all(isinstance(ct, ts.TupleType) for ct in child_types):
            nested_types = [ct for ct in child_types if isinstance(ct, ts.TupleType)]
            elements.append(_unroll(f, nested_types, child_exprs))
        else:
            elements.append(im.call(f)(*child_exprs))

    return im.make_tuple(*elements)


@dataclasses.dataclass
class UnrollTreeMap(eve.NodeTranslator):
    PRESERVED_ANNEX_ATTRS = ("domain",)

    uids: utils.IDGeneratorPool

    @classmethod
    def apply(cls, program: itir.Program, *, uids: utils.IDGeneratorPool):
        return cls(uids=uids).visit(program)

    def visit_FunCall(self, node: itir.FunCall):
        node = self.generic_visit(node)

        if not cpm.is_call_to(node.fun, "tree_map"):
            return node

        f = node.fun.args[0]
        tup_args = node.args
        tup_types: list[ts.TupleType] = []
        for tup in tup_args:
            itir_inference.reinfer(tup)
            assert isinstance(tup.type, ts.TupleType)
            tup_types.append(tup.type)

        tup_refs = [next(self.uids["_utm"]) for _ in tup_args]
        body = _unroll(f, tup_types, [im.ref(r) for r in tup_refs])

        result = body
        for ref_name, tup in reversed(list(zip(tup_refs, tup_args))):
            result = im.let(ref_name, tup)(result)

        itir_inference.reinfer(result)
        return result
