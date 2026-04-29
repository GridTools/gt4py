# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
import functools

from gt4py import eve
from gt4py.next import utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.type_system import inference as itir_inference
from gt4py.next.type_system import type_specifications as ts


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

        @utils.tree_map(
            collection_type=ts.TupleType,
            result_collection_constructor=lambda _, elts: im.make_tuple(*elts),
            with_path_arg=True,
        )
        def mapper(*args):
            *_el_types, path = args
            return im.call(f)(
                *(
                    functools.reduce(lambda expr, i: im.tuple_get(i, expr), path, im.ref(ref_name))
                    for ref_name in tup_refs
                )
            )

        result = im.let(*zip(tup_refs, tup_args))(mapper(*tup_types))

        itir_inference.reinfer(result)
        return result
