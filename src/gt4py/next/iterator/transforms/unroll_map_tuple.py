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


@dataclasses.dataclass
class UnrollMapTuple(eve.NodeTranslator):
    PRESERVED_ANNEX_ATTRS = ("domain",)

    uids: utils.IDGeneratorPool

    @classmethod
    def apply(cls, program: itir.Program, *, uids: utils.IDGeneratorPool):
        return cls(uids=uids).visit(program)

    def visit_FunCall(self, node: itir.Expr):
        node = self.generic_visit(node)

        if cpm.is_call_to(node.fun, "map_tuple"):
            # TODO: we have to duplicate the function here since the domain inference can not handle them yet
            f = node.fun.args[0]
            tup = node.args[0]
            itir_inference.reinfer(tup)
            assert isinstance(tup.type, ts.TupleType)
            tup_ref = next(self.uids["_ump"])

            result = im.let(tup_ref, tup)(
                im.make_tuple(
                    *(im.call(f)(im.tuple_get(i, tup_ref)) for i in range(len(tup.type.types)))
                )
            )
            itir_inference.reinfer(result)

            return result
        return node
