# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.type_system import type_specifications as ts


class NormalizeShifts(PreserveLocationVisitor, NodeTranslator):
    def visit_FunCall(self, node: ir.FunCall):
        node = self.generic_visit(node)
        if (
            isinstance(node.fun, ir.FunCall)
            and isinstance(node.fun.fun, ir.SymRef)
            and node.fun.fun.id == "shift"
            and node.args
            and isinstance(node.args[0], ir.FunCall)
            and isinstance(node.args[0].fun, ir.FunCall)
            and isinstance(node.args[0].fun.fun, ir.SymRef)
            and node.args[0].fun.fun.id == "shift"
        ):
            # shift(args1...)(shift(args2...)(it)) -> shift(args2..., args1...)(it)
            assert len(node.args) == 1
            return ir.FunCall(
                fun=ir.FunCall(
                    fun=ir.SymRef(id="shift"), args=node.args[0].fun.args + node.fun.args
                ),
                args=node.args[0].args,
            )
        return node


class CanonicalizeShiftOffsets(PreserveLocationVisitor, NodeTranslator):
    """
    Turn integral `Literal` shift offsets into `OffsetLiteral`s.

    A shift offset that is only known after inlining, e.g. a static program argument used as in
    `field(IDim + offset)`, reaches the shift as an `ir.Literal`. Downstream consumers (shift
    tracing, backend lowering) expect the canonical `ir.OffsetLiteral` that `ir_makers.shift`
    produces for offsets that are literal from the start.
    """

    def visit_FunCall(self, node: ir.FunCall) -> ir.Node:
        node = self.generic_visit(node)
        if cpm.is_call_to(node, "shift"):
            return ir.FunCall(
                fun=node.fun,
                args=[
                    ir.OffsetLiteral(value=int(arg.value))
                    if isinstance(arg, ir.Literal)
                    and arg.type.kind in (ts.ScalarKind.INT32, ts.ScalarKind.INT64)
                    else arg
                    for arg in node.args
                ],
            )
        return node
