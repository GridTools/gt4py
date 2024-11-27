# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Optional

import gt4py.next.iterator.ir_utils.common_pattern_matcher as cpm
from gt4py import eve
from gt4py.eve import utils as eve_utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms import fuse_as_fieldop, inline_lambdas, trace_shifts
from gt4py.next.iterator.transforms.symbol_ref_utils import collect_symbol_refs


def _dynamic_shift_args(node: itir.Expr) -> None | list[bool]:
    if not cpm.is_applied_as_fieldop(node):
        return None
    params_shifts = trace_shifts.trace_stencil(
        node.fun.args[0],  # type: ignore[attr-defined]  # ensured by is_applied_as_fieldop
        num_args=len(node.args),
        save_to_annex=True,
    )
    dynamic_shifts = [
        any(trace_shifts.Sentinel.VALUE in shifts for shifts in param_shifts)
        for param_shifts in params_shifts
    ]
    return dynamic_shifts


@dataclasses.dataclass
class InlineDynamicShifts(eve.NodeTranslator, eve.VisitorWithSymbolTableTrait):
    uids: eve_utils.UIDGenerator

    @classmethod
    def apply(cls, node: itir.Program, uids: Optional[eve_utils.UIDGenerator] = None):
        if not uids:
            uids = eve_utils.UIDGenerator()

        return cls(uids=uids).visit(node)

    def visit_FunCall(self, node: itir.FunCall, **kwargs):
        node = self.generic_visit(node, **kwargs)

        if cpm.is_let(node) and (
            dynamic_shift_args := _dynamic_shift_args(let_body := node.fun.expr)  # type: ignore[attr-defined]  # ensured by is_let
        ):
            inline_let_params = {p.id: False for p in node.fun.params}  # type: ignore[attr-defined]  # ensured by is_let

            for inp, is_dynamic_shift_arg in zip(let_body.args, dynamic_shift_args, strict=True):
                for ref in collect_symbol_refs(inp):
                    if ref in inline_let_params and is_dynamic_shift_arg:
                        inline_let_params[ref] = True

            if any(inline_let_params):
                node = inline_lambdas.inline_lambda(
                    node, eligible_params=list(inline_let_params.values())
                )

        if dynamic_shift_args := _dynamic_shift_args(node):
            assert len(node.fun.args) in [1, 2]  # type: ignore[attr-defined]  # ensured by is_applied_as_fieldop in _dynamic_shift_args
            fuse_args = [
                not isinstance(inp, itir.SymRef) and dynamic_shift_arg
                for inp, dynamic_shift_arg in zip(node.args, dynamic_shift_args, strict=True)
            ]
            if any(fuse_args):
                return fuse_as_fieldop.fuse_as_fieldop(node, fuse_args, uids=self.uids)

        return node
