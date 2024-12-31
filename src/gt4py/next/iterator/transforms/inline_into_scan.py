# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
# FIXME[#1582](tehrengruber): This transformation is not used anymore. Decide on its fate.
from typing import Sequence, TypeGuard

from gt4py import eve
from gt4py.eve import NodeTranslator, traits
from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms import symbol_ref_utils
from gt4py.next.iterator.transforms.inline_lambdas import inline_lambda
from gt4py.next.iterator.transforms.inline_lifts import InlineLifts


def _is_scan(node: ir.Node) -> TypeGuard[ir.FunCall]:
    return (
        isinstance(node, ir.FunCall)
        and isinstance(node.fun, ir.FunCall)
        and node.fun.fun == ir.SymRef(id="scan")
    )


def _extract_symrefs(nodes: Sequence[ir.Expr], symtable: dict[eve.SymbolName, ir.Sym]) -> list[str]:
    return symbol_ref_utils.collect_symbol_refs(
        nodes, symbol_ref_utils.get_user_defined_symbols(symtable)
    )


def _contains_scan(node: ir.Expr) -> bool:
    return bool(
        node.pre_walk_values().if_isinstance(ir.SymRef).filter(lambda x: x.id == "scan").to_list()
    )


def _should_inline(node: ir.FunCall) -> bool:
    # TODO(havogt): This fails if the scan is captured. Check if the inferred type is a column, once available.
    return not any(_contains_scan(arg) for arg in node.args)


def _lambda_and_lift_inliner(node: ir.FunCall) -> ir.FunCall:
    inlined = inline_lambda(node, opcount_preserving=False, force_inline_lift_args=True)
    inlined = InlineLifts().visit(inlined)
    return inlined


class InlineIntoScan(
    traits.PreserveLocationVisitor, traits.VisitorWithSymbolTableTrait, NodeTranslator
):
    """
    Inline non-SymRef arguments into the scan.

    Preconditions:
      - `FunctionDefinitions` are inlined

    Example:
        scan(λ(state, isym0, isym1) → body(state, isym0, isym1), forward, init)(sym0, f(sym0,sym1,sym2))
    to
        scan(λ(state, sym0, sym1, sym2) → (λ(isym0, isym1) → body(state, isym0, isym1))(sym0, f(sym0,sym1,sym2)), forward, init)(sym0, sym1,sym2)

    Algorithm:
      - take args of scan: `sym0`, `f(sym0, sym1, sym2)`
      - extract all symrefs that are not builtins: `sym0`, `sym1`, `sym2`
      - create a lambda with first (state/carry) param taken from original scanpass (`state`) and new Symbols with the name of the extracted symrefs: `λ(state, sym0, sym1, sym2)`
      - the body is a call to the original scanpass, but with `state` param removed `λ(isym0, isym1) → body(state, isym0, isym1)` (`state` is captured)
      - it is called with the original args of the scan: `sym0, f(sym0,sym1,sym2)`
      - wrap the new scanpass in a scan call with the original `forward` and `init`
      - call it with the extrated symrefs
      - note: there is no symbol clash
    """

    def visit_FunCall(self, node: ir.FunCall, **kwargs):
        if _is_scan(node) and _should_inline(node):
            original_scan_args = node.args
            original_scan_call = node.fun
            assert isinstance(original_scan_call, ir.FunCall)
            refs_in_args = _extract_symrefs(original_scan_args, kwargs["symtable"])
            original_scanpass = original_scan_call.args[0]
            assert isinstance(original_scanpass, ir.Lambda)

            new_scanpass_body = ir.FunCall(
                fun=ir.Lambda(params=[*original_scanpass.params[1:]], expr=original_scanpass.expr),
                args=original_scan_args,
            )
            new_scanpass_body = _lambda_and_lift_inliner(new_scanpass_body)
            new_scanpass = ir.Lambda(
                params=[original_scanpass.params[0], *(ir.Sym(id=ref) for ref in refs_in_args)],
                expr=new_scanpass_body,
            )
            new_scan = ir.FunCall(
                fun=ir.SymRef(id="scan"), args=[new_scanpass, *original_scan_call.args[1:]]
            )
            return ir.FunCall(fun=new_scan, args=[ir.SymRef(id=ref) for ref in refs_in_args])
        return self.generic_visit(node, **kwargs)
