# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from collections import ChainMap

from gt4py import eve
from gt4py.eve import utils as eve_utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im


@dataclasses.dataclass(frozen=True)
class CannonicalizeBoundSymbolNames(eve.NodeTranslator):
    """
    Given an iterator expression cannonicalize all bound symbol names.

    If two such expression are in the same scope and equal so are their values.

    >>> testee1 = im.lambda_("a")(im.plus("a", "b"))
    >>> cannonicalized_testee1 = CannonicalizeBoundSymbolNames.apply(testee1)
    >>> str(cannonicalized_testee1)
    'λ(_csym_1) → _csym_1 + b'

    >>> testee2 = im.lambda_("c")(im.plus("c", "b"))
    >>> cannonicalized_testee2 = CannonicalizeBoundSymbolNames.apply(testee2)
    >>> assert cannonicalized_testee1 == cannonicalized_testee2
    """

    _uids: eve_utils.UIDGenerator = dataclasses.field(
        init=False, repr=False, default_factory=lambda: eve_utils.UIDGenerator(prefix="_csym")
    )

    @classmethod
    def apply(cls, node: itir.Expr):
        return cls().visit(node, sym_map=ChainMap({}))

    def visit_Lambda(self, node: itir.Lambda, *, sym_map: ChainMap):
        sym_map = sym_map.new_child()
        for param in node.params:
            sym_map[str(param.id)] = self._uids.sequential_id()

        return im.lambda_(*sym_map.values())(self.visit(node.expr, sym_map=sym_map))

    def visit_SymRef(self, node: itir.SymRef, *, sym_map: dict[str, str]):
        return im.ref(sym_map[node.id]) if node.id in sym_map else node


def is_equal(a: itir.Expr, b: itir.Expr):
    """
    Return true if two expressions have provably equal values.

    Be aware that this function might return false even though the two expression have the same
    value.

    >>> testee1 = im.lambda_("a")(im.plus("a", "b"))
    >>> testee2 = im.lambda_("c")(im.plus("c", "b"))
    >>> assert is_equal(testee1, testee2)

    >>> testee1 = im.lambda_("a")(im.plus("a", "b"))
    >>> testee2 = im.lambda_("c")(im.plus("c", "d"))
    >>> assert not is_equal(testee1, testee2)
    """
    # TODO(tehrengruber): Extend this function cover more cases than just those with equal
    #  structure, e.g., by also canonicalization of the structure.
    return a == b or (
        CannonicalizeBoundSymbolNames.apply(a) == CannonicalizeBoundSymbolNames.apply(b)
    )


def canonicalize_as_fieldop(expr: itir.FunCall) -> itir.FunCall:
    """
    Canonicalize applied `as_fieldop`s.

    In case the stencil argument is a `deref` wrap it into a lambda such that we have a unified
    format to work with (e.g. each parameter has a name without the need to special case).
    """
    assert cpm.is_applied_as_fieldop(expr)

    stencil = expr.fun.args[0]  # type: ignore[attr-defined]
    domain = expr.fun.args[1] if len(expr.fun.args) > 1 else None  # type: ignore[attr-defined]
    if cpm.is_ref_to(stencil, "deref"):
        stencil = im.lambda_("arg")(im.deref("arg"))
        new_expr = im.as_fieldop(stencil, domain)(*expr.args)

        return new_expr

    return expr


def _remove_let_alias(let_expr: itir.FunCall):
    assert cpm.is_let(let_expr)
    is_aliased_let = True
    for param, arg in zip(let_expr.fun.params, let_expr.args, strict=True):  # type: ignore[attr-defined]  # ensured by cpm.is_let
        is_aliased_let &= cpm.is_ref_to(arg, param.id)
    if is_aliased_let:
        return let_expr.fun.expr  # type: ignore[attr-defined]  # ensured by cpm.is_let
    return let_expr


def unwrap_scan(stencil: itir.Lambda | itir.FunCall):
    """
    If given a scan, extract stencil part of its scan pass and a back-transformation into a scan.

    If a regular stencil is given the stencil is left as-is and the back-transformation is the
    identity function. This function allows treating a scan stencil like a regular stencil during
    a transformation avoiding the complexity introduced by the different IR format.

    >>> scan = im.call("scan")(
    ...     im.lambda_("state", "arg")(im.plus("state", im.deref("arg"))), True, 0.0
    ... )
    >>> stencil, back_trafo = unwrap_scan(scan)
    >>> str(stencil)
    'λ(arg) → state + ·arg'
    >>> str(back_trafo(stencil))
    'scan(λ(state, arg) → state + ·arg, True, 0.0)'

    In case a regular stencil is given it is returned as-is:

    >>> deref_stencil = im.lambda_("it")(im.deref("it"))
    >>> stencil, back_trafo = unwrap_scan(deref_stencil)
    >>> assert stencil == deref_stencil
    """
    if cpm.is_call_to(stencil, "scan"):
        scan_pass, direction, init = stencil.args
        assert isinstance(scan_pass, itir.Lambda)
        # remove scan pass state to be used by caller
        state_param = scan_pass.params[0]
        stencil_like = im.lambda_(*scan_pass.params[1:])(scan_pass.expr)

        def restore_scan(transformed_stencil_like: itir.Lambda):
            new_scan_pass = im.lambda_(state_param, *transformed_stencil_like.params)(
                _remove_let_alias(
                    im.call(transformed_stencil_like)(
                        *(param.id for param in transformed_stencil_like.params)
                    )
                )
            )
            return im.call("scan")(new_scan_pass, direction, init)

        return stencil_like, restore_scan

    assert isinstance(stencil, itir.Lambda)
    return stencil, lambda s: s


def with_altered_arg(node: itir.FunCall, arg_idx: int, new_arg: itir.Expr | str) -> itir.FunCall:
    """Given a itir.FunCall return a new call with one of its argument replaced."""
    return im.call(node.fun)(
        *(arg if i != arg_idx else im.ensure_expr(new_arg) for i, arg in enumerate(node.args))
    )
