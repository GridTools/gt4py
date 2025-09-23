# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from collections import ChainMap
from typing import Callable, Iterable, TypeVar

from gt4py import eve
from gt4py._core import definitions as core_defs
from gt4py.eve import utils as eve_utils
from gt4py.next import common
from gt4py.next.iterator import embedded, ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.transforms import inline_lambdas
from gt4py.next.type_system import type_specifications as ts


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
    def apply(cls, node: itir.Expr) -> itir.Expr:
        return cls().visit(node, sym_map=ChainMap({}))

    def visit_Lambda(self, node: itir.Lambda, *, sym_map: ChainMap) -> itir.Lambda:
        sym_map = sym_map.new_child()
        for param in node.params:
            sym_map[str(param.id)] = self._uids.sequential_id()

        return im.lambda_(*sym_map.values())(self.visit(node.expr, sym_map=sym_map))

    def visit_SymRef(self, node: itir.SymRef, *, sym_map: dict[str, str]) -> itir.SymRef:
        return im.ref(sym_map[node.id]) if node.id in sym_map else node


def is_equal(a: itir.Expr, b: itir.Expr) -> bool:
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

    stencil = expr.fun.args[0]
    domain = expr.fun.args[1] if len(expr.fun.args) > 1 else None
    if cpm.is_ref_to(stencil, "deref"):
        stencil = im.lambda_("arg")(im.deref("arg"))
        new_expr = im.as_fieldop(stencil, domain)(*expr.args)

        return new_expr

    return expr


def _remove_let_alias(let_expr: itir.FunCall) -> itir.FunCall:
    assert cpm.is_let(let_expr)
    is_aliased_let = True
    for param, arg in zip(let_expr.fun.params, let_expr.args, strict=True):
        is_aliased_let &= cpm.is_ref_to(arg, param.id)
    if is_aliased_let:
        assert isinstance(let_expr.fun.expr, itir.FunCall)
        return let_expr.fun.expr
    return let_expr


def unwrap_scan(
    stencil: itir.Lambda | itir.FunCall,
) -> tuple[itir.Lambda, Callable[[itir.Lambda], itir.FunCall | itir.Lambda]]:
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


def extract_projector(
    node: itir.Expr, cur_projector=None, _depth=0
) -> tuple[itir.Lambda | None, itir.Expr]:
    """
    Extract the projector from an expression (only useful for `scan`s).

    A projector is an expression that consists only of `make_tuple` of `tuple_get` of the same expression,
    possibly in a let statement.

    This is needed for expressions like `as_fieldop(scan(λ(state, val) → {val, state[0]+val}))(inp)[1]`,
    where only element 1 of the state is used. In this example the projector is `λ(_proj) → _proj[1]`.

    Returns the projector and the expression it is applied to.

    Note: Supports only unary projectors. Extend to multi-parameter projectors if needed.
    """
    projector: itir.Lambda | None = None
    expr = node
    if cpm.is_let(node) and len(node.fun.params) == 1:
        # a single param let, it's a projector if the let value aka `node.fun.expr` is a projector
        # > let val = expr
        # >  val[x]
        # > end
        # ->
        # `λ(val) → val[x]`, `expr`
        is_projector, _ = extract_projector(node.fun.expr)
        if is_projector is not None:
            # we can directly use this as projector
            projector = node.fun
            expr = node.args[0]
        else:
            projector = None
            expr = node
    elif cpm.is_call_to(node, "tuple_get"):
        # `expr[x]` -> `λ(_proj) → _proj[x]`, `expr`
        index = node.args[0]
        assert isinstance(index, itir.Literal), index
        projector = im.lambda_(f"_proj{_depth}")(im.tuple_get(index, f"_proj{_depth}"))
        expr = node.args[1]
    elif cpm.is_call_to(node, "make_tuple"):
        # `make_tuple(expr[x0], expr[x1], ...)` -> `λ(_proj) → {_proj[x0], _proj[x1], ...}`, `expr`
        projectors, exprs = zip(*(extract_projector(arg) for arg in node.args))
        if all(p is not None for p in projectors) and all(e == exprs[0] for e in exprs):
            projector = im.lambda_(f"_proj{_depth}")(
                im.call("make_tuple")(*(im.call(p)(im.ref(f"_proj{_depth}")) for p in projectors))
            )
            expr = exprs[0]

    if projector is None:
        return cur_projector, expr
    else:
        # nested projectors, e.g. `expr[x][y]`
        projector = projector if cur_projector is None else im.compose(cur_projector, projector)
        projector = inline_lambdas.InlineLambdas.apply(projector)
        return extract_projector(expr, projector, _depth + 1)


def grid_type_from_domain(domain: itir.FunCall) -> common.GridType:
    if cpm.is_call_to(domain, "cartesian_domain"):
        return common.GridType.CARTESIAN
    else:
        assert cpm.is_call_to(domain, "unstructured_domain")
        return common.GridType.UNSTRUCTURED


def _flatten_tuple_expr(domain_expr: itir.Expr) -> tuple[itir.Expr]:
    if cpm.is_call_to(domain_expr, "make_tuple"):
        return sum((_flatten_tuple_expr(arg) for arg in domain_expr.args), start=())
    else:
        return (domain_expr,)


def grid_type_from_program(program: itir.Program) -> common.GridType:
    domain_exprs = program.walk_values().if_isinstance(itir.SetAt).getattr("domain").to_set()
    domains = sum((_flatten_tuple_expr(domain_expr) for domain_expr in domain_exprs), start=())
    grid_types = {grid_type_from_domain(d) for d in domains}
    if len(grid_types) != 1:
        raise ValueError(
            f"Found 'set_at' with more than one 'GridType': '{grid_types}'. This is currently not supported."
        )
    return grid_types.pop()


SymOrStr = TypeVar("SymOrStr", itir.Sym, str)


def unique_symbol(sym: SymOrStr, reserved_names: Iterable[str]) -> SymOrStr:
    """
    Give a symbol and a list of reserved names return a unique symbol with similar or equal name.
    """
    if isinstance(sym, itir.Sym):
        return im.sym(unique_symbol(sym.id, reserved_names), sym.type)  # type: ignore[return-value]  # mypy not smart enough

    assert isinstance(sym, str)
    name: str = sym
    while name in reserved_names:
        name = name + "_"

    return name


def value_from_literal(literal: itir.Literal) -> core_defs.Scalar:
    if literal.type.kind == ts.ScalarKind.BOOL:
        assert literal.value in ["True", "False"]
        return literal.value == "True"
    return getattr(embedded, str(literal.type))(literal.value)
