# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import itertools
from typing import Optional, Any

from gt4py import eve
from gt4py.eve import utils as eve_utils, concepts
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.transforms import (
    inline_center_deref_lift_vars,
    inline_lambdas,
    inline_lifts,
    trace_shifts,
)
from gt4py.next.iterator.type_system import (
    inference as type_inference,
    type_specifications as it_ts,
)
from gt4py.next.type_system import type_info, type_specifications as ts
from gt4py.next.iterator.transforms import merge_let

def _merge_arguments(
    args1: dict[str, itir.Expr], arg2: dict[str, itir.Expr]
) -> dict[str, itir.Expr]:
    new_args = {**args1}
    for stencil_param, stencil_arg in arg2.items():
        if stencil_param not in new_args:
            new_args[stencil_param] = stencil_arg
        else:
            assert new_args[stencil_param] == stencil_arg
    return new_args


def _canonicalize_as_fieldop(expr: itir.FunCall) -> itir.FunCall:
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
        type_inference.copy_type(from_=expr, to=new_expr)

        return new_expr

    return expr


def _is_tuple_expr_of_literals(expr: itir.Expr):
    if cpm.is_call_to(expr, "make_tuple"):
        return all(_is_tuple_expr_of_literals(arg) for arg in expr.args)
    if cpm.is_call_to(expr, "tuple_get"):
        return _is_tuple_expr_of_literals(expr.args[1])
    return isinstance(expr, itir.Literal)


def _inline_as_fieldop_arg(
    arg: itir.Expr, *, uids: eve_utils.UIDGenerator
) -> tuple[itir.Expr, dict[str, itir.Expr]]:
    assert cpm.is_applied_as_fieldop(arg)
    arg = _canonicalize_as_fieldop(arg)

    stencil, *_ = arg.fun.args  # type: ignore[attr-defined]  # ensured by `is_applied_as_fieldop`
    inner_args: list[itir.Expr] = arg.args
    extracted_args: dict[str, itir.Expr] = {}  # mapping from outer-stencil param to arg

    stencil_params: list[itir.Sym] = []
    stencil_body: itir.Expr = stencil.expr

    for inner_param, inner_arg in zip(stencil.params, inner_args, strict=True):
        if isinstance(inner_arg, itir.SymRef):
            # TODO: this change is required for this case to work correctly: 'as_fieldop(λ(it1, it2) → ·it2 + ·it2,)(__sym_13, __sym_13)'
            if inner_arg.id in extracted_args:  # TODO: assert same value
                alias = stencil_params[list(extracted_args.keys()).index(inner_arg.id)]
                stencil_body = im.let(inner_param, im.ref(alias.id))(stencil_body)
            else:
                stencil_params.append(inner_param)
            extracted_args[inner_arg.id] = inner_arg
        elif isinstance(inner_arg, itir.Literal):
            # note: only literals, not all scalar expressions are required as it doesn't make sense
            # for them to be computed per grid point.
            stencil_body = im.let(inner_param, im.promote_to_const_iterator(inner_arg))(
                stencil_body
            )
        else:
            # a scalar expression, a previously not inlined `as_fieldop` call or an opaque
            # expression e.g. containing a tuple
            stencil_params.append(inner_param)
            new_outer_stencil_param = uids.sequential_id(prefix="__iasfop")
            extracted_args[new_outer_stencil_param] = inner_arg

    return im.lift(im.lambda_(*stencil_params)(stencil_body))(
        *extracted_args.keys()
    ), extracted_args


def fuse_as_fieldop(
    expr: itir.Expr, eligible_args: list[bool], *, uids: eve_utils.UIDGenerator
) -> itir.Expr:
    assert cpm.is_applied_as_fieldop(expr) and isinstance(expr.fun.args[0], itir.Lambda)  # type: ignore[attr-defined]  # ensured by is_applied_as_fieldop

    stencil: itir.Lambda = expr.fun.args[0]  # type: ignore[attr-defined]  # ensured by is_applied_as_fieldop
    domain = expr.fun.args[1] if len(expr.fun.args) > 1 else None  # type: ignore[attr-defined]  # ensured by is_applied_as_fieldop

    args: list[itir.Expr] = expr.args

    new_args: dict[str, itir.Expr] = {}
    new_stencil_body: itir.Expr = stencil.expr

    for eligible, stencil_param, arg in zip(eligible_args, stencil.params, args, strict=True):
        if eligible:
            if cpm.is_applied_as_fieldop(arg):
                pass
            elif cpm.is_call_to(arg, "if_"):
                # TODO(tehrengruber): revisit if we want to inline if_
                type_ = arg.type
                arg = im.op_as_fieldop("if_")(*arg.args)
                arg.type = type_
            elif _is_tuple_expr_of_literals(arg):
                arg = im.op_as_fieldop(im.lambda_()(arg))()
            else:
                raise NotImplementedError()

            inline_expr, extracted_args = _inline_as_fieldop_arg(arg, uids=uids)

            new_stencil_body = im.let(stencil_param, inline_expr)(new_stencil_body)

            new_args = _merge_arguments(new_args, extracted_args)
        else:
            # just a safety check if typing information is available
            if arg.type and not isinstance(arg.type, ts.DeferredType):
                assert isinstance(arg.type, ts.TypeSpec)
                dtype = type_info.apply_to_primitive_constituents(type_info.extract_dtype, arg.type)
                assert not isinstance(dtype, it_ts.ListType)
            new_param: str
            if isinstance(
                arg, itir.SymRef
            ):  # use name from outer scope (optional, just to get a nice IR)
                new_param = arg.id
                new_stencil_body = im.let(stencil_param.id, arg.id)(new_stencil_body)
            else:
                new_param = stencil_param.id
            new_args = _merge_arguments(new_args, {new_param: arg})

    stencil = im.lambda_(*new_args.keys())(new_stencil_body)

    # simplify stencil directly to keep the tree small
    new_stencil = inline_lambdas.InlineLambdas.apply(
        stencil, opcount_preserving=True, force_inline_lift_args=False
    )
    trace_shifts.trace_stencil(
        # TODO: required for InlineCenterDerefLiftVars on stencil level, fix pass instead
        new_stencil, num_args=len(new_args), save_to_annex=True
    )
    new_stencil = inline_center_deref_lift_vars.InlineCenterDerefLiftVars.apply(
        new_stencil, uids=uids
    )  # to keep the tree small
    new_stencil = merge_let.MergeLet().visit(new_stencil)
    new_stencil = inline_lambdas.InlineLambdas.apply(
        new_stencil, opcount_preserving=True, force_inline_lift_args=True
    )
    new_stencil = inline_lifts.InlineLifts().visit(new_stencil)

    new_node = im.as_fieldop(new_stencil, domain)(*new_args.values())

    type_inference.copy_type(from_=expr, to=new_node)
    return new_node

def _is_pointwise_as_fieldop(node: itir.Expr) -> bool:
    # TODO: not only check num shifts == 1 but also not ALL_NEIGHBORS to avoid nested neighbor inline
    # TODO: maybe only fuse when the number of reads does not increase, e.g. do not inline here
    # because we have two args:
    # let tmp = as_fieldop(...)(a, b)
    #   {as_fieldop(...)(tmp), as_fieldop(...)(tmp)}
    #
    if not cpm.is_applied_as_fieldop(node):
        return False

    stencil: itir.Lambda = node.fun.args[0]
    shifts = trace_shifts.trace_stencil(stencil, num_args=len(node.args))
    #is_pointwise = all(len(arg_shifts) <= 1 for arg_shifts in shifts)
    is_pointwise = all(len(arg_shifts) <= 1 for arg_shifts in shifts)
    is_pointwise &= not any(trace_shifts.Sentinel.ALL_NEIGHBORS in shift_chain for arg_shifts in shifts for shift_chain in arg_shifts)
    return is_pointwise

def _is_center_pos_as_fieldop(node: itir.Expr) -> bool:
    if not cpm.is_applied_as_fieldop(node):
        return False

    stencil: itir.Lambda = node.fun.args[0]
    shifts = trace_shifts.trace_stencil(stencil, num_args=len(node.args))
    is_center = all(arg_shifts in [set(), {()}] for arg_shifts in shifts)
    return is_center


@dataclasses.dataclass
class FuseAsFieldOp(eve.NodeTranslator):
    """
    Merge multiple `as_fieldop` calls into one.

    >>> from gt4py import next as gtx
    >>> from gt4py.next.iterator.ir_utils import ir_makers as im
    >>> IDim = gtx.Dimension("IDim")
    >>> field_type = ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32))
    >>> d = im.domain("cartesian_domain", {IDim: (0, 1)})
    >>> nested_as_fieldop = im.op_as_fieldop("plus", d)(
    ...     im.op_as_fieldop("multiplies", d)(
    ...         im.ref("inp1", field_type), im.ref("inp2", field_type)
    ...     ),
    ...     im.ref("inp3", field_type),
    ... )
    >>> print(nested_as_fieldop)
    as_fieldop(λ(__arg0, __arg1) → ·__arg0 + ·__arg1, c⟨ IDimₕ: [0, 1) ⟩)(
      as_fieldop(λ(__arg0, __arg1) → ·__arg0 × ·__arg1, c⟨ IDimₕ: [0, 1) ⟩)(inp1, inp2), inp3
    )
    >>> print(
    ...     FuseAsFieldOp.apply(
    ...         nested_as_fieldop, offset_provider_type={}, allow_undeclared_symbols=True
    ...     )
    ... )
    as_fieldop(λ(inp1, inp2, inp3) → ·inp1 × ·inp2 + ·inp3, c⟨ IDimₕ: [0, 1) ⟩)(inp1, inp2, inp3)
    """  # noqa: RUF002  # ignore ambiguous multiplication character

    uids: eve_utils.UIDGenerator

    @classmethod
    def apply(
        cls,
        node: itir.Program,
        *,
        offset_provider_type: common.OffsetProviderType,
        uids: Optional[eve_utils.UIDGenerator] = None,
        allow_undeclared_symbols=False,
    ):
        node = type_inference.infer(
            node,
            offset_provider_type=offset_provider_type,
            allow_undeclared_symbols=allow_undeclared_symbols,
        )

        if not uids:
            uids = eve_utils.UIDGenerator()

        return cls(uids=uids).visit(node)

#     def visit(self, node: concepts.RootNode, **kwargs: Any) -> Any:
#         if str(node) == """as_fieldop(λ(hdef_ic_wpᐞ0) → cast_(·hdef_ic_wpᐞ0 × ·hdef_ic_wpᐞ0, float64),
#            u⟨ Cellₕ: [horizontal_start, horizontal_end), Kᵥ: [vertical_start, vertical_end) ⟩)(
#   hdef_ic_wpᐞ0
# )""" or str(node) == 'hdef_ic_wpᐞ0':
#             breakpoint()
#         return super().visit(node, **kwargs)

    def visit_FunCall(self, node: itir.FunCall):
        # tmp = a + b + c + d
        # tmp1 = tmp+1
        # tmp2 = tmp+2
        # out1 = tmp1+tmp2
        # out2 = tmp1+d

        # tmp = a + b + c + d
        # tmp1 = tmp+1
        # tmp2 = tmp+2
        # out1 = tmp1+tmp2
        # out2 = tmp+d
        if cpm.is_applied_as_fieldop(node):  # don't descend in stencil
            old_node = node
            node = im.as_fieldop(*node.fun.args)(*self.generic_visit(node.args))
            type_inference.copy_type(from_=old_node, to=node)
        else:
            node = self.generic_visit(node)

        if cpm.is_call_to(node, "make_tuple"):
            as_fieldop_args = [arg for arg in node.args if cpm.is_applied_as_fieldop(arg)]
            distinct_domains = set(arg.fun.args[1] for arg in as_fieldop_args)
            if len(distinct_domains) != len(as_fieldop_args):
                new_els = [None for _ in node.args]
                as_fieldop_args_by_domain: dict[itir.Expr, tuple[int, itir.Expr]] = {}
                for i, arg in enumerate(node.args):
                    if cpm.is_applied_as_fieldop(arg):
                        _, domain = arg.fun.args
                        as_fieldop_args_by_domain.setdefault(domain, [])
                        as_fieldop_args_by_domain[domain].append((i, arg))
                    else:
                        new_els[i] = arg  # keep as is
                let_vars = {}
                for domain, inner_as_fieldop_args in as_fieldop_args_by_domain.items():
                    if len(inner_as_fieldop_args) > 1:
                        var = self.uids.sequential_id(prefix="__fasfop")
                        fused_args = im.op_as_fieldop(lambda *args: im.make_tuple(*args), domain)(*(arg for _, arg in inner_as_fieldop_args))
                        fused_args.type = ts.TupleType(types=[arg.type for _, arg in inner_as_fieldop_args])
                        let_vars[var] = self.visit(fused_args)  # TODO: do not recurse into args
                        for outer_tuple_idx, (inner_tuple_idx, _) in enumerate(inner_as_fieldop_args):
                            new_els[inner_tuple_idx] = im.tuple_get(outer_tuple_idx, var)
                    else:
                        i, arg = inner_as_fieldop_args[0]
                        new_els[i].append(arg)
                assert not any(el is None for el in new_els)
                assert let_vars
                new_node = im.let(*let_vars.items())(im.make_tuple(*new_els))
                new_node = inline_lambdas.inline_lambda(new_node, opcount_preserving=True)
                return new_node

        if cpm.is_call_to(node.fun, "as_fieldop"):
            node = _canonicalize_as_fieldop(node)

        if cpm.is_let(node):
            new_node = inline_lambdas.inline_lambda(node, opcount_preserving=True)
            if new_node is node:  # nothing has been inlined
                return new_node
            return self.visit(new_node)
            # TODO
            #eligible_args = [_is_center_pos_as_fieldop(arg) for arg in node.args]
            #if any(eligible_args):
            #    new_node = inline_lambdas.inline_lambda(node, eligible_params=eligible_args)
            #    return self.visit(new_node)

        if cpm.is_call_to(node.fun, "as_fieldop") and isinstance(node.fun.args[0], itir.Lambda):
            stencil: itir.Lambda = node.fun.args[0]
            args: list[itir.Expr] = node.args
            shifts = trace_shifts.trace_stencil(stencil)

            eligible_args = []
            for arg, arg_shifts in zip(args, shifts, strict=True):
                assert isinstance(arg.type, ts.TypeSpec)
                dtype = type_info.apply_to_primitive_constituents(type_info.extract_dtype, arg.type)
                # TODO(tehrengruber): make this configurable
                is_eligible = False
                is_eligible |= _is_tuple_expr_of_literals(arg)
                if isinstance(arg, itir.FunCall) and (
                    cpm.is_call_to(arg.fun, "as_fieldop")
                    and isinstance(arg.fun.args[0], itir.Lambda)
                    or cpm.is_call_to(arg, "if_")  # TODO: this will likely lead to an oob, maybe just on scalars?
                ):
                    is_eligible |= isinstance(dtype, it_ts.ListType)
                    is_eligible |= len(arg_shifts) == 0
                    # if the argument is only accessed at one neighbor location
                    #is_eligible |= len(arg_shifts) == 1 and not any(
                    #    trace_shifts.Sentinel.ALL_NEIGHBORS in shift_chain for shift_chain in arg_shifts)
                    is_eligible |= arg_shifts in [set(), {()}]
                    #is_eligible |= _is_pointwise_as_fieldop(arg)

                    # if cpm.is_applied_as_fieldop(node):
                    #     stencil: itir.Lambda = node.fun.args[0]
                    #     inner_shifts = trace_shifts.trace_stencil(stencil, num_args=len(node.args))
                    #     # is_pointwise = all(len(arg_shifts) <= 1 for arg_shifts in shifts)
                    #     total_inner_shifts = {(*outer_shift_chain, *inner_shift_chain) for outer_shift_chain in arg_shifts for inner_shift_chain in inner_arg_shifts for inner_arg_shifts in inner_shifts}
                    #     is_eligible |=
                    #     #is_pointwise = not any(
                    #     #    trace_shifts.Sentinel.ALL_NEIGHBORS in shift_chain for arg_shifts in shifts
                    #     #    for shift_chain in arg_shifts)
                    #     return is_pointwise
                    #if not is_eligible:
                    #    breakpoint()
                eligible_args.append(is_eligible)
            if any(eligible_args):
                # TODO: fuse again?
                return fuse_as_fieldop(node, eligible_args, uids=self.uids)
        return node
