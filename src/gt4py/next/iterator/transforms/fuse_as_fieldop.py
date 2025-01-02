# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Optional

from gt4py import eve
from gt4py.eve import utils as eve_utils
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.transforms import (
    inline_center_deref_lift_vars,
    inline_lambdas,
    inline_lifts,
    merge_let,
    trace_shifts,
)
from gt4py.next.iterator.type_system import (
    inference as type_inference,
    type_specifications as it_ts,
)
from gt4py.next.type_system import type_info, type_specifications as ts


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
        type_inference.copy_type(from_=expr, to=new_expr, allow_untyped=True)

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
            if inner_arg.id in extracted_args:
                assert extracted_args[inner_arg.id] == inner_arg
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


def _unwrap_scan(stencil: itir.Lambda | itir.FunCall):
    if cpm.is_call_to(stencil, "scan"):
        scan_pass, direction, init = stencil.args
        assert isinstance(scan_pass, itir.Lambda)
        # remove scan pass state to be used by caller
        state_param = scan_pass.params[0]
        stencil_like = im.lambda_(*scan_pass.params[1:])(scan_pass.expr)

        def restore_scan(transformed_stencil_like: itir.Lambda):
            new_scan_pass = im.lambda_(state_param, *transformed_stencil_like.params)(
                im.call(transformed_stencil_like)(
                    *(param.id for param in transformed_stencil_like.params)
                )
            )
            return im.call("scan")(new_scan_pass, direction, init)

        return stencil_like, restore_scan

    assert isinstance(stencil, itir.Lambda)
    return stencil, lambda s: s


def fuse_as_fieldop(
    expr: itir.Expr, eligible_args: list[bool], *, uids: eve_utils.UIDGenerator
) -> itir.Expr:
    assert cpm.is_applied_as_fieldop(expr)

    stencil: itir.Lambda = expr.fun.args[0]  # type: ignore[attr-defined]  # ensured by is_applied_as_fieldop
    assert isinstance(expr.fun.args[0], itir.Lambda) or cpm.is_call_to(stencil, "scan")  # type: ignore[attr-defined]  # ensured by is_applied_as_fieldop
    stencil, restore_scan = _unwrap_scan(stencil)

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
    stencil = restore_scan(stencil)

    # simplify stencil directly to keep the tree small
    new_stencil = inline_lambdas.InlineLambdas.apply(
        stencil, opcount_preserving=True, force_inline_lift_args=False
    )
    new_stencil = inline_center_deref_lift_vars.InlineCenterDerefLiftVars.apply(
        new_stencil, is_stencil=True, uids=uids
    )  # to keep the tree small
    new_stencil = merge_let.MergeLet().visit(new_stencil)
    new_stencil = inline_lambdas.InlineLambdas.apply(
        new_stencil, opcount_preserving=True, force_inline_lift_args=True
    )
    new_stencil = inline_lifts.InlineLifts().visit(new_stencil)

    new_node = im.as_fieldop(new_stencil, domain)(*new_args.values())

    type_inference.copy_type(from_=expr, to=new_node, allow_untyped=True)
    return new_node


def _arg_inline_predicate(node: itir.Expr, shifts):
    if _is_tuple_expr_of_literals(node):
        return True
    # TODO(tehrengruber): write test case ensuring scan is not tried to be inlined (e.g. test_call_scan_operator_from_field_operator)
    if (
        is_applied_fieldop := cpm.is_applied_as_fieldop(node)
        and not cpm.is_call_to(node.fun.args[0], "scan")  # type: ignore[attr-defined]  # ensured by cpm.is_applied_as_fieldop
    ) or cpm.is_call_to(node, "if_"):
        # always inline arg if it is an applied fieldop with only a single arg
        if is_applied_fieldop and len(node.args) == 1:
            return True
        # argument is never used, will be removed when inlined
        if len(shifts) == 0:
            return True
        # applied fieldop with list return type must always be inlined as no backend supports this
        assert isinstance(node.type, ts.TypeSpec)
        dtype = type_info.apply_to_primitive_constituents(type_info.extract_dtype, node.type)
        if isinstance(dtype, it_ts.ListType):
            return True
        # only accessed at the center location
        if shifts in [set(), {()}]:
            return True
        # TODO(tehrengruber): Disabled as the InlineCenterDerefLiftVars does not support this yet
        #  and it would increase the size of the tree otherwise.
        # if len(shifts) == 1 and not any(
        #     trace_shifts.Sentinel.ALL_NEIGHBORS in access for access in shifts
        # ):
        #     return True  # noqa: ERA001 [commented-out-code]

    return False


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
    as_fieldop(λ(__arg0, __arg1) → ·__arg0 + ·__arg1, c⟨ IDimₕ: [0, 1[ ⟩)(
      as_fieldop(λ(__arg0, __arg1) → ·__arg0 × ·__arg1, c⟨ IDimₕ: [0, 1[ ⟩)(inp1, inp2), inp3
    )
    >>> print(
    ...     FuseAsFieldOp.apply(
    ...         nested_as_fieldop, offset_provider_type={}, allow_undeclared_symbols=True
    ...     )
    ... )
    as_fieldop(λ(inp1, inp2, inp3) → ·inp1 × ·inp2 + ·inp3, c⟨ IDimₕ: [0, 1[ ⟩)(inp1, inp2, inp3)
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

    def visit_FunCall(self, node: itir.FunCall, **kwargs):
        # inline all fields with list dtype. This needs to happen before the children are visited
        # such that the `as_fieldop` can be fused.
        # TODO(tehrengruber): what should we do in case the field with list dtype is a let itself?
        #  This could duplicate other expressions which we did not intend to duplicate.
        # TODO(tehrengruber): Write test-case. E.g. Adding two sparse fields. Sara observed this
        #  with a cast to a sparse field, but this is likely already covered.
        if cpm.is_let(node):
            eligible_args = [
                isinstance(arg.type, ts.FieldType) and isinstance(arg.type.dtype, it_ts.ListType)
                for arg in node.args
            ]
            if any(eligible_args):
                node = inline_lambdas.inline_lambda(node, eligible_params=eligible_args)
                return self.visit(node)

        if cpm.is_applied_as_fieldop(node):  # don't descend in stencil
            old_node = node
            node = im.as_fieldop(*node.fun.args)(*self.generic_visit(node.args))  # type: ignore[attr-defined]  # ensured by cpm.is_applied_as_fieldop
            type_inference.copy_type(from_=old_node, to=node)
        elif kwargs.get("recurse", True):
            node = self.generic_visit(node, **kwargs)

        if cpm.is_call_to(node, "make_tuple"):
            as_fieldop_args = [arg for arg in node.args if cpm.is_applied_as_fieldop(arg)]
            distinct_domains = set(arg.fun.args[1] for arg in as_fieldop_args)  # type: ignore[attr-defined]  # ensured by cpm.is_applied_as_fieldop
            if len(distinct_domains) != len(as_fieldop_args):
                new_els: list[itir.Expr | None] = [None for _ in node.args]
                as_fieldop_args_by_domain: dict[itir.Expr, list[tuple[int, itir.Expr]]] = {}
                for i, arg in enumerate(node.args):
                    if cpm.is_applied_as_fieldop(arg):
                        assert arg.type
                        _, domain = arg.fun.args  # type: ignore[attr-defined]  # ensured by cpm.is_applied_as_fieldop
                        as_fieldop_args_by_domain.setdefault(domain, [])
                        as_fieldop_args_by_domain[domain].append((i, arg))
                    else:
                        new_els[i] = arg  # keep as is
                let_vars = {}
                for domain, inner_as_fieldop_args in as_fieldop_args_by_domain.items():
                    if len(inner_as_fieldop_args) > 1:
                        var = self.uids.sequential_id(prefix="__fasfop")
                        fused_args = im.op_as_fieldop(lambda *args: im.make_tuple(*args), domain)(
                            *(arg for _, arg in inner_as_fieldop_args)
                        )
                        fused_args.type = ts.TupleType(
                            types=[arg.type for _, arg in inner_as_fieldop_args]  # type: ignore[misc]  # has type is ensured on list creation
                        )
                        # don't recurse into nested args, but only consider newly created `as_fieldop`
                        let_vars[var] = self.visit(fused_args, **{**kwargs, "recurse": False})
                        for outer_tuple_idx, (inner_tuple_idx, _) in enumerate(
                            inner_as_fieldop_args
                        ):
                            new_els[inner_tuple_idx] = im.tuple_get(outer_tuple_idx, var)
                    else:
                        i, arg = inner_as_fieldop_args[0]
                        new_els[i] = arg
                assert not any(el is None for el in new_els)
                assert let_vars
                new_node = im.let(*let_vars.items())(im.make_tuple(*new_els))
                new_node = inline_lambdas.inline_lambda(new_node, opcount_preserving=True)
                return new_node

        if cpm.is_call_to(node.fun, "as_fieldop"):
            node = _canonicalize_as_fieldop(node)

        # when multiple `as_fieldop` calls are fused that use the same argument, this argument
        # might become referenced once only. In order to be able to continue fusing such arguments
        # try inlining here.
        if cpm.is_let(node):
            new_node = inline_lambdas.inline_lambda(node, opcount_preserving=True)
            if new_node is not node:  # nothing has been inlined
                return self.visit(new_node, **kwargs)

        if cpm.is_call_to(node.fun, "as_fieldop"):
            stencil = node.fun.args[0]
            assert isinstance(stencil, itir.Lambda) or cpm.is_call_to(stencil, "scan")
            args: list[itir.Expr] = node.args
            shifts = trace_shifts.trace_stencil(stencil, num_args=len(args))

            eligible_args = [
                _arg_inline_predicate(arg, arg_shifts)
                for arg, arg_shifts in zip(args, shifts, strict=True)
            ]
            if any(eligible_args):
                return self.visit(
                    fuse_as_fieldop(node, eligible_args, uids=self.uids),
                    **{**kwargs, "recurse": False},
                )
        return node
