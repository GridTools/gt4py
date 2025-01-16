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
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.iterator.transforms import (
    inline_center_deref_lift_vars,
    inline_lambdas,
    inline_lifts,
    merge_let,
    trace_shifts,
)
from gt4py.next.iterator.type_system import inference as type_inference
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
    """
    If given a scan, extract stencil part of its scan pass and a back-transformation into a scan.

    If a regular stencil is given the stencil is left as-is and the back-transformation is the
    identity function. This function allows treating a scan stencil like a regular stencil during
    a transformation avoiding the complexity introduced by the different IR format.

    >>> scan = im.call("scan")(
    ...     im.lambda_("state", "arg")(im.plus("state", im.deref("arg"))), True, 0.0
    ... )
    >>> stencil, back_trafo = _unwrap_scan(scan)
    >>> str(stencil)
    'λ(arg) → state + ·arg'
    >>> str(back_trafo(stencil))
    'scan(λ(state, arg) → (λ(arg) → state + ·arg)(arg), True, 0.0)'

    In case a regular stencil is given it is returned as-is:

    >>> deref_stencil = im.lambda_("it")(im.deref("it"))
    >>> stencil, back_trafo = _unwrap_scan(deref_stencil)
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
                arg = im.op_as_fieldop("if_")(*arg.args)
            elif _is_tuple_expr_of_literals(arg):
                arg = im.op_as_fieldop(im.lambda_()(arg))()
            else:
                raise NotImplementedError()

            inline_expr, extracted_args = _inline_as_fieldop_arg(arg, uids=uids)

            new_stencil_body = im.let(stencil_param, inline_expr)(new_stencil_body)

            new_args = _merge_arguments(new_args, extracted_args)
        else:
            # just a safety check if typing information is available
            type_inference.reinfer(arg)
            if arg.type and not isinstance(arg.type, ts.DeferredType):
                assert isinstance(arg.type, ts.TypeSpec)
                dtype = type_info.apply_to_primitive_constituents(type_info.extract_dtype, arg.type)
                assert not isinstance(dtype, ts.ListType)
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

    return new_node


def _arg_inline_predicate(node: itir.Expr, shifts):
    if _is_tuple_expr_of_literals(node):
        return True

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
        type_inference.reinfer(node)
        assert isinstance(node.type, ts.TypeSpec)
        dtype = type_info.apply_to_primitive_constituents(type_info.extract_dtype, node.type)
        if isinstance(dtype, ts.ListType):
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


def _make_tuple_element_inline_predicate(node: itir.Expr):
    if cpm.is_applied_as_fieldop(node):  # field, or tuple of fields
        return True
    if isinstance(node.type, ts.FieldType) and isinstance(node, itir.SymRef):
        return True
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

    PRESERVED_ANNEX_ATTRS = ("domain",)

    uids: eve_utils.UIDGenerator

    @classmethod
    def apply(
        cls,
        node: itir.Program,
        *,
        offset_provider_type: common.OffsetProviderType,
        uids: Optional[eve_utils.UIDGenerator] = None,
        allow_undeclared_symbols=False,
        within_set_at_expr: Optional[bool] = None,
    ):
        node = type_inference.infer(
            node,
            offset_provider_type=offset_provider_type,
            allow_undeclared_symbols=allow_undeclared_symbols,
        )

        if within_set_at_expr is None:
            within_set_at_expr = not isinstance(node, itir.Program)

        if not uids:
            uids = eve_utils.UIDGenerator()

        return cls(uids=uids).visit(node, within_set_at_expr=within_set_at_expr)

    def visit(self, node, **kwargs):
        new_node = super().visit(node, **kwargs)
        if isinstance(node, itir.Expr) and hasattr(node.annex, "domain"):
            new_node.annex.domain = node.annex.domain
        return new_node

    def visit_SetAt(self, node: itir.SetAt, **kwargs):
        return itir.SetAt(
            expr=self.visit(node.expr, **kwargs | {"within_set_at_expr": True}),
            domain=node.domain,
            target=node.target,
        )

    def visit_FunCall(self, node: itir.FunCall, **kwargs):
        if not kwargs.get("within_set_at_expr"):
            return node

        # inline all fields with list dtype. This needs to happen before the children are visited
        # such that the `as_fieldop` can be fused.
        # TODO(tehrengruber): what should we do in case the field with list dtype is a let itself?
        #  This could duplicate other expressions which we did not intend to duplicate.
        if cpm.is_let(node):
            for arg in node.args:
                type_inference.reinfer(arg)
            eligible_els = [
                isinstance(arg.type, ts.FieldType) and isinstance(arg.type.dtype, ts.ListType)
                for arg in node.args
            ]
            if any(eligible_els):
                node = inline_lambdas.inline_lambda(node, eligible_params=eligible_els)
                return self.visit(node, **kwargs)

        if cpm.is_applied_as_fieldop(node):  # don't descend in stencil
            node = im.as_fieldop(*node.fun.args)(*self.generic_visit(node.args, **kwargs))  # type: ignore[attr-defined]  # ensured by cpm.is_applied_as_fieldop
        elif kwargs.get("recurse", True):
            node = self.generic_visit(node, **kwargs)

        if cpm.is_call_to(node, "make_tuple"):
            for arg in node.args:
                type_inference.reinfer(arg)
                assert not isinstance(arg.type, ts.FieldType) or (
                    hasattr(arg.annex, "domain")
                    and isinstance(arg.annex.domain, domain_utils.SymbolicDomain)
                )

            eligible_els = [_make_tuple_element_inline_predicate(arg) for arg in node.args]
            field_args = [arg for i, arg in enumerate(node.args) if eligible_els[i]]
            distinct_domains = set(arg.annex.domain.as_expr() for arg in field_args)
            if len(distinct_domains) != len(field_args):
                new_els: list[itir.Expr | None] = [None for _ in node.args]
                field_args_by_domain: dict[itir.FunCall, list[tuple[int, itir.Expr]]] = {}
                for i, arg in enumerate(node.args):
                    if eligible_els[i]:
                        assert isinstance(arg.annex.domain, domain_utils.SymbolicDomain)
                        domain = arg.annex.domain.as_expr()
                        field_args_by_domain.setdefault(domain, [])
                        field_args_by_domain[domain].append((i, arg))
                    else:
                        new_els[i] = arg  # keep as is

                if len(field_args_by_domain) == 1 and all(eligible_els):
                    # if we only have a single domain covering all args we don't need to create an
                    # unnecessary let
                    ((domain, inner_field_args),) = field_args_by_domain.items()
                    new_node = im.op_as_fieldop(lambda *args: im.make_tuple(*args), domain)(
                        *(arg for _, arg in inner_field_args)
                    )
                    new_node = self.visit(new_node, **{**kwargs, "recurse": False})
                else:
                    let_vars = {}
                    for domain, inner_field_args in field_args_by_domain.items():
                        if len(inner_field_args) > 1:
                            var = self.uids.sequential_id(prefix="__fasfop")
                            fused_args = im.op_as_fieldop(
                                lambda *args: im.make_tuple(*args), domain
                            )(*(arg for _, arg in inner_field_args))
                            type_inference.reinfer(arg)
                            # don't recurse into nested args, but only consider newly created `as_fieldop`
                            # note: this will always inline (as we inline center accessed)
                            let_vars[var] = self.visit(fused_args, **{**kwargs, "recurse": False})
                            for outer_tuple_idx, (inner_tuple_idx, _) in enumerate(
                                inner_field_args
                            ):
                                new_el = im.tuple_get(outer_tuple_idx, var)
                                new_el.annex.domain = domain_utils.SymbolicDomain.from_expr(domain)
                                new_els[inner_tuple_idx] = new_el
                        else:
                            i, arg = inner_field_args[0]
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

            eligible_els = [
                _arg_inline_predicate(arg, arg_shifts)
                for arg, arg_shifts in zip(args, shifts, strict=True)
            ]
            if any(eligible_els):
                return self.visit(
                    fuse_as_fieldop(node, eligible_els, uids=self.uids),
                    **{**kwargs, "recurse": False},
                )
        return node
