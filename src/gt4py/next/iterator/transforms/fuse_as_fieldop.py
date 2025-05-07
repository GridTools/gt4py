# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import collections
import dataclasses
import enum
import functools
import operator
from typing import Optional

from gt4py import eve
from gt4py.eve import utils as eve_utils
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
    misc as ir_misc,
)
from gt4py.next.iterator.transforms import (
    fixed_point_transformation,
    inline_center_deref_lift_vars,
    inline_lambdas,
    inline_lifts,
    merge_let,
    trace_shifts,
)
from gt4py.next.iterator.type_system import inference as type_inference
from gt4py.next.type_system import type_info, type_specifications as ts


def _merge_arguments(
    args1: dict[str, itir.Expr], args2: dict[str, itir.Expr]
) -> dict[str, itir.Expr]:
    new_args = {**args1}
    for stencil_param, stencil_arg in args2.items():
        assert stencil_param not in new_args
        new_args[stencil_param] = stencil_arg
    return new_args


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
    arg = ir_misc.canonicalize_as_fieldop(arg)

    stencil, *_ = arg.fun.args  # type: ignore[attr-defined]  # ensured by `is_applied_as_fieldop`
    inner_args: list[itir.Expr] = arg.args
    extracted_args: dict[str, itir.Expr] = {}  # mapping from outer-stencil param to arg

    stencil_params: list[itir.Sym] = []
    stencil_body: itir.Expr = stencil.expr

    for inner_param, inner_arg in zip(stencil.params, inner_args, strict=True):
        if isinstance(inner_arg, itir.Literal):
            # note: only literals, not all scalar expressions are required as it doesn't make sense
            # for them to be computed per grid point.
            stencil_body = im.let(inner_param, im.promote_to_const_iterator(inner_arg))(
                stencil_body
            )
        else:
            stencil_params.append(inner_param)
            new_outer_stencil_param = uids.sequential_id(prefix="__iasfop")
            extracted_args[new_outer_stencil_param] = inner_arg

    return im.lift(im.lambda_(*stencil_params)(stencil_body))(
        *extracted_args.keys()
    ), extracted_args


def _deduplicate_as_fieldop_args(
    args: dict[str, itir.Expr], stencil_body: itir.Expr
) -> tuple[dict[str, itir.Expr], itir.Expr]:
    new_args_inverted: dict[itir.Expr, list[str]] = collections.defaultdict(list)
    for name, arg in args.items():
        new_args_inverted[arg].append(name)

    new_args: dict[str, itir.Expr] = {}
    new_stencil_body = stencil_body
    for arg, names in new_args_inverted.items():
        # put internal names at the end
        new_name, *aliases = sorted(names, key=lambda s: (s.startswith("_"), s))
        new_args[new_name] = arg
        if aliases:
            new_stencil_body = im.let(*((alias, new_name) for alias in aliases))(new_stencil_body)

    return new_args, new_stencil_body


def _prettify_as_fieldop_args(
    args: dict[str, itir.Expr], stencil_body: itir.Expr
) -> tuple[dict[str, itir.Expr], itir.Expr]:
    new_args: dict[str, itir.Expr] = {}
    remap_table: dict[str, str] = {}
    for name, arg in args.items():
        if isinstance(arg, itir.SymRef):
            assert arg.id not in new_args  # ensured by deduplication
            new_args[arg.id] = arg
            remap_table[name] = arg.id
        else:
            new_args[name] = arg

    return new_args, im.let(*remap_table.items())(stencil_body)


def fuse_as_fieldop(
    expr: itir.Expr, eligible_args: list[bool], *, uids: eve_utils.UIDGenerator
) -> itir.Expr:
    assert cpm.is_applied_as_fieldop(expr)

    assert isinstance(expr.fun.args[0], itir.Lambda) or cpm.is_call_to(expr.fun.args[0], "scan")
    stencil, restore_scan = ir_misc.unwrap_scan(expr.fun.args[0])

    domain = expr.fun.args[1] if len(expr.fun.args) > 1 else None

    args: list[itir.Expr] = expr.args

    new_args: dict[str, itir.Expr] = {}
    new_stencil_body: itir.Expr = stencil.expr

    for eligible, stencil_param, arg in zip(eligible_args, stencil.params, args, strict=True):
        if eligible:
            if cpm.is_applied_as_fieldop(arg):
                pass
            elif cpm.is_call_to(arg, "if_"):
                # transform scalar `if` into per-grid-point `if`
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
            new_args = _merge_arguments(new_args, {stencil_param.id: arg})

    new_args, new_stencil_body = _deduplicate_as_fieldop_args(new_args, new_stencil_body)
    new_args, new_stencil_body = _prettify_as_fieldop_args(new_args, new_stencil_body)

    stencil = im.lambda_(*new_args.keys())(new_stencil_body)
    new_stencil = restore_scan(stencil)

    # simplify stencil directly to keep the tree small
    new_stencil = inline_lambdas.InlineLambdas.apply(
        new_stencil, opcount_preserving=True, force_inline_lift_args=False
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


def _arg_inline_predicate(node: itir.Expr, shifts: set[tuple[itir.OffsetLiteral, ...]]) -> bool:
    if _is_tuple_expr_of_literals(node):
        return True

    if (
        is_applied_fieldop := cpm.is_applied_as_fieldop(node)
        and not cpm.is_call_to(node.fun.args[0], "scan")
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class FuseAsFieldOp(
    fixed_point_transformation.FixedPointTransformation, eve.PreserveLocationVisitor
):
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

    class Transformation(enum.Flag):
        #: Let `f_expr` be an expression with list dtype then
        #: `let(f, f_expr) -> as_fieldop(...)(f)` -> `as_fieldop(...)(f_expr)`
        FUSE_MAKE_TUPLE = enum.auto()
        #: `as_fieldop(...)(as_fieldop(...)(a, b), c)`
        #: -> as_fieldop(fused_stencil)(a, b, c)
        FUSE_AS_FIELDOP = enum.auto()
        INLINE_LET_VARS_OPCOUNT_PRESERVING = enum.auto()

        @classmethod
        def all(self) -> FuseAsFieldOp.Transformation:
            return functools.reduce(operator.or_, self.__members__.values())

    PRESERVED_ANNEX_ATTRS = ("domain",)
    REINFER_TYPES = True

    enabled_transformations = Transformation.all()

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
        enabled_transformations: Optional[Transformation] = None,
    ):
        enabled_transformations = enabled_transformations or cls.enabled_transformations

        node = type_inference.infer(
            node,
            offset_provider_type=offset_provider_type,
            allow_undeclared_symbols=allow_undeclared_symbols,
        )

        if within_set_at_expr is None:
            within_set_at_expr = not isinstance(node, itir.Program)

        if not uids:
            uids = eve_utils.UIDGenerator()

        new_node = cls(uids=uids, enabled_transformations=enabled_transformations).visit(
            node, within_set_at_expr=within_set_at_expr
        )
        # The `FuseAsFieldOp` pass does not fully preserve the type information yet. In particular
        # for the generated lifts this is tricky and error-prone. For simplicity, we just reinfer
        # everything here ensuring later passes can use the information.
        new_node = type_inference.infer(
            new_node,
            offset_provider_type=offset_provider_type,
            allow_undeclared_symbols=allow_undeclared_symbols,
        )
        return new_node

    def transform_fuse_make_tuple(self, node: itir.Node, **kwargs):
        if not cpm.is_call_to(node, "make_tuple"):
            return None

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
                        fused_args = im.op_as_fieldop(lambda *args: im.make_tuple(*args), domain)(
                            *(arg for _, arg in inner_field_args)
                        )
                        type_inference.reinfer(arg)
                        # don't recurse into nested args, but only consider newly created `as_fieldop`
                        # note: this will always inline (as we inline center accessed)
                        let_vars[var] = self.visit(fused_args, **{**kwargs, "recurse": False})
                        for outer_tuple_idx, (inner_tuple_idx, _) in enumerate(inner_field_args):
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
        return None

    def transform_fuse_as_fieldop(self, node: itir.Node, **kwargs):
        if cpm.is_applied_as_fieldop(node):
            node = ir_misc.canonicalize_as_fieldop(node)
            stencil = node.fun.args[0]  # type: ignore[attr-defined]  # ensure cpm.is_applied_as_fieldop
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
        return None

    def transform_inline_let_vars_opcount_preserving(self, node: itir.Node, **kwargs):
        # when multiple `as_fieldop` calls are fused that use the same argument, this argument
        # might become referenced once only. In order to be able to continue fusing such arguments
        # try inlining here.
        if cpm.is_let(node):
            new_node = inline_lambdas.inline_lambda(node, opcount_preserving=True)
            if new_node is not node:  # nothing has been inlined
                return self.visit(new_node, **kwargs)

        return None

    def generic_visit(self, node, **kwargs):
        if cpm.is_applied_as_fieldop(node):  # don't descend in stencil
            return im.as_fieldop(*node.fun.args)(*self.visit(node.args, **kwargs))

        # TODO(tehrengruber): This is a common pattern that should be absorbed in
        #  `FixedPointTransformation`.
        if kwargs.get("recurse", True):
            return super().generic_visit(node, **kwargs)
        else:
            return node

    def visit(self, node, **kwargs):
        if isinstance(node, itir.SetAt):
            return itir.SetAt(
                expr=self.visit(node.expr, **kwargs | {"within_set_at_expr": True}),
                # rest doesn't need to be visited
                domain=node.domain,
                target=node.target,
            )

        # don't execute transformations unless inside `SetAt` node
        if not kwargs.get("within_set_at_expr"):
            return self.generic_visit(node, **kwargs)

        # inline all fields with list dtype. This needs to happen before the children are visited
        # such that the `as_fieldop` can be fused.
        # TODO(tehrengruber): what should we do in case the field with list dtype is a let itself?
        #  This could duplicate other expressions which we did not intend to duplicate.
        # TODO(tehrengruber): This should be moved into a `transform_` method, but
        #  `FixedPointTransformation` does not support pre-order transformations yet.
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

        node = super().visit(node, **kwargs)

        return node
