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
    ...         nested_as_fieldop, offset_provider={}, allow_undeclared_symbols=True
    ...     )
    ... )
    as_fieldop(λ(inp1, inp2, inp3) → ·inp1 × ·inp2 + ·inp3, c⟨ IDimₕ: [0, 1) ⟩)(inp1, inp2, inp3)
    """  # noqa: RUF002  # ignore ambiguous multiplication character

    uids: eve_utils.UIDGenerator

    def _inline_as_fieldop_arg(self, arg: itir.Expr) -> tuple[itir.Expr, dict[str, itir.Expr]]:
        assert cpm.is_applied_as_fieldop(arg)
        arg = _canonicalize_as_fieldop(arg)

        stencil, *_ = arg.fun.args  # type: ignore[attr-defined]  # ensured by `is_applied_as_fieldop`
        inner_args: list[itir.Expr] = arg.args
        extracted_args: dict[str, itir.Expr] = {}  # mapping from outer-stencil param to arg

        stencil_params: list[itir.Sym] = []
        stencil_body: itir.Expr = stencil.expr

        for inner_param, inner_arg in zip(stencil.params, inner_args, strict=True):
            if isinstance(inner_arg, itir.SymRef):
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
                new_outer_stencil_param = self.uids.sequential_id(prefix="__iasfop")
                extracted_args[new_outer_stencil_param] = inner_arg

        return im.lift(im.lambda_(*stencil_params)(stencil_body))(
            *extracted_args.keys()
        ), extracted_args

    @classmethod
    def apply(
        cls,
        node: itir.Program,
        *,
        offset_provider,
        uids: Optional[eve_utils.UIDGenerator] = None,
        allow_undeclared_symbols=False,
    ):
        node = type_inference.infer(
            node, offset_provider=offset_provider, allow_undeclared_symbols=allow_undeclared_symbols
        )

        if not uids:
            uids = eve_utils.UIDGenerator()

        return cls(uids=uids).visit(node)

    def visit_FunCall(self, node: itir.FunCall):
        node = self.generic_visit(node)

        if cpm.is_call_to(node.fun, "as_fieldop"):
            node = _canonicalize_as_fieldop(node)

        if cpm.is_call_to(node.fun, "as_fieldop") and isinstance(node.fun.args[0], itir.Lambda):
            stencil: itir.Lambda = node.fun.args[0]
            domain = node.fun.args[1] if len(node.fun.args) > 1 else None

            shifts = trace_shifts.trace_stencil(stencil)

            args: list[itir.Expr] = node.args

            new_args: dict[str, itir.Expr] = {}
            new_stencil_body: itir.Expr = stencil.expr

            for stencil_param, arg, arg_shifts in zip(stencil.params, args, shifts, strict=True):
                assert isinstance(arg.type, ts.TypeSpec)
                dtype = type_info.apply_to_primitive_constituents(type_info.extract_dtype, arg.type)
                # TODO(tehrengruber): make this configurable
                should_inline = _is_tuple_expr_of_literals(arg) or (
                    isinstance(arg, itir.FunCall)
                    and (
                        cpm.is_call_to(arg.fun, "as_fieldop")
                        and isinstance(arg.fun.args[0], itir.Lambda)
                        or cpm.is_call_to(arg, "if_")
                    )
                    and (isinstance(dtype, it_ts.ListType) or len(arg_shifts) <= 1)
                )
                if should_inline:
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

                    inline_expr, extracted_args = self._inline_as_fieldop_arg(arg)

                    new_stencil_body = im.let(stencil_param, inline_expr)(new_stencil_body)

                    new_args = _merge_arguments(new_args, extracted_args)
                else:
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

            new_node = im.as_fieldop(im.lambda_(*new_args.keys())(new_stencil_body), domain)(
                *new_args.values()
            )

            # simplify stencil directly to keep the tree small
            new_node = inline_center_deref_lift_vars.InlineCenterDerefLiftVars.apply(
                new_node
            )  # to keep the tree small
            new_node = inline_lambdas.InlineLambdas.apply(
                new_node, opcount_preserving=True, force_inline_lift_args=True
            )
            new_node = inline_lifts.InlineLifts().visit(new_node)

            type_inference.copy_type(from_=node, to=new_node)

            return new_node
        return node
