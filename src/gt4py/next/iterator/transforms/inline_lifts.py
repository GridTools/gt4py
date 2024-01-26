# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
import enum
from collections.abc import Callable
from typing import Optional

import gt4py.eve as eve
from gt4py.eve import NodeTranslator, traits
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.inline_lambdas import inline_lambda
from gt4py.next.iterator.transforms.symbol_ref_utils import collect_symbol_refs
from gt4py.next.iterator.transforms.trace_shifts import TraceShifts



class ValidateRecordedShiftsAnnex(eve.NodeVisitor):
    def visit_FunCall(self, node: ir.FunCall):
        if _is_lift(node):
            if not hasattr(node.annex, "recorded_shifts"):
                breakpoint()
                assert False

            if len(node.annex.recorded_shifts) == 0:
                return

            if isinstance(node.fun.args[0], ir.Lambda):
                stencil = node.fun.args[0]
                for param in stencil.params:
                    if not hasattr(param.annex, "recorded_shifts"):
                        breakpoint()
                        assert False
                    # assert hasattr(param.annex, "recorded_shifts")

        self.generic_visit(node)


def _generate_unique_symbol(
    desired_name: Optional[str | tuple[ir.SymRef | ir.Expr, int]] = None,
    occupied_names=None,
    occupied_symbols=None,
):
    occupied_names = occupied_names or set()
    occupied_symbols = occupied_symbols or set()
    if not desired_name:
        desired_name = "__sym"
    elif isinstance(desired_name, tuple):
        fun, arg_idx = desired_name
        if isinstance(fun, ir.Lambda):
            desired_name = fun.params[arg_idx].id
        else:
            desired_name = f"__arg{arg_idx}"

    new_symbol = ir.Sym(id=desired_name)
    # make unique
    count = 0
    while new_symbol.id in occupied_names or new_symbol in occupied_symbols:
        new_symbol = ir.Sym(id=f"{desired_name}ᐞ{count}")
        count+=1
    return new_symbol


def _is_lift(node: ir.Node):
    return (
        isinstance(node, ir.FunCall)
        and isinstance(node.fun, ir.FunCall)
        and node.fun.fun == ir.SymRef(id="lift")
    )


def _is_shift_lift(node: ir.Expr):
    return (
        isinstance(node, ir.FunCall)
        and isinstance(node.fun, ir.FunCall)
        and node.fun.fun == ir.SymRef(id="shift")
        and isinstance(node.args[0], ir.FunCall)
        and isinstance(node.args[0].fun, ir.FunCall)
        and node.args[0].fun.fun == ir.SymRef(id="lift")
    )


def _is_scan(node: ir.FunCall):
    return node.fun == ir.SymRef(id="scan")


def _transform_and_extract_lift_args(
    node: ir.FunCall,
    symtable: dict[eve.SymbolName, ir.Sym],
    extracted_args: dict[ir.Sym, ir.Expr],
    recorded_shifts_base
):
    """
    Transform and extract non-symbol arguments of a lifted stencil call.

    E.g. ``lift(lambda a, b: ...)(sym1, expr1)`` is transformed into
    ``lift(lambda a, b: ...)(sym1, sym2)`` with the extracted arguments
    being ``{sym1: sym1, sym2: expr1}``.
    """
    assert _is_lift(node)
    assert isinstance(node.fun, ir.FunCall)
    inner_stencil = node.fun.args[0]

    new_args = []
    for i, (param, arg) in enumerate(zip(inner_stencil.params, node.args, strict=True)):
        if isinstance(arg, ir.SymRef):
            new_symbol = ir.Sym(id=arg.id)
            assert new_symbol not in extracted_args or extracted_args[new_symbol] == arg
            extracted_args[new_symbol] = arg
            new_args.append(arg)
        else:
            new_symbol = _generate_unique_symbol(
                desired_name=(inner_stencil, i),
                occupied_names=symtable.keys(),
                occupied_symbols=extracted_args.keys(),
            )
            assert new_symbol not in extracted_args
            extracted_args[new_symbol] = arg
            new_args.append(ir.SymRef(id=new_symbol.id))

        # todo: test this properly. not really sure about it...
        new_symbol.annex.recorded_shifts = set()
        for outer_shift in recorded_shifts_base:
            for inner_shift in param.annex.recorded_shifts:
                new_symbol.annex.recorded_shifts.add((*outer_shift, *inner_shift))

    new_lift = im.lift(inner_stencil)(*new_args)
    if hasattr(node.annex, "recorded_shifts"):
        new_lift.annex.recorded_shifts = node.annex.recorded_shifts
    new_lift.location = node.location
    return (new_lift, extracted_args)


global_counter = 0

def validate_recorded_shifts_annex(node):
    for child in node.pre_walk_values().filter(_is_lift):
        if not hasattr(child.annex, "recorded_shifts"):
            breakpoint()
            assert False
        if isinstance(child.fun.args[0], ir.Lambda):
            stencil = child.fun.args[0]
            for param in stencil.params:
                if child.annex.recorded_shifts and not hasattr(param.annex, "recorded_shifts"):
                    breakpoint()
                    assert False
                #assert hasattr(param.annex, "recorded_shifts")

# TODO(tehrengruber): This pass has many different options that should be written as dedicated
#  passes. Due to a lack of infrastructure (e.g. no pass manager) to combine passes without
#  performance degradation we leave everything as one pass for now.
@dataclasses.dataclass
class InlineLifts(
    traits.PreserveLocationVisitor, traits.VisitorWithSymbolTableTrait, NodeTranslator
):
    """Inline lifted function calls.

    Optionally a predicate function can be passed which can enable or disable inlining of specific
    function nodes.
    """

    PRESERVED_ANNEX_ATTRS = ("recorded_shifts",)

    class Flag(enum.IntEnum):
        #: `shift(...)(lift(f)(args...))` -> `lift(f)(shift(...)(args)...)`
        PROPAGATE_SHIFT = 1
        #: `deref(lift(f)())` -> `f()`
        INLINE_TRIVIAL_DEREF_LIFT = 2
        #: `deref(lift(f)(args...))` -> `f(args...)`
        INLINE_DEREF_LIFT = 2 + 4
        #: `can_deref(lift(f)(args...))` -> `and(can_deref(arg[0]), and(can_deref(arg[1]), ...))`
        PROPAGATE_CAN_DEREF = 8
        #: Inline arguments to lifted stencil calls, e.g.:
        #:  lift(λ(a) → inner_ex(a))(lift(λ(b) → outer_ex(b))(arg))
        #: is transformed into:
        #:  lift(λ(b) → inner_ex(outer_ex(b)))(arg)
        #: Note: This option is only needed when there is no outer `deref` by which the previous
        #: branches eliminate the lift calls. This occurs for example for the `reduce` builtin
        #: or when a better readable expression of a lift statement is needed during debugging.
        #: Due to its complexity we might want to remove this option at some point again,
        #: when we see that it is not required.
        INLINE_LIFTED_ARGS = 16
        INLINE_CENTRE_ONLY_LIFT_ARGS = 32
        REMOVE_UNUSED_LIFT_ARGS = 64

    predicate: Callable[[ir.Expr, bool], bool] = lambda _1, _2: True

    flags: int = (
        Flag.PROPAGATE_SHIFT
        | Flag.INLINE_DEREF_LIFT
        | Flag.PROPAGATE_CAN_DEREF
        | Flag.INLINE_LIFTED_ARGS
        | Flag.INLINE_CENTRE_ONLY_LIFT_ARGS
        | Flag.REMOVE_UNUSED_LIFT_ARGS
    )

    def visit_StencilClosure(self, node: ir.StencilClosure, **kwargs):
        if self.flags & self.Flag.INLINE_CENTRE_ONLY_LIFT_ARGS:
            TraceShifts.apply(node, inputs_only=False, save_to_annex=True)
            ValidateRecordedShiftsAnnex().visit(node)
        return self.generic_visit(node, **kwargs)


    def visit_FunCall(
        self, node: ir.FunCall, *, is_scan_pass_context=False, recurse=True, **kwargs
    ):
        symtable = kwargs["symtable"]

        ignore_recorded_shifts_missing = (kwargs.get("ignore_recorded_shifts_missing", False) or (
                    hasattr(node.annex, "recorded_shifts") and len(
                node.annex.recorded_shifts) == 0))
        kwargs = {**kwargs, "ignore_recorded_shifts_missing": ignore_recorded_shifts_missing}

        recorded_shifts_annex = getattr(node.annex, "recorded_shifts", None)
        old_node = node
        node = (
            ir.FunCall(
                fun=self.generic_visit(node.fun, is_scan_pass_context=_is_scan(node), **kwargs),
                args=self.generic_visit(node.args, **kwargs),
            )
            if recurse
            else node
        )
        if recorded_shifts_annex is not None:
            node.annex.recorded_shifts = recorded_shifts_annex

        #ValidateRecordedShiftsAnnex().visit(node)


        if self.flags & self.Flag.PROPAGATE_SHIFT and _is_shift_lift(node):
            shift = node.fun
            assert len(node.args) == 1
            lift_call = node.args[0]
            new_args = [
                self.visit(ir.FunCall(fun=shift, args=[arg]), recurse=False, **kwargs)
                for arg in lift_call.args  # type: ignore[attr-defined] # lift_call already asserted to be of type ir.FunCall
            ]
            result = ir.FunCall(fun=lift_call.fun, args=new_args)  # type: ignore[attr-defined] # lift_call already asserted to be of type ir.FunCall
            result.annex.recorded_shifts = node.annex.recorded_shifts
            return self.visit(result, recurse=False, **kwargs)
        if self.flags & (self.Flag.INLINE_DEREF_LIFT | self.Flag.INLINE_TRIVIAL_DEREF_LIFT) and node.fun == ir.SymRef(id="deref"):
            # TODO: this does not work for neighbors(..., lift(...)), which is essentially also a deref
            assert len(node.args) == 1
            is_lift = _is_lift(node.args[0])
            is_eligible = is_lift and self.predicate(node.args[0], is_scan_pass_context)
            is_trivial = is_lift and len(node.args[0].args) == 0  # type: ignore[attr-defined] # mypy not smart enough
            if (
                self.flags & self.Flag.INLINE_DEREF_LIFT
                or (self.flags & self.Flag.INLINE_TRIVIAL_DEREF_LIFT and is_trivial)
            ) and is_eligible:
                assert isinstance(node.args[0], ir.FunCall) and isinstance(
                    node.args[0].fun, ir.FunCall
                )
                assert len(node.args[0].fun.args) == 1
                f = node.args[0].fun.args[0]
                args = node.args[0].args
                new_node = ir.FunCall(fun=f, args=args)
                if isinstance(f, ir.Lambda):
                    new_node = inline_lambda(new_node, opcount_preserving=True)
                return self.visit(new_node, **kwargs)
        if self.flags & self.Flag.PROPAGATE_CAN_DEREF and node.fun == ir.SymRef(id="can_deref"):
            # TODO(havogt): this `can_deref` transformation doesn't look into lifted functions,
            #  this need to be changed to be 100% compliant
            assert len(node.args) == 1
            if _is_lift(node.args[0]) and self.predicate(node.args[0], is_scan_pass_context):
                assert isinstance(node.args[0], ir.FunCall) and isinstance(
                    node.args[0].fun, ir.FunCall
                )
                assert len(node.args[0].fun.args) == 1
                args = node.args[0].args
                if len(args) == 0:
                    return ir.Literal(value="True", type="bool")

                res = ir.FunCall(fun=ir.SymRef(id="can_deref"), args=[args[0]])
                for arg in args[1:]:
                    res = ir.FunCall(
                        fun=ir.SymRef(id="and_"),
                        args=[res, ir.FunCall(fun=ir.SymRef(id="can_deref"), args=[arg])],
                    )
                return res
        # if (
        #     self.flags & self.Flag.REMOVE_UNUSED_LIFT_ARGS
        #     and _is_lift(node) and isinstance(node.fun.args[0], ir.Lambda)
        # ):
        #     # we use recorded shifts in the condition as it is computed once anyway
        #     if any([len(arg.annex.recorded_shifts) == 0 for arg in node.args]):
        #         # even if an argument is never derefed it might occur in the stencil expr
        #         used_args |= set(collect_symbol_refs(node.fun.args[0],
        #                          symbol_names=[arg.id for arg in node.args]))
        #         used_args_mask = [param.id in used_args for param in node.fun.args[0].params]
        #         im.lift(
        #             im.lambda_(*[param for i, arg in enumerate(node.fun.args[0].params) if used_args_mask[i]])
        #         )(
        #             arg for i, arg in enumerate(node.args) if used_args_mask[i]
        #         )
        if (
            self.flags & (self.Flag.INLINE_LIFTED_ARGS | self.Flag.INLINE_CENTRE_ONLY_LIFT_ARGS)
            and _is_lift(node)
            and len(node.args) > 0
            and self.predicate(node, is_scan_pass_context)
        ):
            if not ignore_recorded_shifts_missing and not hasattr(node.annex, "recorded_shifts"):
                breakpoint()

            # if the lift is never derefed its params also don't have a recorded_shifts attr and the
            #  following will fail. we don't care about such lifts anyway as they are later on and
            #  disappear
            if ignore_recorded_shifts_missing or len(node.annex.recorded_shifts) == 0:
                return node

            stencil = node.fun.args[0]  # type: ignore[attr-defined] # node already asserted to be of type ir.FunCall
            eligible_lifted_args = [False] * len(node.args)

            if not isinstance(stencil, ir.Lambda):
                return node

            if self.flags & self.Flag.INLINE_LIFTED_ARGS:
                for i, arg in enumerate(node.args):
                    # TODO: if not isinstance(arg.fun.args[0], ir.Lambda) we have a scan. document this
                    eligible_lifted_args[i] |= _is_lift(arg) and isinstance(arg.fun.args[0], ir.Lambda) and self.predicate(arg, is_scan_pass_context)
            if self.flags & self.Flag.INLINE_CENTRE_ONLY_LIFT_ARGS:
                # TODO: write comment why we need params, i.e. local shift, instead of arg to get recorded shifts
                for i, (param, arg) in enumerate(zip(stencil.params, node.args, strict=True)):
                    if _is_lift(arg) and not hasattr(param.annex, "recorded_shifts"):
                        self.visit(old_node.args[1], **kwargs)
                        breakpoint()
                    # TODO: for lift it should be fine as long as shift only occurs once (not only centre)
                    # TODO: if not isinstance(arg.fun.args[0], ir.Lambda) we have a scan. document this
                    eligible_lifted_args[i] |= _is_lift(arg) and isinstance(arg.fun.args[0], ir.Lambda) and (param.annex.recorded_shifts in [set(), {()}])


            if isinstance(stencil, ir.Lambda) and any(eligible_lifted_args):
                # TODO(tehrengruber): we currently only inlining opcount preserving, but what we
                #  actually want is to inline whenever the argument is not shifted. This is
                #  currently beyond the capabilities of the inliner and the shift tracer.
                new_arg_exprs: dict[ir.Sym, ir.Expr] = {}
                bound_scalars: dict = {}
                inlined_args = []
                for i, (param, arg, eligible) in enumerate(zip(stencil.params, node.args, eligible_lifted_args)):
                    if eligible:
                        assert isinstance(arg, ir.FunCall)
                        transformed_arg, _ = _transform_and_extract_lift_args(
                            arg, symtable, new_arg_exprs, param.annex.recorded_shifts
                        )

                        if param.annex.recorded_shifts in [set(), {()}]:
                            global global_counter
                            new_name = f"bound_scalar_{global_counter}"
                            global_counter+=1
                            bound_scalars[new_name] = InlineLifts(flags=self.Flag.INLINE_TRIVIAL_DEREF_LIFT).visit(im.deref(transformed_arg), recurse=False)
                            new_lift = im.lift(im.lambda_()(new_name))()
                            new_lift.annex.recorded_shifts = arg.annex.recorded_shifts  # todo: isn't this empty all the time
                            inlined_args.append(new_lift)
                        else:
                            inlined_args.append(transformed_arg)
                    else:
                        if isinstance(arg, ir.SymRef):
                            new_arg_sym = ir.Sym(id=arg.id)
                        else:
                            new_arg_sym = _generate_unique_symbol(
                                desired_name=(stencil, i),
                                occupied_names=symtable.keys(),
                                occupied_symbols=new_arg_exprs.keys(),
                            )

                        new_arg_sym.annex.recorded_shifts = param.annex.recorded_shifts

                        if new_arg_sym in new_arg_exprs:
                            for key in new_arg_exprs.keys():
                                if key == new_arg_sym:
                                    new_arg_sym.annex.recorded_shifts.update(key.annex.recorded_shifts)

                        assert new_arg_sym not in new_arg_exprs or new_arg_exprs[new_arg_sym] == arg
                        new_arg_exprs[new_arg_sym] = arg
                        inlined_args.append(ir.SymRef(id=new_arg_sym.id))

                inlined_call = self.visit(
                    inline_lambda(
                        ir.FunCall(fun=stencil, args=inlined_args), opcount_preserving=True
                    ),
                    **kwargs,
                )

                if bound_scalars:
                    # TODO(tehrengruber): propagate let outwards
                    inlined_call = im.let(*bound_scalars.items())(inlined_call)
                else:
                    inlined_call = inlined_call

                for new_arg_sym in new_arg_exprs.keys():
                    if not hasattr(new_arg_sym.annex, "recorded_shifts"):
                        breakpoint()
                    assert hasattr(new_arg_sym.annex, "recorded_shifts")

                new_stencil = im.lambda_(*new_arg_exprs.keys())(inlined_call)
                new_applied_lift = im.lift(new_stencil)(*new_arg_exprs.values())
                if hasattr(node.annex, "recorded_shifts"):
                    new_applied_lift.annex.recorded_shifts = node.annex.recorded_shifts
                return new_applied_lift

        return node
