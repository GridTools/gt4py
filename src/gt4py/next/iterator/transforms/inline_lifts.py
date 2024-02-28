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
import functools
import operator
from collections.abc import Callable
from typing import Optional

import gt4py.eve as eve
from gt4py.eve import NodeTranslator, traits, utils as eve_utils
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.inline_lambdas import inline_lambda
from gt4py.next.iterator.transforms.trace_shifts import TraceShifts, copy_recorded_shifts


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
        count += 1
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
    recorded_shifts_base=None,
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
    for i, arg in enumerate(node.args):
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
        if recorded_shifts_base is not None:
            if isinstance(inner_stencil, ir.Lambda):
                recorded_shifts = inner_stencil.params[i].annex.recorded_shifts
            elif inner_stencil == im.ref("deref"):
                recorded_shifts = ()
            else:
                raise AssertionError("Expected a Lambda function or deref as stencil.")
            new_symbol.annex.recorded_shifts = set()
            for outer_shift in recorded_shifts_base:
                for inner_shift in recorded_shifts:
                    new_symbol.annex.recorded_shifts.add((*outer_shift, *inner_shift))

    new_lift = im.lift(inner_stencil)(*new_args)
    if hasattr(node.annex, "recorded_shifts"):
        copy_recorded_shifts(from_=node, to=new_lift)
    new_lift.location = node.location
    return (new_lift, extracted_args)


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

    uids: Optional[eve_utils.UIDGenerator] = (
        None  # optional since not all transformations need this.
    )

    class Flag(enum.Flag):
        #: `shift(...)(lift(f)(args...))` -> `lift(f)(shift(...)(args)...)`
        PROPAGATE_SHIFT = enum.auto()
        #: `deref(lift(f)())` -> `f()`
        INLINE_TRIVIAL_DEREF_LIFT = enum.auto()
        #: `deref(lift(f)(args...))` -> `f(args...)`
        INLINE_DEREF_LIFT = enum.auto()
        #: `can_deref(lift(f)(args...))` -> `and(can_deref(arg[0]), and(can_deref(arg[1]), ...))`
        PROPAGATE_CAN_DEREF = enum.auto()
        #: Inline arguments to lifted stencil calls, e.g.:
        #:  lift(λ(a) → inner_ex(a))(lift(λ(b) → outer_ex(b))(arg))
        #: is transformed into:
        #:  lift(λ(b) → inner_ex(outer_ex(b)))(arg)
        #: Note: This option is only needed when there is no outer `deref` by which the previous
        #: branches eliminate the lift calls. This occurs for example for the `reduce` builtin
        #: or when a better readable expression of a lift statement is needed during debugging.
        #: Due to its complexity we might want to remove this option at some point again,
        #: when we see that it is not required.
        INLINE_LIFTED_ARGS = enum.auto()

        @classmethod
        def all(self):  # noqa: A003  # shadowing a python builtin
            return functools.reduce(operator.or_, self.__members__.values())

    predicate: Callable[[ir.Expr, bool], bool] = lambda _1, _2: True

    inline_centre_lift_args_only: bool = False

    flags: Flag = Flag.all()

    @classmethod
    def apply(
        cls,
        node: ir.Node,
        *,
        flags: Flag = flags,
        uids: Optional[eve_utils.UIDGenerator] = uids,
        predicate: Callable[[ir.Expr, bool], bool] = predicate,
        recurse: bool = True,
        inline_center_lift_args_only: bool = inline_centre_lift_args_only,
    ):
        if inline_center_lift_args_only:
            assert isinstance(node, ir.FencilDefinition)
            TraceShifts.apply(node, inputs_only=False, save_to_annex=True)

        return cls(
            uids=uids,
            flags=flags,
            predicate=predicate,
            inline_centre_lift_args_only=inline_center_lift_args_only,
        ).visit(node, recurse=recurse, is_scan_pass_context=False)

    def transform_propagate_shift(self, node: ir.FunCall, *, recurse, **kwargs):
        if not _is_shift_lift(node):
            return None
        shift = node.fun
        assert len(node.args) == 1
        lift_call = node.args[0]
        new_args = [
            self.visit(ir.FunCall(fun=shift, args=[arg]), recurse=False, **kwargs)
            for arg in lift_call.args  # type: ignore[attr-defined] # lift_call already asserted to be of type ir.FunCall
        ]
        result = ir.FunCall(fun=lift_call.fun, args=new_args)  # type: ignore[attr-defined] # lift_call already asserted to be of type ir.FunCall
        # TODO: describe everywhere
        copy_recorded_shifts(from_=node, to=result, required=False)
        return self.visit(result, recurse=False, **kwargs)

    def transform_inline_deref_lift(self, node: ir.FunCall, is_eligible=None, **kwargs):
        # TODO: this does not work for neighbors(..., lift(...)), which is essentially also a deref
        is_scan_pass_context = kwargs["is_scan_pass_context"]
        if not node.fun == ir.SymRef(id="deref"):
            return None
        assert len(node.args) == 1
        is_lift = _is_lift(node.args[0])
        is_eligible = is_eligible or (
            is_lift and self.predicate(node.args[0], is_scan_pass_context)
        )

        if is_eligible:
            assert isinstance(node.args[0], ir.FunCall) and isinstance(node.args[0].fun, ir.FunCall)
            assert len(node.args[0].fun.args) == 1
            f = node.args[0].fun.args[0]
            args = node.args[0].args
            new_node = ir.FunCall(fun=f, args=args)
            if isinstance(f, ir.Lambda):
                new_node = inline_lambda(new_node, opcount_preserving=True)
            return self.visit(new_node, **kwargs)
        return None

    def transform_inline_trivial_deref_lift(self, node: ir.FunCall, **kwargs):
        if not node.fun == ir.SymRef(id="deref"):
            return None
        is_trivial = _is_lift(node.args[0]) and len(node.args[0].args) == 0  # type: ignore[attr-defined] # mypy not smart enough
        return self.transform_inline_deref_lift(node, is_trivial, **kwargs)

    def transform_propagate_can_deref(self, node: ir.FunCall, **kwargs):
        if not node.fun == ir.SymRef(id="can_deref"):
            return None
        # TODO(havogt): this `can_deref` transformation doesn't look into lifted functions,
        #  this need to be changed to be 100% compliant
        is_scan_pass_context = kwargs.get("is_scan_pass_context", False)
        assert len(node.args) == 1
        if _is_lift(node.args[0]) and self.predicate(node.args[0], is_scan_pass_context):
            assert isinstance(node.args[0], ir.FunCall) and isinstance(node.args[0].fun, ir.FunCall)
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
        return None

    def transform_inline_lifted_args(self, node: ir.FunCall, **kwargs):  # noqa: C901
        symtable, is_scan_pass_context = kwargs["symtable"], kwargs.get(
            "is_scan_pass_context", False
        )
        if not (
            _is_lift(node) and len(node.args) > 0 and self.predicate(node, is_scan_pass_context)
        ):
            return None

        stencil = node.fun.args[0]  # type: ignore[attr-defined] # node already asserted to be of type ir.FunCall

        if not isinstance(stencil, ir.Lambda):
            return None

        eligible_lifted_args = [False] * len(node.args)
        for i, (param, arg) in enumerate(zip(stencil.params, node.args, strict=True)):
            # TODO: if not isinstance(arg.fun.args[0], ir.Lambda) we have a scan. document this
            if not _is_lift(arg) or not isinstance(arg.fun.args[0], ir.Lambda):  # type: ignore[attr-defined] # ensure by _is_lift
                continue
            if not self.predicate(arg, is_scan_pass_context):
                continue
            if self.inline_centre_lift_args_only:
                if param.annex.recorded_shifts not in [set(), {()}]:
                    continue
            eligible_lifted_args[i] = True

        if isinstance(stencil, ir.Lambda) and any(eligible_lifted_args):
            new_arg_exprs: dict[ir.Sym, ir.Expr] = {}
            bound_scalars: dict = {}
            inlined_args = []
            for i, (param, arg, eligible) in enumerate(
                zip(stencil.params, node.args, eligible_lifted_args)
            ):
                if eligible:
                    assert isinstance(arg, ir.FunCall)
                    transformed_arg, _ = _transform_and_extract_lift_args(
                        arg, symtable, new_arg_exprs, getattr(param.annex, "recorded_shifts", None)
                    )

                    if self.inline_centre_lift_args_only and param.annex.recorded_shifts in [
                        set(),
                        {()},
                    ]:
                        assert self.uids
                        bound_arg_name = self.uids.sequential_id(prefix="_icdla")
                        bound_scalars[bound_arg_name] = InlineLifts.apply(
                            im.deref(transformed_arg),
                            flags=self.Flag.INLINE_TRIVIAL_DEREF_LIFT,
                            recurse=False,
                        )
                        capture_lift = im.promote_to_const_iterator(bound_arg_name)
                        assert arg.annex.recorded_shifts in [set(), {()}]  # just a sanity check
                        copy_recorded_shifts(from_=arg, to=capture_lift)
                        inlined_args.append(capture_lift)
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

                    copy_recorded_shifts(
                        from_=param, to=new_arg_sym, required=self.inline_centre_lift_args_only
                    )

                    if new_arg_sym in new_arg_exprs:
                        for key in new_arg_exprs.keys():
                            if key == new_arg_sym:
                                new_arg_sym.annex.recorded_shifts.update(key.annex.recorded_shifts)

                    assert new_arg_sym not in new_arg_exprs or new_arg_exprs[new_arg_sym] == arg
                    new_arg_exprs[new_arg_sym] = arg
                    inlined_args.append(ir.SymRef(id=new_arg_sym.id))

            inlined_call = self.visit(
                inline_lambda(ir.FunCall(fun=stencil, args=inlined_args), opcount_preserving=True),
                **kwargs,
            )

            if bound_scalars:
                # TODO(tehrengruber): propagate let outwards
                inlined_call = im.let(*bound_scalars.items())(inlined_call)
            else:
                inlined_call = inlined_call

            if self.inline_centre_lift_args_only:
                assert all(
                    hasattr(new_arg_sym.annex, "recorded_shifts")
                    for new_arg_sym in new_arg_exprs.keys()
                )

            new_stencil = im.lambda_(*new_arg_exprs.keys())(inlined_call)
            new_applied_lift = im.lift(new_stencil)(*new_arg_exprs.values())
            copy_recorded_shifts(from_=node, to=new_applied_lift, required=False)
            return new_applied_lift

    def visit_FunCall(self, node: ir.FunCall, **kwargs):
        recurse = kwargs["recurse"]

        if recurse:
            fun_kwargs = {k: v for k, v in kwargs.items() if k != "is_scan_pass_context"}
            new_node = ir.FunCall(
                fun=self.generic_visit(node.fun, is_scan_pass_context=_is_scan(node), **fun_kwargs),
                args=self.generic_visit(node.args, **kwargs),
            )
            copy_recorded_shifts(from_=node, to=new_node, required=False)
        else:
            new_node = node

        for transformation in self.Flag:
            if self.flags & transformation:
                assert isinstance(transformation.name, str)
                method = getattr(self, f"transform_{transformation.name.lower()}")
                transformed_node = method(new_node, **kwargs)
                if transformed_node is not None:
                    return transformed_node

        return new_node
