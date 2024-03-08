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
from typing import Callable, Optional, TypeGuard

import gt4py.eve as eve
from gt4py.eve import NodeTranslator, traits
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import trace_shifts
from gt4py.next.iterator.transforms.inline_lambdas import inline_lambda


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


def _is_lift(node: ir.Node) -> TypeGuard[ir.FunCall]:
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


def _is_scan(node: ir.Expr) -> TypeGuard[ir.FunCall]:
    return isinstance(node, ir.FunCall) and node.fun == ir.SymRef(id="scan")


def _transform_and_extract_lift_args(
    node: ir.FunCall,
    reserved_symbol_names: list[eve.SymbolName],
    extracted_args: dict[ir.Sym, ir.Expr],
    recorded_shifts_base: Optional[set[tuple[ir.OffsetLiteral, ...]]] = None,
) -> tuple[ir.FunCall, dict[ir.Sym, ir.Expr]]:
    """
    Transform and extract non-symbol arguments of a lifted stencil call.

    E.g. ``lift(lambda a, b: ...)(sym1, expr1)`` is transformed into
    ``lift(lambda a, b: ...)(sym1, sym2)`` with the extracted arguments
    being ``{sym1: sym1, sym2: expr1}``.

    Arguments:
        node: The applied lift expression to transform and extract from.
        reserved_symbol_names: Do not declare any symbols in this list.
        extracted_args: A dictionary into which the arguments are extracted to.
        recorded_shifts_base: When extracting an argument we generate a symbol (key in the
            `extracted_args` dict). Prepend the given shifts to the `recorded_shifts` annex of this
            symbol (the tail is inherited from the stencil argument).

    """
    assert _is_lift(node)
    assert isinstance(node.fun, ir.FunCall)
    inner_stencil = node.fun.args[0]

    new_args = []
    for i, arg in enumerate(node.args):
        if isinstance(arg, ir.SymRef):
            new_symbol = im.sym(arg.id)
            assert new_symbol not in extracted_args or extracted_args[new_symbol] == arg
            extracted_args[new_symbol] = arg
            new_args.append(arg)
        else:
            new_symbol = _generate_unique_symbol(
                desired_name=(inner_stencil, i),
                occupied_names=reserved_symbol_names,
                occupied_symbols=extracted_args.keys(),
            )
            assert new_symbol not in extracted_args
            extracted_args[new_symbol] = arg
            new_arg = im.ref(new_symbol.id)
            trace_shifts.copy_recorded_shifts(from_=arg, to=new_arg, required=False)
            new_args.append(new_arg)

        # todo: test this properly. not really sure about it...
        if recorded_shifts_base is not None:
            if isinstance(inner_stencil, ir.Lambda):
                recorded_shifts = inner_stencil.params[i].annex.recorded_shifts
            elif inner_stencil == im.ref("deref"):
                recorded_shifts = ((),)
            elif _is_scan(inner_stencil) and isinstance(inner_stencil.args[0], ir.Lambda):
                recorded_shifts = inner_stencil.args[0].params[i + 1].annex.recorded_shifts
            else:
                raise AssertionError("Expected a Lambda function or deref as stencil.")
            new_symbol.annex.recorded_shifts = set()
            for outer_shift in recorded_shifts_base:
                for inner_shift in recorded_shifts:
                    new_symbol.annex.recorded_shifts.add((*outer_shift, *inner_shift))

    new_lift = im.lift(inner_stencil)(*new_args)
    trace_shifts.copy_recorded_shifts(from_=node, to=new_lift, required=False)
    new_lift.location = node.location
    return (new_lift, extracted_args)


def _lift_args_eligible_for_inlining(
    node: ir.FunCall, inline_single_pos_deref_lift_args_only: bool
) -> list[bool]:
    assert _is_lift(node)
    stencil = node.fun.args[0]  # type: ignore[attr-defined] # ensured by _is_lift

    eligible_lifted_args = [False] * len(node.args)
    for i, (param, arg) in enumerate(zip(stencil.params, node.args, strict=True)):
        if not _is_lift(arg):
            continue
        # we don't want to inline lift args derefed at multiple positions as this would
        # disallow creating a temporary for them if the (outer) lift can not be extracted into
        # a temporary.
        if inline_single_pos_deref_lift_args_only:
            if len(param.annex.recorded_shifts) > 0:
                continue
        eligible_lifted_args[i] = True

    return eligible_lifted_args


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
        def all(self):  # shadowing a python builtin
            return functools.reduce(operator.or_, self.__members__.values())

    predicate: Callable[[ir.FunCall, bool], bool] = lambda _1, _2: True  # noqa: E731  # assigning a lambda is fine in this special case

    inline_single_pos_deref_lift_args_only: bool = False

    flags: Flag = Flag.all()  # noqa: RUF009 [function-call-in-dataclass-default-argument]

    @classmethod
    def apply(
        cls,
        node: ir.Node,
        *,
        flags: Flag = flags,
        predicate: Callable[[ir.FunCall, bool], bool] = predicate,
        recurse: bool = True,
        inline_single_pos_deref_lift_args_only: bool = inline_single_pos_deref_lift_args_only,
    ) -> ir.Node:
        if inline_single_pos_deref_lift_args_only:
            assert isinstance(node, ir.FencilDefinition)
            trace_shifts.TraceShifts.apply(node, inputs_only=False, save_to_annex=True)

        return cls(
            flags=flags,
            predicate=predicate,
            inline_single_pos_deref_lift_args_only=inline_single_pos_deref_lift_args_only,
        ).visit(node, recurse=recurse, is_scan_pass_context=False)

    def transform_propagate_shift(
        self, node: ir.FunCall, *, recurse: bool, **kwargs
    ) -> Optional[ir.Node]:
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
        trace_shifts.copy_recorded_shifts(from_=node, to=result, required=False)
        return self.visit(result, recurse=False, **kwargs)

    def transform_inline_deref_lift_impl(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        # TODO: this does not work for neighbors(..., lift(...)), which is essentially also a deref
        assert isinstance(node.args[0], ir.FunCall) and isinstance(node.args[0].fun, ir.FunCall)
        assert len(node.args[0].fun.args) == 1
        f = node.args[0].fun.args[0]
        args = node.args[0].args
        new_node = ir.FunCall(fun=f, args=args)
        if isinstance(f, ir.Lambda):
            new_node = inline_lambda(new_node, opcount_preserving=True)
        return self.visit(new_node, **kwargs)

    def transform_inline_deref_lift(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        if not node.fun == ir.SymRef(id="deref"):
            return None
        is_scan_pass_context = kwargs["is_scan_pass_context"]
        is_eligible = _is_lift(node.args[0]) and self.predicate(node.args[0], is_scan_pass_context)
        if not is_eligible:
            return None
        return self.transform_inline_deref_lift_impl(node, **kwargs)

    def transform_inline_trivial_deref_lift(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        if not node.fun == ir.SymRef(id="deref"):
            return None
        is_trivial = _is_lift(node.args[0]) and len(node.args[0].args) == 0
        if not is_trivial:
            return None
        return self.transform_inline_deref_lift_impl(node, **kwargs)

    def transform_propagate_can_deref(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
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

    def transform_inline_lifted_args(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        symtable, is_scan_pass_context = (
            kwargs["symtable"],
            kwargs.get("is_scan_pass_context", False),
        )
        if not (
            _is_lift(node) and len(node.args) > 0 and self.predicate(node, is_scan_pass_context)
        ):
            return None

        stencil = node.fun.args[0]  # type: ignore[attr-defined] # node already asserted to be of type ir.FunCall

        if not isinstance(stencil, ir.Lambda):
            return None

        eligible_lifted_args = _lift_args_eligible_for_inlining(
            node, self.inline_single_pos_deref_lift_args_only
        )

        if not isinstance(stencil, ir.Lambda) or not any(eligible_lifted_args):
            return None

        new_arg_exprs: dict[ir.Sym, ir.Expr] = {}
        inlined_args: dict[ir.Sym, ir.Expr] = {}

        reserved_symbol_names = list(set(symtable.keys()) | {param.id for param in stencil.params})
        for param, arg, eligible in zip(
            stencil.params, node.args, eligible_lifted_args, strict=True
        ):
            if eligible:
                transformed_arg, _ = _transform_and_extract_lift_args(
                    arg,  # type: ignore[arg-type] # ensure by _is_lift on arg
                    reserved_symbol_names,
                    new_arg_exprs,
                    getattr(param.annex, "recorded_shifts", None),
                )
                inlined_args[param] = transformed_arg
            else:
                # the false-branch is completely sufficient, but this makes the resulting
                # expression much more readable
                # TODO(tehrengruber): the true-branch could just be a standalone preprocessing
                #  pass. Since the tests of this transformation rely on it we preserve the
                #  behaviour here for now.
                if isinstance(arg, ir.SymRef) and arg.id not in (
                    reserved_symbol_names + list(new_arg_exprs.keys())
                ):
                    new_param = im.sym(arg.id)
                    trace_shifts.copy_recorded_shifts(
                        from_=param,
                        to=new_param,
                        required=self.inline_single_pos_deref_lift_args_only,
                    )
                    new_arg_exprs[new_param] = arg

                    new_arg = im.ref(arg.id)
                    trace_shifts.copy_recorded_shifts(
                        from_=param,
                        to=new_arg,
                        required=self.inline_single_pos_deref_lift_args_only,
                    )
                    assert param not in inlined_args
                    inlined_args[param] = new_arg
                else:
                    new_arg_exprs[param] = arg

        new_stencil_body = im.let(
            *((param, inlined_lift) for param, inlined_lift in inlined_args.items())  # type: ignore[arg-type] # mypy not smart enough
        )(stencil.expr)

        # it is likely that some of the let args can be immediately inlined. do this eagerly
        # here
        new_stencil_body = inline_lambda(new_stencil_body, opcount_preserving=True)

        new_stencil = im.lambda_(*new_arg_exprs.keys())(new_stencil_body)
        new_applied_lift = im.lift(new_stencil)(*new_arg_exprs.values())
        trace_shifts.copy_recorded_shifts(from_=node, to=new_applied_lift, required=False)
        return self.visit(new_applied_lift, **kwargs)

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Node:
        recurse: bool = kwargs["recurse"]

        if recurse:
            fun_kwargs = {k: v for k, v in kwargs.items() if k != "is_scan_pass_context"}
            new_node = ir.FunCall(
                fun=self.generic_visit(node.fun, is_scan_pass_context=_is_scan(node), **fun_kwargs),
                args=self.generic_visit(node.args, **kwargs),
            )
            trace_shifts.copy_recorded_shifts(from_=node, to=new_node, required=False)
        else:
            new_node = node

        for transformation in self.Flag:
            if self.flags & transformation:
                assert isinstance(transformation.name, str)
                method = getattr(self, f"transform_{transformation.name.lower()}")
                transformed_node = method(new_node, **kwargs)
                # if the transformation returned `None` it did not apply and we continue.
                if transformed_node is not None:
                    return transformed_node

        return new_node
