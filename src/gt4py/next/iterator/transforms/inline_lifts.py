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
from collections.abc import Callable
from typing import Optional

import gt4py.eve as eve
from gt4py.eve import NodeTranslator, traits
from gt4py.next.ffront import itir_makers as im
from gt4py.next.iterator import ir
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
    while new_symbol.id in occupied_names or new_symbol in occupied_symbols:
        new_symbol = ir.Sym(id=new_symbol.id + "_")
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
):
    """
    Transform and extract non-symbol arguments of a lifted stencil call.

    E.g. ``lift(lambda a: ...)(sym1, expr1)`` is transformed into
    ``lift(lambda a: ...)(sym1, sym2)`` with the extracted arguments
    being ``{sym1: sym1, sym2: expr1}``.
    """
    assert _is_lift(node)
    assert isinstance(node.fun, ir.FunCall)
    inner_stencil = node.fun.args[0]

    new_args = []
    for i, arg in enumerate(node.args):
        if isinstance(arg, ir.SymRef):
            sym = ir.Sym(id=arg.id)
            assert sym not in extracted_args or extracted_args[sym] == arg
            extracted_args[sym] = arg
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

    return (im.lift_(inner_stencil)(*new_args), extracted_args)


@dataclasses.dataclass
class InlineLifts(traits.VisitorWithSymbolTableTrait, NodeTranslator):
    """Inline lifted function calls.

    Optionally a predicate function can be passed which can enable or disable inlining of specific function nodes.
    """

    def __init__(self, predicate: Optional[Callable[[ir.Expr, bool], bool]] = None) -> None:
        super().__init__()
        if predicate is None:
            self.predicate = lambda _1, _2: True
        else:
            self.predicate = predicate

    def visit_FunCall(
        self, node: ir.FunCall, *, is_scan_pass_context=False, recurse=True, **kwargs
    ):
        symtable = kwargs["symtable"]

        node = (
            ir.FunCall(
                fun=self.generic_visit(node.fun, is_scan_pass_context=_is_scan(node), **kwargs),
                args=self.generic_visit(node.args, **kwargs),
            )
            if recurse
            else node
        )

        if _is_shift_lift(node):
            # shift(...)(lift(f)(args...)) -> lift(f)(shift(...)(args)...)
            shift = node.fun
            assert len(node.args) == 1
            lift_call = node.args[0]
            new_args = [
                self.visit(ir.FunCall(fun=shift, args=[arg]), recurse=False, **kwargs)
                for arg in lift_call.args  # type: ignore[attr-defined] # lift_call already asserted to be of type ir.FunCall
            ]
            result = ir.FunCall(fun=lift_call.fun, args=new_args)  # type: ignore[attr-defined] # lift_call already asserted to be of type ir.FunCall
            return self.visit(result, recurse=False, **kwargs)
        elif node.fun == ir.SymRef(id="deref"):
            assert len(node.args) == 1
            if _is_lift(node.args[0]) and self.predicate(node.args[0], is_scan_pass_context):
                # deref(lift(f)(args...)) -> f(args...)
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
        elif node.fun == ir.SymRef(id="can_deref"):
            # TODO(havogt): this `can_deref` transformation doesn't look into lifted functions,
            #  this need to be changed to be 100% compliant
            assert len(node.args) == 1
            if _is_lift(node.args[0]) and self.predicate(node.args[0], is_scan_pass_context):
                # can_deref(lift(f)(args...)) -> and(can_deref(arg[0]), and(can_deref(arg[1]), ...))
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
        elif _is_lift(node) and len(node.args) > 0 and self.predicate(node, is_scan_pass_context):
            # Inline arguments to lifted stencil calls, e.g.:
            #  lift(λ(a) → inner_ex(a))(lift(λ(b) → outer_ex(b))(arg))
            # is transformed into:
            #  lift(λ(b) → inner_ex(outer_ex(b)))(arg)
            #  lift(λ(a) → inner_ex(shift(...)(a)))(lift(λ(b) → outer_ex(b))(arg))
            # Note: This branch is only needed when there is no outer `deref` by which the previous
            # branches eliminate the lift calls. This occurs for example for the `reduce` builtin
            # or when a better readable expression of a lift statement is needed during debugging.
            # Due to its complexity we might want to remove this branch at some point again,
            # when we see that it is not required.
            stencil = node.fun.args[0]  # type: ignore[attr-defined] # node already asserted to be of type ir.FunCall
            eligible_lifted_args = [
                _is_lift(arg) and self.predicate(arg, is_scan_pass_context) for arg in node.args
            ]

            if isinstance(stencil, ir.Lambda) and any(eligible_lifted_args):
                # TODO(tehrengruber): we currently only inlining opcount preserving, but what we
                #  actually want is to inline whenever the argument is not shifted. This is
                #  currently beyond the capabilities of the inliner and the shift tracer.
                new_arg_exprs: dict[ir.Sym, ir.Expr] = {}
                inlined_args = []
                for i, (arg, eligible) in enumerate(zip(node.args, eligible_lifted_args)):
                    if eligible:
                        assert isinstance(arg, ir.FunCall)
                        inlined_arg, _ = _transform_and_extract_lift_args(
                            arg, symtable, new_arg_exprs
                        )
                        inlined_args.append(inlined_arg)
                    else:
                        if isinstance(arg, ir.SymRef):
                            new_arg_sym = ir.Sym(id=arg.id)
                        else:
                            new_arg_sym = _generate_unique_symbol(
                                desired_name=(stencil, i),
                                occupied_names=symtable.keys(),
                                occupied_symbols=new_arg_exprs.keys(),
                            )

                        new_arg_exprs[new_arg_sym] = arg
                        inlined_args.append(ir.SymRef(id=new_arg_sym.id))

                inlined_call = self.visit(
                    inline_lambda(
                        ir.FunCall(fun=stencil, args=inlined_args), opcount_preserving=True
                    ),
                    **kwargs,
                )

                new_stencil = im.lambda__(*new_arg_exprs.keys())(inlined_call)
                return im.lift_(new_stencil)(*new_arg_exprs.values())

        return node
