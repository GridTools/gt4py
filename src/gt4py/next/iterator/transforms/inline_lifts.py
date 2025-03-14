# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import enum
from typing import Callable, ClassVar, Optional

import gt4py.eve as eve
from gt4py.eve import NodeTranslator, traits
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im
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
    node: ir.FunCall, symtable: dict[eve.SymbolName, ir.Sym], extracted_args: dict[ir.Sym, ir.Expr]
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
            # TODO(tehrengruber): Is it possible to reinfer the type if it is not inherited here?
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
            # TODO(tehrengruber): Is it possible to reinfer the type if it is not inherited here?
            new_args.append(ir.SymRef(id=new_symbol.id))

    itir_node = im.lift(inner_stencil)(*new_args)
    itir_node.location = node.location
    return (itir_node, extracted_args)


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

    PRESERVED_ANNEX_ATTRS: ClassVar[tuple[str, ...]] = ("domain",)

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

    predicate: Callable[[ir.Expr, bool], bool] = lambda _1, _2: True

    flags: int = (
        Flag.PROPAGATE_SHIFT
        | Flag.INLINE_DEREF_LIFT
        | Flag.PROPAGATE_CAN_DEREF
        | Flag.INLINE_LIFTED_ARGS
    )

    def visit_FunCall(
        self, node: ir.FunCall, *, is_scan_pass_context=False, recurse=True, **kwargs
    ):
        symtable = kwargs["symtable"]

        node = (
            ir.FunCall(
                fun=self.generic_visit(node.fun, is_scan_pass_context=_is_scan(node), **kwargs),
                args=self.generic_visit(node.args, **kwargs),
                type=node.type,
            )
            if recurse
            else node
        )

        if self.flags & self.Flag.PROPAGATE_SHIFT and _is_shift_lift(node):
            shift = node.fun
            # This transformation does not preserve the type (the position dims of the iterator
            # change). Delete type to avoid errors.
            shift.type = None
            assert len(node.args) == 1
            lift_call = node.args[0]
            new_args = [
                self.visit(ir.FunCall(fun=shift, args=[arg]), recurse=False, **kwargs)
                for arg in lift_call.args  # type: ignore[attr-defined] # lift_call already asserted to be of type ir.FunCall
            ]
            result = ir.FunCall(fun=lift_call.fun, args=new_args)  # type: ignore[attr-defined] # lift_call already asserted to be of type ir.FunCall
            return self.visit(result, recurse=False, **kwargs)
        elif self.flags & self.Flag.INLINE_DEREF_LIFT and node.fun == ir.SymRef(id="deref"):
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
        elif self.flags & self.Flag.PROPAGATE_CAN_DEREF and node.fun == ir.SymRef(id="can_deref"):
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
                    return im.literal_from_value(True)

                res = im.can_deref(args[0])
                for arg in args[1:]:
                    res = ir.FunCall(
                        fun=ir.SymRef(id="and_"),
                        args=[res, im.can_deref(arg)],
                    )
                return res
        elif (
            self.flags & self.Flag.INLINE_LIFTED_ARGS
            and _is_lift(node)
            and len(node.args) > 0
            and self.predicate(node, is_scan_pass_context)
        ):
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

                new_stencil = im.lambda_(*new_arg_exprs.keys())(inlined_call)
                return im.lift(new_stencil)(*new_arg_exprs.values())

        return node
