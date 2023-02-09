# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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
from functools import partial
from typing import Optional, Union, cast

from gt4py.eve import NodeTranslator
from gt4py.eve.utils import UIDGenerator
from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.remap_symbols import RemapSymbolRefs


@dataclasses.dataclass(frozen=True)
class PopupTmps(NodeTranslator):
    """Transformation for “popping up” nested lifts to lambda arguments.

    In the simplest case, `(λ(x) → deref(lift(deref)(x)))(y)` is translated to
    `(λ(x, tmp) → deref(tmp))(y, lift(deref)(y))` (where `tmp` is an arbitrary
    new symbol name).

    Note that there are edge cases of lifts which can not be popped up; for
    example, popping up of a lift call that references a closure argument
    (like `lift(deref)(x)` where `x` is a closure argument) is not possible
    as we can not pop the expression to be a closure input (because closures
    just take unmodified fencil arguments as inputs).
    """

    # we use one UID generator per instance such that the generated ids are
    #  stable across multiple runs (required for caching to properly work)
    uids: UIDGenerator = dataclasses.field(init=False, repr=False, default_factory=UIDGenerator)

    @staticmethod
    def _extract_lambda(
        node: ir.FunCall,
    ) -> Optional[
        tuple[ir.Lambda, list[ir.Sym], bool, Callable[[ir.Lambda, list[ir.Expr]], ir.FunCall]]
    ]:
        """Extract the lambda function which is relevant for popping up lifts.

        Further, returns a bool indicating if the given function call was as a
        lift expression and a wrapper function that undos the extraction.

        So The behavior is the following:
        - For `lift(f)(args...)` it returns `(f, True, wrap)`.
        - For `lift(scan(f, dir, init))(args...)` it returns `(f, True, wrap)`.
        - For `lift(reduce(f, init))(args...)` it returns `(None, True, wrap)`.
        - For `f(args...)` it returns `(f, False, wrap)`.
        - For any other expression, it returns `None`.

        The returned `wrap` function undos the extraction in all cases; for example,
        `wrap(f, args...)` returns `lift(f)(args...)` in the first case.
        """
        if isinstance(node.fun, ir.FunCall) and node.fun.fun == ir.SymRef(id="lift"):
            # lifted lambda call or lifted scan
            assert len(node.fun.args) == 1
            fun = node.fun.args[0]

            is_scan = isinstance(fun, ir.FunCall) and fun.fun == ir.SymRef(id="scan")
            is_reduce = isinstance(fun, ir.FunCall) and fun.fun == ir.SymRef(id="reduce")
            if is_scan:
                fun = fun.args[0]  # type: ignore[attr-defined] # fun already asserted to be of type ir.FunCall
                assert isinstance(fun, ir.Lambda)
                params = fun.params[1:]
            elif is_reduce:
                fun = fun.args[0]  # type: ignore[attr-defined] # fun already asserted to be of type ir.FunCall
                assert isinstance(fun, ir.Lambda)
                params = fun.params[1:]
            else:
                assert isinstance(fun, ir.Lambda)
                params = fun.params

            def wrap(fun: ir.Lambda, args: list[ir.Expr]) -> ir.FunCall:
                if is_scan:
                    assert isinstance(node.fun, ir.FunCall) and isinstance(
                        node.fun.args[0], ir.FunCall
                    )  # TODO(fthaler): first part of the assertion already checked above, however mypy does not catch it
                    scan_args = [cast(ir.Expr, fun)] + node.fun.args[0].args[1:]
                    f: Union[ir.Lambda, ir.FunCall] = ir.FunCall(
                        fun=ir.SymRef(id="scan"), args=scan_args
                    )
                elif is_reduce:
                    assert isinstance(node.fun, ir.FunCall) and isinstance(
                        node.fun.args[0], ir.FunCall
                    )  # TODO(fthaler): first part of the assertion already checked above, however mypy does not catch it
                    assert fun == node.fun.args[0].args[0], "Unexpected lift in reduction function."
                    f = node.fun.args[0]
                else:
                    f = fun
                return ir.FunCall(fun=ir.FunCall(fun=ir.SymRef(id="lift"), args=[f]), args=args)

            assert isinstance(fun, ir.Lambda)
            return fun, params, True, wrap
        if isinstance(node.fun, ir.Lambda):
            # direct lambda call

            def wrap(fun: ir.Lambda, args: list[ir.Expr]) -> ir.FunCall:
                return ir.FunCall(fun=fun, args=args)

            return node.fun, node.fun.params, False, wrap

        return None

    def visit_FunCall(
        self, node: ir.FunCall, *, lifts: Optional[dict[ir.Expr, ir.SymRef]] = None
    ) -> Union[ir.SymRef, ir.FunCall]:
        if call_info := self._extract_lambda(node):
            fun, params, is_lift, wrap = call_info

            nested_lifts = dict[ir.Expr, ir.SymRef]()
            fun = self.visit(fun, lifts=nested_lifts)
            # Note: lifts in arguments are just passed to the parent node
            args = self.visit(node.args, lifts=lifts)

            if is_lift:
                assert lifts is not None

                # check if the lifted expression captures symbols from the outer scope
                symrefs = fun.walk_values().if_isinstance(ir.SymRef).getattr("id").to_set()
                captured = (
                    symrefs
                    - {p.id for p in fun.params}
                    - {n.id for n in nested_lifts.values()}
                    - ir.BUILTINS
                )
                if captured:
                    # if symbols from an outer scope are captured, the lift has to
                    # be handled at that scope, so skip here and pass nested lifts on
                    lifts |= nested_lifts
                    return wrap(fun, args)

            # remap referenced function parameters in lift expression to passed argument values
            assert len(params) == len(args)
            symbol_map = {str(param.id): arg for param, arg in zip(params, args)}
            remap = partial(RemapSymbolRefs().visit, symbol_map=symbol_map)

            nested_lifts = {remap(expr): ref for expr, ref in nested_lifts.items()}
            if lifts:
                # lifts have to be updated in place as they are passed to parent node
                lifted = list(lifts.items())
                lifts.clear()
                for expr, ref in lifted:
                    lifts[remap(expr)] = remap(ref)

            # extend parameter list of the function with popped lifts
            new_params = [ir.Sym(id=p.id) for p in nested_lifts.values()]
            fun = ir.Lambda(params=fun.params + new_params, expr=fun.expr)
            # for the arguments, we have to resolve possible cross-references of lifts
            symbol_map = {str(v.id): k for k, v in nested_lifts.items()}
            new_args = [
                RemapSymbolRefs().visit(a, symbol_map=symbol_map) for a in nested_lifts.keys()
            ]

            # updated function call, having lifts passed as arguments
            call = wrap(fun, args + new_args)

            if not is_lift:
                # if this is not a lift expression, we are done...
                return call

            # ... otherwise we check if the same expression has already been
            # lifted before, then we reference that one
            assert lifts is not None
            if (previous_ref := lifts.get(call)) is not None:
                return previous_ref

            # if this is the first time we lift that expression, create a new
            # symbol for it and register it so the parent node knows about it
            ref = ir.SymRef(id=self.uids.sequential_id(prefix="_lift"))
            lifts[call] = ref
            return ref
        return self.generic_visit(node, lifts=lifts)
