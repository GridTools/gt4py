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
from typing import TypeGuard

from gt4py.eve import NodeTranslator, traits
from gt4py.eve.utils import UIDGenerator
from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.inline_lambdas import inline_lambda


def _is_map(node: ir.Node) -> TypeGuard[ir.FunCall]:
    return (
        isinstance(node, ir.FunCall)
        and isinstance(node.fun, ir.FunCall)
        and node.fun.fun == ir.SymRef(id="map_")
    )


@dataclasses.dataclass(frozen=True)
class FuseMaps(traits.VisitorWithSymbolTableTrait, NodeTranslator):
    """
    Fuses nested `map_`s.

    Preconditions:
      - `FunctionDefinitions` are inlined

    Example:
        map(λ(x, y)->f(x, y))(a, map(λ(z, w)->g(z, w))(b, c))
    to
        TODOmap(λ(a, b, c) → f(a, g(b, c)))(a, b, c)


    map(λ(x, b,c)->f(x, (λ(z, w)->g(z, w))(b,c)))(a, map(λ(z, w)->g(z, w))(b, c))

    λ(x, b, c)->(λ(x, y)->f(x, y))(x, λ(z, w)->g(z, w))(b,c)


    Algorithm:
      - example: map(λ(x, y)->f(x, y))(a, map(λ(z, w)->g(z, w))(b, c))
      - the arguments to the mapped operation are either `SymRef`s or calls to `neighbor` or calls to `deref`
      - the new op is a lambda where the to-inline argument is replaced by the
      - create a new op from
    """

    uids: UIDGenerator = dataclasses.field(init=False, repr=False, default_factory=UIDGenerator)

    def _as_lambda(self, fun: ir.SymRef | ir.Lambda, param_count: int) -> ir.Lambda:
        if isinstance(fun, ir.Lambda):
            return fun
        params = [ir.Sym(id=self.uids.sequential_id(prefix="sym")) for _ in range(param_count)]
        return ir.Lambda(
            params=params,
            expr=ir.FunCall(fun=fun, args=[ir.SymRef(id=p.id) for p in params]),
        )

    # TODO think about clashes
    def visit_FunCall(self, node: ir.FunCall, **kwargs):
        node = self.generic_visit(node)
        if _is_map(node):
            if any(_is_map(arg) for arg in node.args):
                assert isinstance(node.fun, ir.FunCall)
                assert isinstance(node.fun.args[0], (ir.Lambda, ir.SymRef))
                outer_op = self._as_lambda(node.fun.args[0], len(node.args))
                # inner_op =
                inlined_args = []
                new_params = []
                new_args = []
                for i in range(len(node.args)):
                    if _is_map(node.args[i]):
                        map_call = node.args[i]
                        assert isinstance(map_call, ir.FunCall)
                        assert isinstance(map_call.fun, ir.FunCall)
                        assert isinstance(map_call.fun.args[0], (ir.Lambda, ir.SymRef))
                        inner_op = self._as_lambda(map_call.fun.args[0], len(map_call.args))
                        inlined_args.append(
                            inline_lambda(
                                ir.FunCall(
                                    fun=inner_op,
                                    args=[*(ir.SymRef(id=param.id) for param in inner_op.params)],
                                )
                            )
                        )
                        new_params.extend(inner_op.params)
                        new_args.extend(map_call.args)
                    else:
                        inlined_args.append(ir.SymRef(id=outer_op.params[i].id))
                        new_params.append(outer_op.params[i])
                        new_args.append(node.args[i])

                new_body = inline_lambda(
                    ir.FunCall(
                        fun=outer_op,
                        args=inlined_args,
                    )
                )
                new_op = ir.Lambda(
                    params=new_params,
                    expr=new_body,
                )
                result = ir.FunCall(
                    fun=ir.FunCall(fun=ir.SymRef(id="map_"), args=[new_op]),
                    args=new_args,
                )
                return result
        return self.generic_visit(node)
