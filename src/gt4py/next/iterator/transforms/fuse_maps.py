# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import TypeGuard

from gt4py.eve import NodeTranslator, traits
from gt4py.eve.utils import UIDGenerator
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.iterator.transforms import inline_lambdas


def _is_map(node: ir.Node) -> TypeGuard[ir.FunCall]:
    return (
        isinstance(node, ir.FunCall)
        and isinstance(node.fun, ir.FunCall)
        and node.fun.fun == ir.SymRef(id="map_")
    )


@dataclasses.dataclass(frozen=True)
class FuseMaps(traits.PreserveLocationVisitor, traits.VisitorWithSymbolTableTrait, NodeTranslator):
    """
    Fuses nested `map_`s.

    Preconditions:
      - `FunctionDefinitions` are inlined
      - Pass must be only constructed once (and `_fuse_mapsX` are reserved symbols)

    Example:
        map(λ(x, y)->f(x, y))(a, map(λ(z, w)->g(z, w))(b, c))
    to
        map(λ(a, b, c) → f(a, g(b, c)))(a, b, c)

        reduce(λ(x, y) → f(x, y), init)(map_(g(z, w))(a, b))
    to
        reduce(λ(x, y, z) → f(x, g(y, z)), init)(a, b)
    """

    uids: UIDGenerator = dataclasses.field(init=False, repr=False, default_factory=UIDGenerator)

    def _as_lambda(self, fun: ir.SymRef | ir.Lambda, param_count: int) -> ir.Lambda:
        # if fun is already a Lambda we still wrap it to get unique symbol names to avoid symbol clashes
        params = [
            ir.Sym(id=self.uids.sequential_id(prefix="_fuse_maps")) for _ in range(param_count)
        ]
        return ir.Lambda(
            params=params,
            expr=ir.FunCall(fun=fun, args=[ir.SymRef(id=p.id) for p in params]),
            location=fun.location,
        )

    def visit_FunCall(self, node: ir.FunCall, **kwargs):
        node = self.generic_visit(node)
        if _is_map(node) or cpm.is_applied_reduce(node):
            if any(_is_map(arg) for arg in node.args):
                first_param = (
                    0 if _is_map(node) else 1
                )  # index of the first param of op that maps to args (0 for map, 1 for reduce)
                assert isinstance(node.fun, ir.FunCall)
                assert isinstance(node.fun.args[0], (ir.Lambda, ir.SymRef))
                outer_op = self._as_lambda(node.fun.args[0], len(node.args) + first_param)

                inlined_args = []
                new_params = []
                new_args = []
                if cpm.is_applied_reduce(node):
                    # param corresponding to reduce acc
                    inlined_args.append(ir.SymRef(id=outer_op.params[0].id))
                    new_params.append(outer_op.params[0])

                for i in range(len(node.args)):
                    if _is_map(node.args[i]):
                        map_call = node.args[i]
                        assert isinstance(map_call, ir.FunCall)
                        assert isinstance(map_call.fun, ir.FunCall)
                        assert isinstance(map_call.fun.args[0], (ir.Lambda, ir.SymRef))
                        inner_op = self._as_lambda(map_call.fun.args[0], len(map_call.args))
                        inlined_args.append(
                            inline_lambdas.inline_lambda(
                                ir.FunCall(
                                    fun=inner_op,
                                    args=[ir.SymRef(id=param.id) for param in inner_op.params],
                                    location=node.location,
                                )
                            )
                        )
                        new_params.extend(inner_op.params)
                        new_args.extend(map_call.args)
                    else:
                        inlined_args.append(ir.SymRef(id=outer_op.params[i + first_param].id))
                        new_params.append(outer_op.params[i + first_param])
                        new_args.append(node.args[i])
                new_body = ir.FunCall(fun=outer_op, args=inlined_args)
                new_body = inline_lambdas.inline_lambda(
                    new_body
                )  # removes one level of nesting (the recursive inliner could simplify more, however this can also be done on the full tree later)
                new_op = ir.Lambda(params=new_params, expr=new_body)
                if _is_map(node):
                    return ir.FunCall(
                        fun=ir.FunCall(fun=ir.SymRef(id="map_"), args=[new_op]), args=new_args
                    )
                else:  # is_applied_reduce(node)
                    return ir.FunCall(
                        fun=ir.FunCall(fun=ir.SymRef(id="reduce"), args=[new_op, node.fun.args[1]]),
                        args=new_args,
                    )
        return node
