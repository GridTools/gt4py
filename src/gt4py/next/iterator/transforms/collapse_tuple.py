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
import enum
from typing import Callable

import dataclasses

from gt4py import eve
from gt4py.eve.utils import UIDGenerator
from gt4py.next import type_inference
from gt4py.next.iterator import ir, type_inference as it_type_inference, ir_makers as im
from gt4py.next.iterator.transforms.inline_lambdas import inline_lambda


def _get_tuple_size(type_: type_inference.Type) -> int:
    assert isinstance(type_, it_type_inference.Val) and isinstance(
        type_.dtype, it_type_inference.Tuple
    )
    return len(type_.dtype)

def _is_let(node: ir.Node) -> bool:
    return isinstance(node, ir.FunCall) and isinstance(node.fun, ir.Lambda)

def _is_if_call(node: ir.Expr):
    return isinstance(node, ir.FunCall) and node.fun == im.ref("if_")

def _with_altered_arg(node: ir.FunCall, arg_idx: int, new_arg: ir.Expr):
    return ir.FunCall(
        fun=node.fun,
        args=[arg if i != arg_idx else new_arg for i, arg in enumerate(node.args)]
    )

def _is_trivial_make_tuple_call(node: ir.Expr):
    if not (isinstance(node, ir.FunCall) and node.fun == im.ref("make_tuple")):
        return False
    if not all(isinstance(arg, (ir.SymRef, ir.Literal)) or _is_trivial_make_tuple_call(arg) for arg in node.args):
        return False
    return True

def nlet(bindings: list[tuple[ir.Sym, str], ir.Expr]):
    return im.let(*[el for tup in bindings for el in tup])

@dataclasses.dataclass(frozen=True)
class CollapseTuple(eve.NodeTranslator):
    """
    Simplifies `make_tuple`, `tuple_get` calls.

      - `make_tuple(tuple_get(0, t), tuple_get(1, t), ..., tuple_get(N-1,t))` -> `t`
      - `tuple_get(i, make_tuple(e_0, e_1, ..., e_i, ..., e_N))` -> `e_i`
    """

    class Flag(enum.IntEnum):
        #: `make_tuple(tuple_get(0, t), tuple_get(1, t), ..., tuple_get(N-1,t))` -> `t`
        COLLAPSE_MAKE_TUPLE_TUPLE_GET = 1
        #: `tuple_get(i, make_tuple(e_0, e_1, ..., e_i, ..., e_N))` -> `e_i`
        COLLAPSE_TUPLE_GET_MAKE_TUPLE = 2
        #: `tuple_get(i, let(...)(make_tuple()))` -> `let(...)(tuple_get(i, make_tuple()))`
        PROPAGATE_TUPLE_GET = 4
        #: `{1, 2}` -> `(λ(_tuple_el_1, _tuple_el_2) → {_tuple_el_1, _tuple_el_2})(1, 2)`
        LETIFY_MAKE_TUPLE_ELEMENTS = 8
        #: TODO
        INLINE_TRIVIAL_MAKE_TUPLE = 16
        #: TODO
        PROPAGATE_TO_IF_ON_TUPLES = 32
        #: TODO
        PROPAGATE_NESTED_LET=64
        #: TODO
        INLINE_TRIVIAL_LET=128

    ignore_tuple_size: bool
    flags: int = (Flag.COLLAPSE_MAKE_TUPLE_TUPLE_GET
                 | Flag.COLLAPSE_TUPLE_GET_MAKE_TUPLE
                 | Flag.PROPAGATE_TUPLE_GET
                 | Flag.LETIFY_MAKE_TUPLE_ELEMENTS
                 | Flag.INLINE_TRIVIAL_MAKE_TUPLE
                 | Flag.PROPAGATE_TO_IF_ON_TUPLES
                 | Flag.PROPAGATE_NESTED_LET
                 | Flag.INLINE_TRIVIAL_LET)

    PRESERVED_ANNEX_ATTRS = ("type",)

    # we use one UID generator per instance such that the generated ids are
    # stable across multiple runs (required for caching to properly work)
    _letify_make_tuple_uids: UIDGenerator = dataclasses.field(
        init=False, repr=False, default_factory=lambda: UIDGenerator(prefix="_tuple_el")
    )

    @classmethod
    def apply(
        cls,
        node: ir.Node,
        *,
        ignore_tuple_size: bool = False,
        # manually passing flags is mostly for allowing separate testing of the modes
        flags = None
    ) -> ir.Node:
        """
        Simplifies `make_tuple`, `tuple_get` calls.

        If `ignore_tuple_size`, apply the transformation even if length of the inner tuple
        is greater than the length of the outer tuple.
        """
        it_type_inference.infer_all(node, save_to_annex=True)

        return cls(
            ignore_tuple_size=ignore_tuple_size,
            flags=flags or cls.flags
        ).visit(node)

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Node:
        node = self.generic_visit(node)

        if (
            self.flags & self.Flag.COLLAPSE_MAKE_TUPLE_TUPLE_GET
            and node.fun == ir.SymRef(id="make_tuple")
            and all(
                isinstance(arg, ir.FunCall) and arg.fun == ir.SymRef(id="tuple_get")
                for arg in node.args
            )
        ):
            # `make_tuple(tuple_get(0, t), tuple_get(1, t), ..., tuple_get(N-1,t))` -> `t`
            assert isinstance(node.args[0], ir.FunCall)
            first_expr = node.args[0].args[1]

            for i, v in enumerate(node.args):
                assert isinstance(v, ir.FunCall)
                assert isinstance(v.args[0], ir.Literal)
                if not (int(v.args[0].value) == i and v.args[1] == first_expr):
                    # tuple argument differs, just continue with the rest of the tree
                    return self.generic_visit(node)

            if self.ignore_tuple_size or _get_tuple_size(first_expr.annex.type) == len(
                node.args
            ):
                return first_expr

        if (
            self.flags & self.Flag.COLLAPSE_TUPLE_GET_MAKE_TUPLE
            and node.fun == ir.SymRef(id="tuple_get")
            and isinstance(node.args[1], ir.FunCall)
            and node.args[1].fun == ir.SymRef(id="make_tuple")
            and isinstance(node.args[0], ir.Literal)
        ):
            # `tuple_get(i, make_tuple(e_0, e_1, ..., e_i, ..., e_N))` -> `e_i`
            assert node.args[0].type in ir.INTEGER_BUILTINS
            make_tuple_call = node.args[1]
            idx = int(node.args[0].value)
            assert idx < len(
                make_tuple_call.args
            ), f"Index {idx} is out of bounds for tuple of size {len(make_tuple_call.args)}"
            return node.args[1].args[idx]

        if (
            self.flags & self.Flag.PROPAGATE_TUPLE_GET
            and node.fun == ir.SymRef(id="tuple_get")
            and isinstance(node.args[0], ir.Literal)  # TODO: extend to general symbols as long as the tail call in the let does not capture
        ):
            # `tuple_get(i, let(...)(make_tuple()))` -> `let(...)(tuple_get(i, make_tuple()))`
            if _is_let(node.args[1]):
                idx, let_expr = node.args
                return self.visit(
                    im.call(im.lambda_(*let_expr.fun.params)(im.tuple_get(idx, let_expr.fun.expr)))(*let_expr.args)
                )
            elif isinstance(node.args[1], ir.FunCall) and node.args[1].fun == im.ref("if_"):
                idx = node.args[0]
                cond, true_branch, false_branch = node.args[1].args
                return self.visit(
                    im.if_(cond, im.tuple_get(idx, true_branch), im.tuple_get(idx, false_branch))
                ) # todo: check if visit needed

        if (
            self.flags & self.Flag.LETIFY_MAKE_TUPLE_ELEMENTS
            and node.fun == ir.SymRef(id="make_tuple")
        ):
            # `make_tuple(expr1, expr1)`
            # -> `let((_tuple_el_1, expr1), (_tuple_el_2, expr2))(make_tuple(_tuple_el_1, _tuple_el_2))`
            bound_vars: dict[str, ir.Expr] = {}
            new_args: list[ir.Expr] = []
            for i, arg in enumerate(node.args):
                if isinstance(node, ir.FunCall) and node.fun == im.ref(
                        "make_tuple") and not _is_trivial_make_tuple_call(node):
                    el_name = self._letify_make_tuple_uids.sequential_id()
                    new_args.append(im.ref(el_name))
                    bound_vars[el_name] = arg
                else:
                    new_args.append(arg)

            if bound_vars:
                return self.visit(im.let(*(el for item in bound_vars.items() for el in item))(
                    im.call(node.fun)(*new_args)))

        if self.flags & self.Flag.INLINE_TRIVIAL_MAKE_TUPLE and _is_let(node):
            # `let(tup, make_tuple(trivial_expr1, trivial_expr2))(tup)`
            #  -> `make_tuple(trivial_expr1, trivial_expr2)`
            eligible_params = [_is_trivial_make_tuple_call(arg) for arg in node.args]
            if any(eligible_params):
                return self.visit(inline_lambda(node, eligible_params=eligible_params))

        if self.flags & self.Flag.PROPAGATE_TO_IF_ON_TUPLES and not node.fun == im.ref("if_"):
            # TODO(tehrengruber): This significantly increases the size of the tree. Revisit.
            # TODO(tehrengruber): Only inline if type of branch value is a tuple.
            # `(if cond then {1, 2} else {3, 4})[0]` -> `if cond then {1, 2}[0] else {3, 4}[0]`
            for i, arg in enumerate(node.args):
                if _is_if_call(arg):
                    cond, true_branch, false_branch = arg.args
                    new_true_branch = self.visit(_with_altered_arg(node, i, true_branch), **kwargs)
                    new_false_branch = self.visit(_with_altered_arg(node, i, false_branch), **kwargs)
                    return im.if_(cond, new_true_branch, new_false_branch)

        if self.flags & self.Flag.PROPAGATE_NESTED_LET and _is_let(node):
            # `let((a, let(b, 1)(a_val)))(a)`-> `let(b, 1)(let(a, a_val)(a))`
            outer_vars = {}
            inner_vars = {}
            original_inner_expr = node.fun.expr
            for arg_sym, arg in zip(node.fun.params, node.args):
                if _is_let(arg):
                    for sym, val in zip(arg.fun.params, arg.args):
                        outer_vars[sym] = val
                    assert arg_sym not in inner_vars  # TODO: fix collisions
                    inner_vars[arg_sym] = arg.fun.expr
                else:
                    inner_vars[arg_sym] = arg
            if outer_vars:
                node = self.visit(nlet(tuple(outer_vars.items()))(nlet(tuple(inner_vars.items()))(original_inner_expr)))

        if self.flags & self.Flag.INLINE_TRIVIAL_LET and _is_let(node) and isinstance(node.fun.expr, ir.SymRef):
            # `let(a, 1)(a)` -> `1`
            for arg_sym, arg in zip(node.fun.params, node.args):
                if node.fun.expr == im.ref(arg_sym.id):
                    return arg

        return node