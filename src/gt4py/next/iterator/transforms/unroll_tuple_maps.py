# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
import functools

from gt4py import eve
from gt4py.next import utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.type_system import inference as itir_inference
from gt4py.next.type_system import type_specifications as ts


def _collapsing_tuple_get(expr: itir.Expr, i: int) -> itir.Expr:
    """Like `im.tuple_get`, but collapses immediately when `expr` is a `make_tuple` call.

    Note: argument order is `(expr, i)` to allow use as a `functools.reduce` reducer.
    """
    if cpm.is_call_to(expr, "make_tuple"):
        return expr.args[i]
    return im.tuple_get(i, expr)


def _tree_map_tuple_body(
    f: itir.Expr, tup_exprs: list[itir.Expr], tup_types: list[ts.TupleType]
) -> itir.Expr:
    """Recursively unroll `tree_map_tuple(f)(t1, ..., tN)` into `make_tuple` calls."""

    @utils.tree_map(
        collection_type=ts.TupleType,
        result_collection_constructor=lambda _, elts: im.make_tuple(*elts),
        with_path_arg=True,
    )
    def mapper(*args):
        *_el_types, path = args
        return im.call(f)(
            *(functools.reduce(_collapsing_tuple_get, path, tup_expr) for tup_expr in tup_exprs)
        )

    return mapper(*tup_types)


def _map_tuple_body(
    f: itir.Expr, tup_exprs: list[itir.Expr], tup_types: list[ts.TupleType]
) -> itir.Expr:
    """Unroll `map_tuple(f)(t)` over top-level elements only (no recursion)."""
    (tup_expr,) = tup_exprs
    (tup_type,) = tup_types
    return im.make_tuple(
        *(im.call(f)(_collapsing_tuple_get(tup_expr, i)) for i in range(len(tup_type.types)))
    )


_UNROLLERS = {
    "tree_map_tuple": _tree_map_tuple_body,
    "map_tuple": _map_tuple_body,
}


@dataclasses.dataclass
class UnrollTupleMaps(eve.NodeTranslator):
    """Unroll tuple-map ITIR builtins (`tree_map_tuple`, `map_tuple`) into `make_tuple`."""

    PRESERVED_ANNEX_ATTRS = ("domain",)

    uids: utils.IDGeneratorPool

    @classmethod
    def apply(cls, program: itir.Program, *, uids: utils.IDGeneratorPool):
        return cls(uids=uids).visit(program)

    def visit_FunCall(self, node: itir.FunCall):
        node = self.generic_visit(node)

        builtin_name = next((name for name in _UNROLLERS if cpm.is_call_to(node.fun, name)), None)
        if builtin_name is None:
            return node

        assert isinstance(node.fun, itir.FunCall)
        f = node.fun.args[0]
        tup_args = node.args

        tup_types: list[ts.TupleType] = []
        for tup in tup_args:
            itir_inference.reinfer(tup)
            assert isinstance(tup.type, ts.TupleType)
            tup_types.append(tup.type)

        # For trivial args (those that can be duplicated without cost or side effects),
        # we substitute them directly into the body. This avoids leaving behind
        # `tuple_get(i, make_tuple(...))` patterns that would otherwise require a
        # separate cleanup pass (CollapseTuple). For non-trivial args we still
        # introduce a `let` binding to avoid duplicating expensive sub-expressions.
        substituted_exprs: list[itir.Expr] = []
        let_bindings: list[tuple[str, itir.Expr]] = []
        for tup in tup_args:
            if isinstance(tup, (itir.SymRef, itir.Literal)) or cpm.is_call_to(tup, "make_tuple"):
                substituted_exprs.append(tup)
            else:
                ref_name = next(self.uids["_utm"])
                let_bindings.append((ref_name, tup))
                substituted_exprs.append(im.ref(ref_name))

        body = _UNROLLERS[builtin_name](f, substituted_exprs, tup_types)

        result = im.let(*let_bindings)(body) if let_bindings else body
        itir_inference.reinfer(result)
        return result
