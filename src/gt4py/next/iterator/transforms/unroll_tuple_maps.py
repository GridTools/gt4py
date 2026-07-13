# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
import functools
from typing import Callable, TypeVar

from gt4py import eve
from gt4py.next import common, utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.type_system import inference as itir_inference
from gt4py.next.type_system import type_specifications as ts


def _tree_map_tuple_body(f: itir.Expr, tup_expr: itir.Expr, tup_type: ts.TupleType) -> itir.Expr:
    """Recursively unroll `tree_map_tuple(f)(t)` into `make_tuple` calls."""

    @utils.tree_map(
        collection_type=ts.TupleType,
        result_collection_constructor=lambda _, elts: im.make_tuple(*elts),
        with_path_arg=True,
    )
    def mapper(_el_type, path):
        return im.call(f)(functools.reduce(lambda expr, i: im.tuple_get(i, expr), path, tup_expr))

    return mapper(tup_type)


def _map_tuple_body(f: itir.Expr, tup_expr: itir.Expr, tup_type: ts.TupleType) -> itir.Expr:
    """Unroll `map_tuple(f)(t)` over top-level elements only (no recursion)."""
    return im.make_tuple(
        *(im.call(f)(im.tuple_get(i, tup_expr)) for i in range(len(tup_type.types)))
    )


_UNROLLERS: dict[str, Callable[[itir.Expr, itir.Expr, ts.TupleType], itir.Expr]] = {
    "tree_map_tuple": _tree_map_tuple_body,
    "map_tuple": _map_tuple_body,
}


ProgramOrExpr = TypeVar("ProgramOrExpr", bound=itir.Program | itir.Expr)


@dataclasses.dataclass
class UnrollTupleMaps(eve.NodeTranslator):
    """Unroll tuple-map ITIR builtins (`tree_map_tuple`, `map_tuple`) into `make_tuple`."""

    PRESERVED_ANNEX_ATTRS = ("domain",)

    uids: utils.IDGeneratorPool

    @classmethod
    def apply(
        cls,
        node: ProgramOrExpr,
        *,
        uids: utils.IDGeneratorPool | None,
        offset_provider_type: common.OffsetProviderType | None = None,
    ) -> ProgramOrExpr:
        if node.type is None:
            node = itir_inference.infer(
                node,
                offset_provider_type=offset_provider_type or {},
                allow_undeclared_symbols=not isinstance(node, itir.Program),
            )
        if uids is None:
            uids = utils.IDGeneratorPool()
        return cls(uids=uids).visit(node)

    def visit_FunCall(self, node: itir.FunCall):
        node = self.generic_visit(node)

        if not cpm.is_call_to(node.fun, tuple(_UNROLLERS.keys())):
            return node

        assert isinstance(node.fun, itir.FunCall)
        f = node.fun.args[0]
        (tup,) = node.args
        itir_inference.reinfer(tup)
        assert isinstance(tup.type, ts.TupleType)

        # Let bind the tuple arg to avoid expression duplication.
        ref_name = next(self.uids["_utm"])
        body = _UNROLLERS[node.fun.fun.id](f, im.ref(ref_name), tup.type)

        result = im.let(ref_name, tup)(body)
        itir_inference.reinfer(result)
        return result
