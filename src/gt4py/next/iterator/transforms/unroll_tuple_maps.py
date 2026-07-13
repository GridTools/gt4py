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
from gt4py.next.type_system import type_info, type_specifications as ts


def _tree_map_tuple_body(
    f: itir.Expr, tup_exprs: list[itir.Expr], tup_types: list[ts.TupleType]
) -> itir.Expr:
    """Recursively unroll `tree_map_tuple(f)(t1, ..., tN)` into `make_tuple` calls."""

    expected_structure = type_info.tuple_structure(tup_types[0])
    if any(type_info.tuple_structure(tup_type) != expected_structure for tup_type in tup_types[1:]):
        raise TypeError("'tree_map_tuple' requires all arguments to have the same tuple structure.")

    @utils.tree_map(
        collection_type=ts.TupleType,
        result_collection_constructor=lambda _, elts: im.make_tuple(*elts),
        with_path_arg=True,
    )
    def mapper(*args):
        *_el_types, path = args
        return im.call(f)(
            *(
                functools.reduce(lambda expr, i: im.tuple_get(i, expr), path, tup_expr)
                for tup_expr in tup_exprs
            )
        )

    return mapper(*tup_types)


def _map_tuple_body(
    f: itir.Expr, tup_exprs: list[itir.Expr], tup_types: list[ts.TupleType]
) -> itir.Expr:
    """Unroll `map_tuple(f)(t)` over top-level elements only (no recursion)."""
    (tup_expr,) = tup_exprs
    (tup_type,) = tup_types
    return im.make_tuple(
        *(im.call(f)(im.tuple_get(i, tup_expr)) for i in range(len(tup_type.types)))
    )


_UNROLLERS: dict[str, Callable[[itir.Expr, list[itir.Expr], list[ts.TupleType]], itir.Expr]] = {
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
        tup_args = node.args

        tup_types: list[ts.TupleType] = []
        for tup in tup_args:
            itir_inference.reinfer(tup)
            assert isinstance(tup.type, ts.TupleType)
            tup_types.append(tup.type)

        # Let bind tuple args to avoid expression duplication
        let_bindings = {next(self.uids["_utm"]): tup for tup in tup_args}
        substituted_exprs: list[itir.Expr] = [im.ref(ref_name) for ref_name in let_bindings.keys()]

        body = _UNROLLERS[node.fun.fun.id](f, substituted_exprs, tup_types)

        result = im.let(*let_bindings.items())(body)
        itir_inference.reinfer(result)
        return result
