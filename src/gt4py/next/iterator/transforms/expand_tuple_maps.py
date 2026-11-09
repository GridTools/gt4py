# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
import functools
from typing import TypeVar

from gt4py import eve
from gt4py.next import common, utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.type_system import inference as itir_inference
from gt4py.next.type_system import type_info, type_specifications as ts


def _tree_map_tuple_body(f: itir.Expr, tup_expr: itir.Expr, tup_type: ts.TupleType) -> itir.Expr:
    """Recursively expand `tree_map_tuple(f)(t)` into `make_tuple` calls."""

    @type_info.tree_map_type(
        result_collection_constructor=lambda _, elts: im.make_tuple(*elts), with_path_arg=True
    )
    def mapper(_el_type, path):
        return im.call(f)(functools.reduce(lambda expr, i: im.tuple_get(i, expr), path, tup_expr))

    return mapper(tup_type)


def _map_tuple_body(f: itir.Expr, tuple_refs: list[tuple[itir.Expr, ts.TupleType]]) -> itir.Expr:
    """Expand `map_tuple(f)(t1, ..., tn)` over the top-level elements, zipping the tuples."""
    length = len(tuple_refs[0][1].types)
    return im.make_tuple(
        *(im.call(f)(*(im.tuple_get(i, ref) for ref, _ in tuple_refs)) for i in range(length))
    )


ProgramOrExpr = TypeVar("ProgramOrExpr", bound=itir.Program | itir.Expr)


@dataclasses.dataclass
class ExpandTupleMaps(eve.NodeTranslator):
    """Expand tuple-map ITIR builtins (`tree_map_tuple`, `map_tuple`) into `make_tuple`."""

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

        if not cpm.is_call_to(node.fun, ("tree_map_tuple", "map_tuple")):
            return node

        assert isinstance(node.fun, itir.FunCall)
        name = node.fun.fun.id
        f = node.fun.args[0]

        # Let-bind each tuple arg to avoid expression duplication.
        bindings: list[tuple[str, itir.Expr]] = []
        tuple_types: list[ts.TupleType] = []
        for tup in node.args:
            itir_inference.reinfer(tup)
            assert isinstance(tup.type, ts.TupleType)
            bindings.append((next(self.uids["_etm"]), tup))
            tuple_types.append(tup.type)

        if name == "tree_map_tuple":
            ((ref_name, _),) = bindings
            (tup_type,) = tuple_types
            body = _tree_map_tuple_body(f, im.ref(ref_name), tup_type)
        else:  # `map_tuple`: zip all tuples in lockstep
            body = _map_tuple_body(
                f,
                [
                    (im.ref(ref_name), tup_type)
                    for (ref_name, _), tup_type in zip(bindings, tuple_types)
                ],
            )

        result = im.let(*bindings)(body)
        itir_inference.reinfer(result)
        return result
