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


def _tree_map_tuple_body(
    f: itir.Expr, tup_refs: list[str], tup_types: list[ts.TupleType]
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
            *(
                functools.reduce(lambda expr, i: im.tuple_get(i, expr), path, im.ref(ref_name))
                for ref_name in tup_refs
            )
        )

    return mapper(*tup_types)


def _map_tuple_body(f: itir.Expr, tup_refs: list[str], tup_types: list[ts.TupleType]) -> itir.Expr:
    """Unroll `map_tuple(f)(t)` over top-level elements only (no recursion)."""
    (ref_name,) = tup_refs
    (tup_type,) = tup_types
    return im.make_tuple(
        *(im.call(f)(im.tuple_get(i, im.ref(ref_name))) for i in range(len(tup_type.types)))
    )


_UNROLLERS = {
    "tree_map_tuple": _tree_map_tuple_body,
    "map_tuple": _map_tuple_body,
}


@dataclasses.dataclass
class UnrollTreeMap(eve.NodeTranslator):
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

        tup_refs = [next(self.uids["_utm"]) for _ in tup_args]
        body = _UNROLLERS[builtin_name](f, tup_refs, tup_types)

        result = im.let(*zip(tup_refs, tup_args))(body)
        itir_inference.reinfer(result)
        return result
