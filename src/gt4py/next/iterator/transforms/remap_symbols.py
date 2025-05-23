# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Optional, Set

from gt4py.eve import NodeTranslator, PreserveLocationVisitor, SymbolTableTrait
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import symbol_ref_utils
from gt4py.next.iterator.type_system import inference as type_inference


def unique_name(name, prohibited_symbols):
    while name in prohibited_symbols:
        name += "_"
    return name


class RemapSymbolRefs(PreserveLocationVisitor, NodeTranslator):
    # This pass preserves, but doesn't use the `type`, `recorded_shifts`, `domain` annex.
    PRESERVED_ANNEX_ATTRS = ("type", "recorded_shifts", "domain")

    @classmethod
    def apply(cls, node: ir.Node, symbol_map: Dict[str, ir.Node]):
        external_symbols = set().union(
            *(symbol_ref_utils.collect_symbol_refs(expr) for expr in [node, *symbol_map.values()])
        )
        return cls().visit(node, symbol_map=symbol_map, reserved_params=external_symbols)

    def visit_SymRef(
        self, node: ir.SymRef, *, symbol_map: Dict[str, ir.Node], reserved_params: set[str]
    ):
        return symbol_map.get(str(node.id), node)

    def visit_Lambda(
        self, node: ir.Lambda, *, symbol_map: Dict[str, ir.Node], reserved_params: set[str]
    ):
        params = {str(p.id) for p in node.params}

        clashes = params & reserved_params
        if clashes:
            reserved_params = {*reserved_params}
            new_symbol_map: Dict[str, ir.Node] = {}
            new_params: list[ir.Sym] = []
            for param in node.params:
                if param.id in clashes:
                    new_param = im.sym(unique_name(param.id, reserved_params), param.type)
                    assert new_param.id not in symbol_map
                    new_symbol_map[param.id] = im.ref(new_param.id, param.type)
                    reserved_params.add(new_param.id)
                else:
                    new_param = param
                new_params.append(new_param)

            new_symbol_map = symbol_map | new_symbol_map
        else:
            new_params = node.params  # keep params as is
            new_symbol_map = symbol_map

        filtered_symbol_map = {k: v for k, v in new_symbol_map.items() if k not in new_params}
        return ir.Lambda(
            params=new_params,
            expr=self.visit(
                node.expr, symbol_map=filtered_symbol_map, reserved_params=reserved_params
            ),
        )

    def generic_visit(self, node: ir.Node, **kwargs: Any):  # type: ignore[override]
        assert isinstance(node, SymbolTableTrait) == isinstance(
            node, ir.Lambda
        ), "found unexpected new symbol scope"
        return super().generic_visit(node, **kwargs)


class RenameSymbols(PreserveLocationVisitor, NodeTranslator):
    # This pass preserves, but doesn't use the `type`, `recorded_shifts`, `domain` annex.
    PRESERVED_ANNEX_ATTRS = ("type", "recorded_shifts", "domain")

    def visit_Sym(
        self, node: ir.Sym, *, name_map: Dict[str, str], active: Optional[Set[str]] = None
    ):
        if active and node.id in active:
            return ir.Sym(id=name_map.get(node.id, node.id))
        return node

    def visit_SymRef(
        self, node: ir.SymRef, *, name_map: Dict[str, str], active: Optional[Set[str]] = None
    ):
        if active and node.id in active:
            new_ref = ir.SymRef(id=name_map.get(node.id, node.id))
            type_inference.copy_type(from_=node, to=new_ref, allow_untyped=True)
            return new_ref
        return node

    def generic_visit(  # type: ignore[override]
        self, node: ir.Node, *, name_map: Dict[str, str], active: Optional[Set[str]] = None
    ):
        if isinstance(node, SymbolTableTrait):
            if active is None:
                active = set()
            active = active | set(node.annex.symtable)
        return super().generic_visit(node, name_map=name_map, active=active)
