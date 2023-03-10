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
from typing import Iterable, Sequence

import gt4py.eve as eve
from gt4py.next.iterator import ir as itir


@dataclasses.dataclass
class CountSymbolRefs(eve.NodeVisitor):
    ref_counts: dict[str, int]

    @classmethod
    def apply(
        cls, node: itir.Node | Sequence[itir.Node], symbol_names: Iterable[str]
    ) -> dict[str, int]:
        ref_counts = {name: 0 for name in symbol_names}
        active_refs = set(symbol_names)

        obj = cls(ref_counts=ref_counts)
        obj.visit(node, active_refs=active_refs)

        return obj.ref_counts

    def visit_SymRef(self, node: itir.SymRef, *, active_refs: set[str]):
        if node.id in active_refs:
            self.ref_counts[node.id] += 1

    def visit_Lambda(self, node: itir.Lambda, *, active_refs: set[str]):
        active_refs = active_refs - {param.id for param in node.params}

        self.generic_visit(node, active_refs=active_refs)


def collect_symbol_refs(
    node: itir.Node | Sequence[itir.Node], symbol_names: Iterable[str]
) -> list[str]:
    return [
        symbol_name
        for symbol_name, count in CountSymbolRefs.apply(node, symbol_names).items()
        if count > 0
    ]


def get_user_defined_symbols(symtable: dict[eve.SymbolName, itir.Sym]) -> set[str]:
    return {str(sym) for sym in symtable.keys()} - {
        str(n.id) for n in itir.FencilDefinition._NODE_SYMBOLS_
    }
