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
from collections import defaultdict
from typing import Iterable, Optional, Sequence

import gt4py.eve as eve
from gt4py.next.iterator import ir as itir


@dataclasses.dataclass
class CountSymbolRefs(eve.NodeVisitor):
    ref_counts: dict[str, int] = dataclasses.field(default_factory=lambda: defaultdict(int))

    @classmethod
    def apply(
        cls,
        node: itir.Node | Sequence[itir.Node],
        symbol_names: Optional[Iterable[str]] = None,
        *,
        ignore_builtins: bool = True,
    ) -> dict[str, int]:
        """
        Count references to given or all symbols in scope.

        Examples:
            >>> import gt4py.next.iterator.ir_utils.ir_makers as im
            >>> expr = im.plus(im.plus("x", "y"), im.plus(im.plus("x", "y"), "z"))
            >>> CountSymbolRefs.apply(expr)
            {'x': 2, 'y': 2, 'z': 1}

            If only some symbols are of interests the search can be restricted:

            >>> CountSymbolRefs.apply(expr, symbol_names=["x", "z"])
            {'x': 2, 'z': 1}
        """
        if ignore_builtins:
            inactive_refs = {str(n.id) for n in itir.FencilDefinition._NODE_SYMBOLS_}
        else:
            inactive_refs = set()

        obj = cls()
        obj.visit(node, inactive_refs=inactive_refs)

        if symbol_names:
            return {k: obj.ref_counts.get(k, 0) for k in symbol_names}
        return dict(obj.ref_counts)

    def visit_SymRef(self, node: itir.SymRef, *, inactive_refs: set[str]):
        if node.id not in inactive_refs:
            self.ref_counts[str(node.id)] += 1

    def visit_Lambda(self, node: itir.Lambda, *, inactive_refs: set[str]):
        inactive_refs = inactive_refs | {param.id for param in node.params}

        self.generic_visit(node, inactive_refs=inactive_refs)


def collect_symbol_refs(
    node: itir.Node | Sequence[itir.Node],
    symbol_names: Optional[Iterable[str]] = None,
    *,
    ignore_builtins: bool = True,
) -> list[str]:
    return [
        symbol_name
        for symbol_name, count in CountSymbolRefs.apply(
            node, symbol_names, ignore_builtins=ignore_builtins
        ).items()
        if count > 0
    ]


def get_user_defined_symbols(symtable: dict[eve.SymbolName, itir.Sym]) -> set[str]:
    return {str(sym) for sym in symtable.keys()} - {
        str(n.id) for n in itir.FencilDefinition._NODE_SYMBOLS_
    }
