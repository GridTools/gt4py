# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from collections import Counter

import gt4py.eve as eve
from gt4py.eve.extended_typing import Iterable, Literal, Optional, Sequence, cast, overload
from gt4py.next.iterator import ir as itir


@dataclasses.dataclass
class CountSymbolRefs(eve.PreserveLocationVisitor, eve.NodeVisitor):
    ref_counts: Counter[itir.SymRef] = dataclasses.field(default_factory=Counter)

    @overload
    @classmethod
    def apply(
        cls,
        node: itir.Node | Sequence[itir.Node],
        symbol_names: Optional[Iterable[str]] = None,
        *,
        ignore_builtins: bool = True,
        as_ref: Literal[False] = False,
    ) -> Counter[str]: ...

    @overload
    @classmethod
    def apply(
        cls,
        node: itir.Node | Sequence[itir.Node],
        symbol_names: Optional[Iterable[str]] = None,
        *,
        ignore_builtins: bool = True,
        as_ref: Literal[True],
    ) -> Counter[itir.SymRef]: ...

    @classmethod
    def apply(
        cls,
        node: itir.Node | Sequence[itir.Node],
        symbol_names: Optional[Iterable[str]] = None,
        *,
        ignore_builtins: bool = True,
        as_ref: bool = False,
    ) -> Counter[str] | Counter[itir.SymRef]:
        """
        Count references to given or all symbols in scope.

        Examples:
            >>> import gt4py.next.iterator.ir_utils.ir_makers as im
            >>> expr = im.plus(im.plus("x", "y"), im.plus(im.plus("x", "y"), "z"))
            >>> CountSymbolRefs.apply(expr)
            Counter({'x': 2, 'y': 2, 'z': 1})

            If only some symbols are of interests the search can be restricted:

            >>> CountSymbolRefs.apply(expr, symbol_names=["x", "z"])
            Counter({'x': 2, 'z': 1})

            In some cases, e.g. when the type of the reference is required, the references instead
            of strings can be retrieved.
            >>> CountSymbolRefs.apply(expr, as_ref=True)
            Counter({SymRef(id=SymbolRef('x')): 2, SymRef(id=SymbolRef('y')): 2, SymRef(id=SymbolRef('z')): 1})
        """
        if ignore_builtins:
            inactive_refs = {str(n.id) for n in itir.Program._NODE_SYMBOLS_}
        else:
            inactive_refs = set()

        obj = cls()
        obj.visit(node, inactive_refs=inactive_refs)

        if symbol_names:
            ref_counts = Counter({k: v for k, v in obj.ref_counts.items() if k.id in symbol_names})
        else:
            ref_counts = obj.ref_counts

        result: Counter[str] | Counter[itir.SymRef]
        if as_ref:
            result = ref_counts
        else:
            result = Counter({str(k.id): v for k, v in ref_counts.items()})

        return result

    def visit_SymRef(self, node: itir.SymRef, *, inactive_refs: set[str]):
        if node.id not in inactive_refs:
            self.ref_counts[node] += 1

    def visit_Lambda(self, node: itir.Lambda, *, inactive_refs: set[str]):
        inactive_refs = inactive_refs | {param.id for param in node.params}

        self.generic_visit(node, inactive_refs=inactive_refs)


@overload
def collect_symbol_refs(
    node: itir.Node | Sequence[itir.Node],
    symbol_names: Optional[Iterable[str]] = None,
    *,
    ignore_builtins: bool = True,
    as_ref: Literal[False] = False,
) -> list[str]: ...


@overload
def collect_symbol_refs(
    node: itir.Node | Sequence[itir.Node],
    symbol_names: Optional[Iterable[str]] = None,
    *,
    ignore_builtins: bool = True,
    as_ref: Literal[True],
) -> list[itir.SymRef]: ...


def collect_symbol_refs(
    node: itir.Node | Sequence[itir.Node],
    symbol_names: Optional[Iterable[str]] = None,
    *,
    ignore_builtins: bool = True,
    as_ref: bool = False,
):
    assert as_ref in [True, False]
    return [
        symbol_name
        for symbol_name, count in CountSymbolRefs.apply(
            node,
            symbol_names,
            ignore_builtins=ignore_builtins,
            as_ref=cast(Literal[True, False], as_ref),
        ).items()
        if count > 0
    ]


def get_user_defined_symbols(symtable: dict[eve.SymbolName, itir.Sym]) -> set[str]:
    return {str(sym) for sym in symtable.keys()} - {str(n.id) for n in itir.Program._NODE_SYMBOLS_}
