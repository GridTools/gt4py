import dataclasses
from typing import Iterable, Sequence

import eve
from functional.iterator import ir as itir


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
