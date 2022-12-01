import dataclasses

import eve
from functional.iterator import ir as itir


@dataclasses.dataclass
class CountSymbolRefs(eve.NodeVisitor):
    ref_counts: dict[str, int]

    @classmethod
    def apply(cls, node: itir.Node, symbol_names: list[str]) -> dict[str, int]:
        ref_counts = {name: 0 for name in symbol_names}
        active_refs = set(symbol_names)

        obj = cls(ref_counts=ref_counts)
        obj.visit(node, active_refs=active_refs)

        return ref_counts

    def visit_SymRef(self, node: itir.Node, *, active_refs: set[str]):
        if node.id in active_refs:
            self.ref_counts[node.id] += 1

    def visit_Lambda(self, node: itir.Lambda, *, active_refs: set[str]):
        active_refs = active_refs - set(param.id for param in node.params)

        self.generic_visit(node, active_refs=active_refs)
