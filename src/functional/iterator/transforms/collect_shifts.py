from typing import Dict, List

from eve import NodeVisitor
from functional.iterator import ir


class CollectShifts(NodeVisitor):
    def visit_FunCall(self, node: ir.FunCall, *, shifts: Dict[str, List[tuple]]):
        if isinstance(node.fun, ir.SymRef) and node.fun.id == "deref":
            assert len(node.args) == 1
            arg = node.args[0]
            if isinstance(arg, ir.SymRef):
                # direct deref of a symbol: deref(sym)
                shifts.setdefault(arg.id, []).append(())
            elif (
                isinstance(arg, ir.FunCall)
                and isinstance(arg.fun, ir.FunCall)
                and isinstance(arg.fun.fun, ir.SymRef)
                and arg.fun.fun.id == "shift"
                and isinstance(arg.args[0], ir.SymRef)
            ):
                # deref of a shifted symbol: deref(shift(...)(sym))
                assert len(arg.args) == 1
                sym = arg.args[0]
                shift_args = arg.fun.args
                shifts.setdefault(sym.id, []).append(tuple(shift_args))
            else:
                raise RuntimeError(f"Unexpected node: {node}")
        elif isinstance(node.fun, ir.SymRef) and node.fun.id in ("lift", "scan"):
            raise RuntimeError(f"Unsupported node: {node}")
        return self.generic_visit(node, shifts=shifts)
