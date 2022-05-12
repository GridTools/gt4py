import boltons.typeutils

from eve import NodeVisitor
from functional.iterator import ir


class CollectShifts(NodeVisitor):
    ALL_NEIGHBORS = boltons.typeutils.make_sentinel(name="ALL_NEIGHBORS", var_name="ALL_NEIGHBORS")

    def visit_FunCall(self, node: ir.FunCall, *, shifts: dict[str, list[tuple]]):
        if node.fun == ir.SymRef(id="deref"):
            assert len(node.args) == 1
            arg = node.args[0]
            if isinstance(arg, ir.SymRef):
                # direct deref of a symbol: deref(sym)
                shifts.setdefault(arg.id, []).append(())
                return
            if (
                isinstance(arg, ir.FunCall)
                and isinstance(arg.fun, ir.FunCall)
                and arg.fun.fun == ir.SymRef(id="shift")
                and isinstance(arg.args[0], ir.SymRef)
            ):
                # deref of a shifted symbol: deref(shift(...)(sym))
                return self.visit(arg, shifts=shifts)
            raise ValueError(f"Unexpected node: {node}")
        if isinstance(node.fun, ir.FunCall):
            if node.fun.fun == ir.SymRef(id="shift") and isinstance(node.args[0], ir.SymRef):
                # just shifting: shift(...)(sym)
                assert len(node.args) == 1
                sym = node.args[0]
                shift_args = node.fun.args
                shifts.setdefault(sym.id, []).append(tuple(shift_args))
                return
            if node.fun.fun == ir.SymRef(id="reduce"):
                # neighborhood reduction: reduce(..., ...)(args...)
                assert len(node.fun.args) == 2
                nested_shifts: dict[str, list[tuple]] = dict()
                self.visit(node.args, shifts=nested_shifts)
                for symname, nshifts in nested_shifts.items():
                    for s in nshifts:
                        shifts.setdefault(symname, []).append(s + (self.ALL_NEIGHBORS,))
                return
        if node.fun in (ir.SymRef(id="lift"), ir.SymRef(id="scan")) or not isinstance(
            node.fun, ir.SymRef
        ):
            raise ValueError(f"Unsupported node: {node}")
        self.generic_visit(node, shifts=shifts)
