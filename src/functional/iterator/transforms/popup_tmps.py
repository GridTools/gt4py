from typing import Optional, Union

from eve import NodeTranslator
from eve.utils import UIDs
from functional.iterator import ir
from functional.iterator.transforms.remap_symbols import RemapSymbolRefs


class PopupTmps(NodeTranslator):
    def visit_FunCall(
        self, node: ir.FunCall, *, lifts: Optional[dict[str, ir.Node]] = None
    ) -> Union[ir.SymRef, ir.FunCall]:
        if isinstance(node.fun, ir.FunCall) and node.fun.fun == ir.SymRef(id="lift"):
            # lifted lambda call
            assert len(node.fun.args) == 1
            assert lifts is not None

            fun = node.fun.args[0]
            if isinstance(fun, ir.FunCall) and fun.fun == ir.SymRef(id="scan"):
                fun = fun.args[0]
                is_scan = True
            else:
                is_scan = False
            assert isinstance(fun, ir.Lambda)
            nested_lifts = dict[str, ir.Node]()
            fun = self.visit(fun, lifts=nested_lifts)
            args = self.visit(node.args, lifts=lifts)

            symrefs = fun.iter_tree().if_isinstance(ir.SymRef).getattr("id").to_set()
            captured = symrefs - {p.id for p in fun.params} - set(nested_lifts) - ir.BUILTINS
            if captured:
                lifts |= nested_lifts
                if is_scan:
                    fun = ir.FunCall(
                        fun=ir.SymRef(id="scan"), args=[fun] + node.fun.args[0].args[1:]
                    )
                return ir.FunCall(fun=ir.FunCall(fun=ir.SymRef(id="lift"), args=[fun]), args=args)

            symbol_map = {param.id: arg for param, arg in zip(fun.params, args)}

            for k, v in nested_lifts.items():
                nested_lifts[k] = RemapSymbolRefs().visit(v, symbol_map=symbol_map)
            for k, v in lifts.items():
                lifts[k] = RemapSymbolRefs().visit(v, symbol_map=symbol_map)

            new_args = list(nested_lifts.values())
            fun = ir.Lambda(params=fun.params + [ir.Sym(id=p) for p in nested_lifts], expr=fun.expr)
            if is_scan:
                fun = ir.FunCall(fun=ir.SymRef(id="scan"), args=[fun] + node.fun.args[0].args[1:])
            call = ir.FunCall(
                fun=ir.FunCall(fun=ir.SymRef(id="lift"), args=[fun]), args=args + new_args
            )

            for k, v in lifts.items():
                if call == v:
                    return ir.SymRef(id=k)

            symbol = UIDs.sequential_id(prefix="_lift")
            lifts[symbol] = call
            return ir.SymRef(id=symbol)
        if isinstance(node.fun, ir.Lambda):
            fun = node.fun
            nested_lifts = dict()
            fun = self.visit(fun, lifts=nested_lifts)
            args = self.visit(node.args, lifts=lifts)

            symbol_map = {param.id: arg for param, arg in zip(fun.params, args)}

            for k, v in nested_lifts.items():
                nested_lifts[k] = RemapSymbolRefs().visit(v, symbol_map=symbol_map)
            if lifts is not None:
                for k, v in lifts.items():
                    lifts[k] = RemapSymbolRefs().visit(v, symbol_map=symbol_map)

            new_args = list(nested_lifts.values())
            fun = ir.Lambda(params=fun.params + [ir.Sym(id=p) for p in nested_lifts], expr=fun.expr)
            call = ir.FunCall(fun=fun, args=args + new_args)
            return call
        return self.generic_visit(node, lifts=lifts)
