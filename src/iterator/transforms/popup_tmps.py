from typing import Dict, Optional

from eve import NodeTranslator
from iterator import ir
from iterator.transforms.remap_symbols import RemapSymbolRefs


class PopupTmps(NodeTranslator):
    _counter = 0

    def visit_FunCall(self, node: ir.FunCall, *, lifts: Optional[Dict[str, ir.Node]] = None):
        if (
            isinstance(node.fun, ir.FunCall)
            and isinstance(node.fun.fun, ir.SymRef)
            and node.fun.fun.id == "lift"
        ):
            # lifted lambda call
            assert len(node.fun.args) == 1 and isinstance(node.fun.args[0], ir.Lambda)
            assert lifts is not None

            nested_lifts: Dict[str, ir.Node] = dict()
            res = self.generic_visit(node, lifts=nested_lifts)
            # TODO: avoid possible symbol name clashes
            symbol = f"t{self._counter}"
            self._counter += 1

            symbol_map = {param.id: arg for param, arg in zip(res.fun.args[0].params, res.args)}
            new_args = [
                RemapSymbolRefs().visit(arg, symbol_map=symbol_map) for arg in nested_lifts.values()
            ]
            assert len(res.fun.args[0].params) == len(res.args + new_args)
            call = ir.FunCall(fun=res.fun, args=res.args + new_args)

            # return existing definition if the same expression was lifted before
            for k, v in lifts.items():
                if call == v:
                    return ir.SymRef(id=k)

            lifts[symbol] = call
            return ir.SymRef(id=symbol)
        elif isinstance(node.fun, ir.Lambda):
            # direct lambda call
            lifts = dict()
            res = self.generic_visit(node, lifts=lifts)
            symbol_map = {param.id: arg for param, arg in zip(res.fun.params, res.args)}
            new_args = [
                RemapSymbolRefs().visit(arg, symbol_map=symbol_map) for arg in lifts.values()
            ]
            assert len(res.fun.params) == len(res.args + new_args)
            return ir.FunCall(fun=res.fun, args=res.args + new_args)

        return self.generic_visit(node, lifts=lifts)

    def visit_Lambda(self, node: ir.Lambda, *, lifts):
        node = self.generic_visit(node, lifts=lifts)
        if not lifts:
            return node

        new_params = [ir.Sym(id=param) for param in lifts.keys()]
        return ir.Lambda(params=node.params + new_params, expr=node.expr)
