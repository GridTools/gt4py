from typing import Dict, Optional

from eve import NodeTranslator
from functional.iterator import ir
from functional.iterator.transforms.remap_symbols import RemapSymbolRefs


class PopupTmps(NodeTranslator):
    _counter = 0

    def visit_FunCall(self, node: ir.FunCall, *, lifts: Optional[Dict[str, ir.Node]] = None):
        if (
            isinstance(node.fun, ir.FunCall)
            and isinstance(node.fun.fun, ir.SymRef)
            and node.fun.fun.id == "lift"
        ):
            # lifted lambda call
            assert len(node.fun.args) == 1
            assert lifts is not None

            nested_lifts: Dict[str, ir.Node] = dict()
            res = self.generic_visit(node, lifts=nested_lifts)
            lambda_fun = res.fun.args[0]
            if isinstance(lambda_fun, ir.FunCall):
                if isinstance(lambda_fun.fun, ir.SymRef) and lambda_fun.fun.id == "scan":
                    lambda_fun = lambda_fun.args[0]
            assert isinstance(lambda_fun, ir.Lambda)

            symrefs = lambda_fun.iter_tree().if_isinstance(ir.SymRef).getattr("id").to_set()
            captured = symrefs - {p.id for p in lambda_fun.params} - ir.BUILTINS
            if captured:
                lifts |= nested_lifts
                return res

            # TODO: avoid possible symbol name clashes
            symbol = f"t{self._counter}"
            self._counter += 1

            symbol_map = {param.id: arg for param, arg in zip(lambda_fun.params, res.args)}
            new_args = [
                RemapSymbolRefs().visit(arg, symbol_map=symbol_map) for arg in nested_lifts.values()
            ]
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
        elif node.fun == ir.SymRef(id="reduce"):
            # protect reduction function from possible modification
            return node

        return self.generic_visit(node, lifts=lifts)

    def visit_Lambda(self, node: ir.Lambda, *, lifts):
        node = self.generic_visit(node, lifts=lifts)
        if not lifts:
            return node

        new_params = [ir.Sym(id=param) for param in lifts.keys()]
        return ir.Lambda(params=node.params + new_params, expr=node.expr)
