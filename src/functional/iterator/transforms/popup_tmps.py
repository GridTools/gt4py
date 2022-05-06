from collections.abc import Callable
from functools import partial
from typing import Optional, Union

from eve import NodeTranslator
from eve.utils import UIDs
from functional.iterator import ir
from functional.iterator.transforms.remap_symbols import RemapSymbolRefs


class PopupTmps(NodeTranslator):
    @staticmethod
    def _extract_lambda(
        node: ir.FunCall,
    ) -> Optional[tuple[ir.Lambda, bool, Callable[[ir.Lambda, list[ir.Node]], ir.FunCall]]]:
        """Extract the lambda function which is relevant for popping up lifts.

        Further, returns a bool indicating if the given function call was as a
        lift expression and a wrapper function that undos the extraction.

        So The behavior is the following:
        - For `lift(f)(args...)` it returns `(f, True, wrap)`.
        - For `lift(scan(f, dir, init))(args...)` it returns `(f, True, wrap)`.
        - For `f(args...)` it returns `(f, False, wrap)`.
        - For any other expression, it returns `None`.

        The returned `wrap` function undos the extraction in all cases; for example,
        `wrap(f, args...)` returns `lift(f)(args...)` in the first case.
        """
        if isinstance(node.fun, ir.FunCall) and node.fun.fun == ir.SymRef(id="lift"):
            # lifted lambda call or lifted scan
            assert len(node.fun.args) == 1
            fun = node.fun.args[0]

            is_scan = isinstance(fun, ir.FunCall) and fun.fun == ir.SymRef(id="scan")
            if is_scan:
                fun = fun.args[0]

            def wrap(fun: ir.Lambda, args: list[ir.Node]) -> ir.FunCall:
                if is_scan:
                    fun = ir.FunCall(
                        fun=ir.SymRef(id="scan"), args=[fun] + node.fun.args[0].args[1:]
                    )
                return ir.FunCall(fun=ir.FunCall(fun=ir.SymRef(id="lift"), args=[fun]), args=args)

            return fun, True, wrap
        if isinstance(node.fun, ir.Lambda):
            # direct lambda call

            def wrap(fun: ir.Lambda, args: list[ir.Node]) -> ir.FunCall:
                return ir.FunCall(fun=fun, args=args)

            return node.fun, False, wrap

        return None

    def visit_FunCall(
        self, node: ir.FunCall, *, lifts: Optional[dict[ir.Expr, ir.SymRef]] = None
    ) -> Union[ir.SymRef, ir.FunCall]:
        if call_info := self._extract_lambda(node):
            fun, is_lift, wrap = call_info

            nested_lifts = dict[ir.Expr, ir.SymRef]()
            fun = self.visit(fun, lifts=nested_lifts)
            # Note: lifts in arguments are just passed to the parent node
            args = self.visit(node.args, lifts=lifts)

            if is_lift:
                assert lifts is not None

                # check if the lifted expression captures symbols from the outer scope
                symrefs = fun.iter_tree().if_isinstance(ir.SymRef).getattr("id").to_set()
                captured = (
                    symrefs
                    - {p.id for p in fun.params}
                    - {n.id for n in nested_lifts.values()}
                    - ir.BUILTINS
                )
                if captured:
                    # if symbols from an outer scope are captured, the lift has to
                    # be handled at that scope, so skip here and pass nested lifts on
                    lifts |= nested_lifts
                    return wrap(fun, args)

            # remap referenced function parameters in lift expression to passed argument values
            symbol_map = {param.id: arg for param, arg in zip(fun.params, args)}
            remap = partial(RemapSymbolRefs().visit, symbol_map=symbol_map)

            nested_lifts = {remap(expr): ref for expr, ref in nested_lifts.items()}
            if lifts:
                # lifts have to be updated in place as they are passed to parent node
                lifted = list(lifts.items())
                lifts.clear()
                for expr, ref in lifted:
                    lifts[remap(expr)] = ref

            # extend parameter and argument list of the function with popped lifts
            new_params = [ir.Sym(id=p.id) for p in nested_lifts.values()]
            new_args = list(nested_lifts.keys())
            fun = ir.Lambda(params=fun.params + new_params, expr=fun.expr)

            # updated function call, having lifts passed as arguments
            call = wrap(fun, args + new_args)

            if not is_lift:
                # if this is not a lift expression, we are done...
                return call

            # ... otherwise we check if the same expression has already been
            # lifted before, then we reference that one
            assert lifts is not None
            if (previous_ref := lifts.get(call)) is not None:
                return previous_ref

            # if this is the first time we lift that expression, create a new
            # symbol for it and register it so the parent node knows about it
            ref = ir.SymRef(id=UIDs.sequential_id(prefix="_lift"))
            lifts[call] = ref
            return ref
        return self.generic_visit(node, lifts=lifts)
