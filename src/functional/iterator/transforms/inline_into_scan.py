from collections.abc import Callable
from typing import Optional

from eve import NodeTranslator, traits
import eve
from functional.iterator import ir
from functional.iterator.transforms.inline_lifts import InlineLifts


def _generate_unique_symbol(
    desired_name: Optional[eve.SymbolName | tuple[ir.Lambda | ir.SymRef, int]] = None,
    occupied_names=None,
    occupied_symbols=None,
):
    occupied_names = occupied_names or set()
    occupied_symbols = occupied_symbols or set()
    if not desired_name:
        desired_name = "__sym"
    elif isinstance(desired_name, tuple):
        fun, arg_idx = desired_name
        if isinstance(fun, ir.Lambda):
            desired_name = fun.params[arg_idx].id
        else:
            desired_name = f"__arg{arg_idx}"

    new_symbol = ir.Sym(id=desired_name)
    # make unique
    while new_symbol.id in occupied_names or new_symbol in occupied_symbols:
        new_symbol = ir.Sym(id=new_symbol.id + "_")
    return new_symbol


def _transform_and_extract_lift_args(
    node: ir.FunCall,
    symtable: dict[eve.SymbolName, ir.Sym],
    extracted_args: dict[ir.Sym, ir.Expr],
):
    """
    Transform and extract non-symbol arguments of a lifted stencil call.

    E.g. ``lift(lambda a: ...)(sym1, expr1)`` is transformed into
    ``lift(lambda a: ...)(sym1, sym2)`` with the extracted arguments
    being ``{sym1: sym1, sym2: expr1}``.
    """
    # assert _is_lift(node)
    inner_stencil = node.fun.args[0]
    print("node args")
    print(node.args)
    print("inner stencil")
    print(inner_stencil)

    new_args = set()  # []
    for i, arg in enumerate(node.args):
        if isinstance(arg, ir.SymRef):
            sym = ir.Sym(id=arg.id)
            assert sym not in extracted_args or extracted_args[sym] == arg
            extracted_args[sym] = arg
            new_args.add(arg)
        else:
            assert isinstance(arg, ir.FunCall)
            new_symbol = _generate_unique_symbol(
                desired_name=(inner_stencil, i + 1),
                occupied_names=symtable.keys(),
                occupied_symbols=extracted_args.keys(),
            )
            assert new_symbol not in extracted_args
            extracted_args[new_symbol] = arg
            for inner_arg in arg.args:
                new_args.add(inner_arg)
            # new_args.append(ir.SymRef(id=new_symbol.id))

    acc = inner_stencil.params[0]
    inner_stencil.params.pop(0)
    return (
        ir.FunCall(
            fun=ir.FunCall(
                fun=ir.SymRef(id="scan"),
                args=[
                    ir.Lambda(
                        expr=ir.FunCall(fun=inner_stencil, args=[*extracted_args.values()]),
                        # params=[acc, *extracted_args.keys()],
                        params=[acc, *(ir.Sym(id=str(r.id)) for r in new_args)],
                    ), *node.fun.args[1:]
                ],
            ),
            args=[*new_args],
        ),
        extracted_args,
    )


def _matching_names(node: ir.FunCall):
    for i in len(node.args):
        if str(node.args[i].id) != str(node.fun.params[i]):
            return False
    return True


class InlineIntoScan(traits.VisitorWithSymbolTableTrait, NodeTranslator):
    """ """

    @staticmethod
    def _is_scan(node: ir.Node):
        return (
            isinstance(node, ir.FunCall)
            and isinstance(node.fun, ir.FunCall)
            and node.fun.fun == ir.SymRef(id="scan")
        )

    def visit_FunCall(self, node: ir.FunCall, **kwargs):
        if isinstance(node.fun, ir.Lambda) and _matching_names(node):
            print("matching")
        elif self._is_scan(node):
            extracted_args = {}
            print(node)
            a, b = _transform_and_extract_lift_args(node, kwargs["symtable"], extracted_args)
            # return
            print(a)
            print(b)
            print("scan ")
            return a
        return node


class InlineIntoScan2(NodeTranslator):
    """ """

    @staticmethod
    def _is_scan(node: ir.Node):
        return (
            isinstance(node, ir.FunCall)
            and isinstance(node.fun, ir.FunCall)
            and node.fun.fun == ir.SymRef(id="scan")
        )

    def visit_SymRef(self, node: ir.SymRef, **kwargs):
        if "param_id_map" in kwargs:
            param_id_map = kwargs["param_id_map"]
            if node.id in param_id_map:
                if param_id_map[node.id] in kwargs["to_inline"]:
                    idx = param_id_map[node.id]
                    fun = kwargs["to_inline"][idx]
                    return ir.FunCall(
                        fun=fun,
                        args=[ir.SymRef(id=str(kwargs["params"][kwargs["param_map"][idx]].id))],
                    )
            return node
        return self.generic_visit(node)

    def visit_Lambda(self, node: ir.Lambda, **kwargs):
        if "to_inline" in kwargs:
            params = [node.params[0]]
            param_id_map = {v.id: i for i, v in enumerate(node.params[1:])}
            for k, v in kwargs["param_map"].items():
                if k == v:
                    params.append(node.params[k + 1])
            return ir.Lambda(
                expr=InlineLifts().visit(
                    self.visit(node.expr, params=params[1:], param_id_map=param_id_map, **kwargs)
                ),
                params=params,
            )
        return self.generic_visit(node)

    def visit_FunCall(self, node: ir.FunCall, **kwargs):
        if self._is_scan(node):
            to_inline = {}
            rewritten_args = []
            arg_map = {}
            for i, arg in enumerate(node.args):
                if isinstance(arg, ir.FunCall):
                    to_inline[i] = arg.fun
                    assert len(arg.args) == 1
                    rewritten_arg = arg.args[0]
                else:
                    rewritten_arg = arg
                if rewritten_arg not in rewritten_args:
                    rewritten_args.append(rewritten_arg)
                    arg_map[i] = i
                else:
                    arg_map[i] = rewritten_args.index(rewritten_arg)

            return ir.FunCall(
                fun=self.visit(node.fun, to_inline=to_inline, param_map=arg_map),
                args=rewritten_args,
            )

        return self.generic_visit(node, **kwargs)
