from typing import Tuple

from functional.iterator import ir


def extract_function(node: ir.Node, new_name: str) -> Tuple[ir.SymRef, ir.FunctionDefinition]:
    if not isinstance(node, ir.Lambda):
        raise NotImplementedError(type(node))
    return (
        ir.SymRef(id=new_name),
        ir.FunctionDefinition(id=new_name, params=node.params, expr=node.expr),
    )
