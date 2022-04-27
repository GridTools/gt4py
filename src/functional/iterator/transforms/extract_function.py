from functional.iterator import ir


def extract_function(node: ir.Node, new_name: str) -> tuple[ir.SymRef, ir.FunctionDefinition]:
    """
    Extract a node into a FunctionDefinition.

    Currently only supports extracting Lambdas, but could be extended to other nodes.
    """
    if not isinstance(node, ir.Lambda):
        raise NotImplementedError(type(node))
    return (
        ir.SymRef(id=new_name),
        ir.FunctionDefinition(id=new_name, params=node.params, expr=node.expr),
    )
