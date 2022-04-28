import eve
from functional.iterator import ir


def add_fundefs(
    root: ir.FencilDefinition, fundefs: list[ir.FunctionDefinition]
) -> ir.FencilDefinition:
    return ir.FencilDefinition(
        id=root.id,
        function_definitions=root.function_definitions + fundefs,
        params=root.params,
        closures=root.closures,
    )


def replace_nodes(root: eve.concepts.AnyNode, idmap: dict[int, eve.Node]) -> eve.concepts.AnyNode:
    class ReplaceNode(eve.NodeTranslator):
        def visit_Node(self, node: eve.Node) -> eve.Node:
            if id(node) in idmap:
                return idmap[id(node)]
            return self.generic_visit(node)

    return ReplaceNode().visit(root)
