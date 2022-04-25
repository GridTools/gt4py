import eve
from functional.iterator import ir


def add_fundef(root: ir.FencilDefinition, fundef: ir.FunctionDefinition) -> ir.FencilDefinition:
    return ir.FencilDefinition(
        id=root.id,
        function_definitions=[*root.function_definitions, fundef],
        params=root.params,
        closures=root.closures,
    )


def replace_node(root: eve.Node, src: eve.Node, dst: eve.Node) -> eve.Node:
    class ReplaceNode(eve.NodeTranslator):
        def visit(self, node: eve.Node):
            if node is src:
                return dst
            return self.generic_visit(node)

    return ReplaceNode().visit(root)
