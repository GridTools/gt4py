from typing import Dict, Any
import eve

from ..gtscript_ast import GTScriptASTNode, SymbolRef, External, Constant, Subscript

class ConstExprEvaluator(eve.NodeVisitor):
    @classmethod
    def apply(cls, symtable, values):
        instance = cls()
        return instance.visit(values, symtable=symtable)

    def visit_Node(self, node, **kwargs):
        raise ValueError("Evaluation failed.")

    def visit_SymbolRef(self, node: SymbolRef, *, symtable, **kwargs):
        return self.visit(symtable[node.name])

    def visit_External(self, node: External, **kwargs):
        return node.value

    def visit_Constant(self, node: Constant, **kwargs):
        return node.value

    def visit_Subscript(self, node: Subscript, **kwargs):
        return self.visit(node.value, **kwargs)[tuple(self.visit(idx, **kwargs) for idx in node.indices)]

def evaluate_const_expr(symtable: Dict[Any, GTScriptASTNode], node: GTScriptASTNode):
    return ConstExprEvaluator.apply(symtable, node)