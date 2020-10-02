from gt4py.ir import IRNodeVisitor
from gt4py.ir.nodes import StencilDefinition


class DefIRToGTIR(IRNodeVisitor):
    def visit_StencilDefinition(self, node: StencilDefinition):
        pass
