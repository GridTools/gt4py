from devtools import debug

from eve import NodeTranslator
from eve.type_definitions import SymbolName
from functional.iterator import ir as itir
from functional.iterator.backends.gtfn.gtfn_ir import (
    Backend,
    Expr,
    FencilDefinition,
    FunCall,
    FunctionDefinition,
    Program,
    StencilExecution,
    Sym,
    SymRef,
    UnaryExpr,
)


class GTFN_lowering(NodeTranslator):
    _binary_op_map = {}
    _unary_op_map = {}

    def visit_Sym(self, node: itir.Sym, **kwargs) -> Sym:
        return Sym(id=node.id)

    def visit_SymRef(self, node: itir.SymRef, **kwargs) -> SymRef:
        return SymRef(id=node.id)

    def visit_FunCall(self, node: itir.FunCall, **kwargs) -> Expr:
        if isinstance(node.fun, itir.SymRef):
            if node.fun.id in self._unary_op_map:
                assert len(node.args) == 1
                return UnaryExpr(op=self._unary_op_map[node.fun.id], expr=self.visit(node.args[0]))
            else:
                return FunCall(fun=self.visit(node.fun), args=self.visit(node.args))

    def visit_FunctionDefinition(
        self, node: itir.FunctionDefinition, **kwargs
    ) -> FunctionDefinition:
        return FunctionDefinition(
            id=SymbolName(node.id), params=self.visit(node.params), expr=self.visit(node.expr)
        )

    def visit_StencilClosure(self, node: itir.StencilClosure, **kwargs) -> StencilExecution:
        backend = Backend(domain=self.visit(node.domain), backend_tag="backend::naive{}")  # TODO
        debug(self.visit(node.stencil))
        return StencilExecution(
            backend=backend,
            stencil=self.visit(node.stencil),
            output=self.visit(node.outputs[0]),
            inputs=self.visit(node.inputs),
        )

    def visit_FencilDefinition(self, node: itir.FencilDefinition, **kwargs) -> FencilDefinition:
        return FencilDefinition(
            id=SymbolName(node.id),
            params=self.visit(node.params),
            executions=self.visit(node.closures),
        )  # TODO

    def visit_Program(self, node: itir.Program, **kwargs) -> Program:
        return Program(
            function_definitions=self.visit(node.function_definitions),
            fencil_definitions=self.visit(node.fencil_definitions),
        )
