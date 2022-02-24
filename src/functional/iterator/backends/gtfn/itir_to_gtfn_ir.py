from devtools import debug

from eve import NodeTranslator
from eve.type_definitions import SymbolName
from functional.iterator import ir as itir
from functional.iterator.backends.gtfn.gtfn_ir import (
    Backend,
    BinaryExpr,
    Expr,
    FencilDefinition,
    FunCall,
    FunctionDefinition,
    Lambda,
    OffsetLiteral,
    Program,
    StencilExecution,
    Sym,
    SymRef,
    UnaryExpr,
)


class GTFN_lowering(NodeTranslator):
    _binary_op_map = {"minus": "-", "plus": "+", "multiplies": "*", "divides": "/"}
    _unary_op_map = {"not": "!"}

    def visit_Sym(self, node: itir.Sym, **kwargs) -> Sym:
        return Sym(id=node.id)

    def visit_SymRef(self, node: itir.SymRef, **kwargs) -> SymRef:
        return SymRef(id=node.id)

    def visit_Lambda(self, node: itir.Lambda, **kwargs) -> Lambda:
        debug(node.expr)
        return Lambda(params=self.visit(node.params), expr=self.visit(node.expr))

    def visit_OffsetLiteral(self, node: itir.OffsetLiteral, **kwargs) -> OffsetLiteral:
        return OffsetLiteral(value=node.value)

    def visit_FunCall(self, node: itir.FunCall, **kwargs) -> Expr:
        if isinstance(node.fun, itir.SymRef):
            if node.fun.id in self._unary_op_map:
                assert len(node.args) == 1
                return UnaryExpr(op=self._unary_op_map[node.fun.id], expr=self.visit(node.args[0]))
            elif node.fun.id in self._binary_op_map:
                assert len(node.args) == 2
                return BinaryExpr(
                    op=self._binary_op_map[node.fun.id],
                    lhs=self.visit(node.args[0]),
                    rhs=self.visit(node.args[1]),
                )
        elif (
            isinstance(node.fun, itir.FunCall)
            and isinstance(node.fun.fun, itir.SymRef)
            and node.fun.fun.id == "shift"
        ):
            assert len(node.args) == 1
            return FunCall(
                fun=self.visit(node.fun.fun), args=self.visit(node.args) + self.visit(node.fun.args)
            )
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
