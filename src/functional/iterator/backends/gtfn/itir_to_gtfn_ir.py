from typing import Any

from eve import NodeTranslator, iter_tree
from eve.type_definitions import SymbolName
from functional.iterator import ir as itir
from functional.iterator.backends.gtfn.gtfn_ir import (
    Backend,
    BinaryExpr,
    Expr,
    FencilDefinition,
    FunCall,
    FunctionDefinition,
    GridType,
    Lambda,
    Literal,
    OffsetLiteral,
    StencilExecution,
    Sym,
    SymRef,
    TemplatedFunCall,
    TernaryExpr,
    UnaryExpr,
)


class GTFN_lowering(NodeTranslator):
    _binary_op_map = {
        "plus": "+",
        "minus": "-",
        "multiplies": "*",
        "divides": "/",
        "eq": "==",
        "less": "<",
        "greater": ">",
        "and_": "&&",
        "or_": "||",
    }
    _unary_op_map = {"not_": "!"}

    def visit_Sym(self, node: itir.Sym, **kwargs: Any) -> Sym:
        return Sym(id=node.id)

    def visit_SymRef(self, node: itir.SymRef, **kwargs: Any) -> SymRef:
        return SymRef(id=node.id)

    def visit_Lambda(self, node: itir.Lambda, **kwargs: Any) -> Lambda:
        return Lambda(params=self.visit(node.params), expr=self.visit(node.expr))

    def visit_Literal(self, node: itir.Literal, **kwargs: Any) -> Literal:
        return Literal(value=node.value, type=node.type)

    def visit_OffsetLiteral(self, node: itir.OffsetLiteral, **kwargs: Any) -> OffsetLiteral:
        return OffsetLiteral(value=node.value)

    def visit_FunCall(self, node: itir.FunCall, **kwargs: Any) -> Expr:
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
            elif node.fun.id == "if_":
                assert len(node.args) == 3
                return TernaryExpr(
                    cond=self.visit(node.args[0]),
                    true_expr=self.visit(node.args[1]),
                    false_expr=self.visit(node.args[2]),
                )
            elif node.fun.id == "make_tuple":
                return FunCall(fun=SymRef(id="tuple"), args=self.visit(node.args))
            elif node.fun.id == "tuple_get":
                return TemplatedFunCall(
                    fun=SymRef(id="get"),
                    template_args=[self.visit(node.args[0])],
                    args=self.visit(node.args[1:]),
                )
        elif isinstance(node.fun, itir.FunCall) and node.fun.fun == itir.SymRef(id="shift"):
            assert len(node.args) == 1
            return FunCall(
                fun=self.visit(node.fun.fun), args=self.visit(node.args) + self.visit(node.fun.args)
            )
        return FunCall(fun=self.visit(node.fun), args=self.visit(node.args))

    def visit_FunctionDefinition(
        self, node: itir.FunctionDefinition, **kwargs: Any
    ) -> FunctionDefinition:
        return FunctionDefinition(
            id=node.id, params=self.visit(node.params), expr=self.visit(node.expr)
        )

    def visit_StencilClosure(self, node: itir.StencilClosure, **kwargs: Any) -> StencilExecution:
        backend = Backend(domain=self.visit(node.domain))
        return StencilExecution(
            stencil=self.visit(node.stencil),
            output=self.visit(node.output),
            inputs=self.visit(node.inputs),
            backend=backend,
        )

    @staticmethod
    def _collect_offsets(node: itir.FencilDefinition) -> set[str]:
        return (
            iter_tree(node)
            .if_isinstance(itir.OffsetLiteral)
            .getattr("value")
            .if_isinstance(str)
            .to_set()
        )

    def visit_FencilDefinition(
        self, node: itir.FencilDefinition, *, grid_type: str, **kwargs: Any
    ) -> FencilDefinition:
        grid_type = getattr(GridType, grid_type.upper())
        return FencilDefinition(
            id=SymbolName(node.id),
            params=self.visit(node.params),
            executions=self.visit(node.closures),
            grid_type=grid_type,
            offset_declarations=self._collect_offsets(node),
            function_definitions=self.visit(node.function_definitions),
        )
