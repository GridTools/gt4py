from devtools import debug

from eve import NodeTranslator, iter_tree
from eve.type_definitions import SymbolName
from functional.iterator import ir as itir
from functional.iterator.backends.gtfn.gtfn_ir import (
    Backend,
    BinaryExpr,
    BoolLiteral,
    Expr,
    FencilDefinition,
    FloatLiteral,
    FunCall,
    FunctionDefinition,
    GridType,
    IntLiteral,
    Lambda,
    OffsetLiteral,
    Program,
    StencilExecution,
    StringLiteral,
    Sym,
    SymRef,
    TemplatedFunCall,
    TernaryExpr,
    UnaryExpr,
)


class GTFN_lowering(NodeTranslator):
    _binary_op_map = {
        "minus": "-",
        "plus": "+",
        "multiplies": "*",
        "divides": "/",
        "and_": "&&",
        "or_": "||",
    }
    _unary_op_map = {"not_": "!"}

    def visit_Sym(self, node: itir.Sym, **kwargs) -> Sym:
        return Sym(id=node.id)

    def visit_SymRef(self, node: itir.SymRef, **kwargs) -> SymRef:
        return SymRef(id=node.id)

    def visit_Lambda(self, node: itir.Lambda, **kwargs) -> Lambda:
        debug(node.expr)
        return Lambda(params=self.visit(node.params), expr=self.visit(node.expr))

    def visit_IntLiteral(self, node: itir.IntLiteral, **kwargs):
        return IntLiteral(value=node.value)

    def visit_StringLiteral(self, node: itir.StringLiteral, **kwargs):
        return StringLiteral(value=node.value)

    def visit_BoolLiteral(self, node: itir.BoolLiteral, **kwargs):
        return BoolLiteral(value=node.value)

    def visit_FloatLiteral(self, node: itir.FloatLiteral, **kwargs):
        return FloatLiteral(value=node.value)

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
        )

    @staticmethod
    def _collect_offsets(node: itir.Program) -> set[str]:
        return (
            iter_tree(node)
            .if_isinstance(itir.OffsetLiteral)
            .getattr("value")
            .if_isinstance(str)
            .to_set()
        )

    def visit_Program(self, node: itir.Program, *, grid_type: str, **kwargs) -> Program:
        grid_type = (
            GridType.Cartesian if grid_type.lower() == "cartesian" else GridType.Unstructured
        )
        return Program(
            function_definitions=self.visit(node.function_definitions),
            fencil_definitions=self.visit(node.fencil_definitions),
            offsets=self._collect_offsets(node),
            grid_type=grid_type,
        )
