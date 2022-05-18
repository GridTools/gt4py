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

    def visit_AxisLiteral(self, node: itir.AxisLiteral, **kwargs: Any) -> Literal:
        return Literal(
            value="NOT_SUPPORTED", type="axis_literal"
        )  # TODO(havogt) decide if domain is part of the IR

    @staticmethod
    def _is_sparse_deref_shift(node: itir.FunCall) -> bool:
        return (
            node.fun == itir.SymRef(id="deref")
            and isinstance(node.args[0], itir.FunCall)
            and isinstance(node.args[0].fun, itir.FunCall)
            and node.args[0].fun.fun == itir.SymRef(id="shift")
            and bool(len(node.args[0].fun.args) % 2)
        )

    def _sparse_deref_shift_to_tuple_get(self, node: itir.FunCall) -> itir.FunCall:
        # deref(shift(i)(sparse)) -> tuple_get(i, deref(sparse))
        # TODO: remove once ‘real’ sparse field handling is available
        offsets = node.args[0].fun.args
        deref_arg = node.args[0].args[0]
        if len(offsets) > 1:
            deref_arg = itir.FunCall(
                fun=itir.FunCall(fun=itir.SymRef(id="shift"), args=offsets[:-1]),
                args=[deref_arg],
            )
        derefed = itir.FunCall(fun=itir.SymRef(id="deref"), args=[deref_arg])
        sparse_access = itir.FunCall(fun=itir.SymRef(id="tuple_get"), args=[offsets[-1], derefed])
        return self.visit(sparse_access)

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
            elif self._is_sparse_deref_shift(node):
                return self._sparse_deref_shift_to_tuple_get(node)
        elif isinstance(node.fun, itir.FunCall) and node.fun.fun == itir.SymRef(id="shift"):
            assert len(node.args) == 1
            return FunCall(
                fun=self.visit(node.fun.fun), args=self.visit(node.args) + self.visit(node.fun.args)
            )
        elif isinstance(node.fun, itir.FunCall) and node.fun == itir.SymRef(id="shift"):
            raise ValueError("unapplied shift call not supported: {node}")
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
            offset_declarations=list(self._collect_offsets(node)),
            function_definitions=self.visit(node.function_definitions),
        )
