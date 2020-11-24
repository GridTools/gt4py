from types import MappingProxyType
from typing import ClassVar, Dict, List, Mapping, Union

from gt4py.gtc import common, gtir
from gt4py.ir import IRNodeVisitor
from gt4py.ir.nodes import (
    ArgumentInfo,
    Assign,
    AxisBound,
    AxisInterval,
    BinaryOperator,
    BinOpExpr,
    BlockStmt,
    ComputationBlock,
    FieldDecl,
    FieldRef,
    If,
    IterationOrder,
    LevelMarker,
    ScalarLiteral,
    StencilDefinition,
    VarDecl,
    VarRef,
)


def transform_offset(offset: Dict[str, int]) -> gtir.CartesianOffset:
    i = offset["I"] if "I" in offset else 0
    j = offset["J"] if "J" in offset else 0
    k = offset["K"] if "K" in offset else 0
    return gtir.CartesianOffset(i=i, j=j, k=k)


class DefIRToGTIR(IRNodeVisitor):

    GT4PY_ITERATIONORDER_TO_GTIR_LOOPORDER: ClassVar[
        Mapping[IterationOrder, int]
    ] = MappingProxyType(
        {
            IterationOrder.BACKWARD: common.LoopOrder.BACKWARD,
            IterationOrder.PARALLEL: common.LoopOrder.PARALLEL,
            IterationOrder.FORWARD: common.LoopOrder.FORWARD,
        }
    )

    GT4PY_LEVELMARKER_TO_GTIR_LEVELMARKER: ClassVar[Mapping[LevelMarker, str]] = MappingProxyType(
        {LevelMarker.START: common.LevelMarker.START, LevelMarker.END: common.LevelMarker.END}
    )

    GT4PY_OP_TO_GTIR_OP: ClassVar[
        Mapping[
            BinaryOperator,
            Union[common.ArithmeticOperator, common.LogicalOperator, common.ComparisonOperator],
        ]
    ] = MappingProxyType(
        {
            # arithmetic
            BinaryOperator.ADD: common.ArithmeticOperator.ADD,
            BinaryOperator.SUB: common.ArithmeticOperator.SUB,
            BinaryOperator.MUL: common.ArithmeticOperator.MUL,
            BinaryOperator.DIV: common.ArithmeticOperator.DIV,
            # logical
            BinaryOperator.AND: common.LogicalOperator.AND,
            BinaryOperator.OR: common.LogicalOperator.OR,
            # comparison
            BinaryOperator.EQ: common.ComparisonOperator.EQ,
            BinaryOperator.NE: common.ComparisonOperator.NE,
            BinaryOperator.LT: common.ComparisonOperator.LT,
            BinaryOperator.LE: common.ComparisonOperator.LE,
            BinaryOperator.GT: common.ComparisonOperator.GT,
            BinaryOperator.GE: common.ComparisonOperator.GE,
        }
    )

    @classmethod
    def apply(cls, root, **kwargs):
        return cls().visit(root)

    def visit_StencilDefinition(self, node: StencilDefinition) -> gtir.Stencil:
        vertical_loops = [self.visit(c) for c in node.computations]
        field_params = {f.name: self.visit(f) for f in node.api_fields}
        scalar_params = {p.name: self.visit(p) for p in node.parameters}
        return gtir.Stencil(
            name=node.name,
            params=[
                self.visit(f, all_params={**field_params, **scalar_params})
                for f in node.api_signature
            ],
            vertical_loops=vertical_loops,
        )

    def visit_ArgumentInfo(
        self, node: ArgumentInfo, all_params: Dict[str, Union[gtir.Decl]]
    ) -> Union[gtir.Decl]:
        return all_params[node.name]

    def visit_ComputationBlock(self, node: ComputationBlock) -> List[gtir.VerticalLoop]:
        stmts = [s for s in self.visit(node.body)]
        start, end = self.visit(node.interval)
        interval = gtir.Interval(start=start, end=end)
        return gtir.VerticalLoop(
            interval=interval,
            loop_order=self.GT4PY_ITERATIONORDER_TO_GTIR_LOOPORDER[node.iteration_order],
            body=stmts,
        )

    def visit_BlockStmt(self, node: BlockStmt) -> List[gtir.Stmt]:
        return [self.visit(s) for s in node.stmts]

    def visit_Assign(self, node: Assign) -> gtir.ParAssignStmt:
        assert isinstance(node.target, FieldRef)
        left = self.visit(node.target)
        return gtir.ParAssignStmt(left=left, right=self.visit(node.value))

    def visit_ScalarLiteral(self, node: ScalarLiteral) -> gtir.Literal:
        return gtir.Literal(value=str(node.value), dtype=common.DataType(node.data_type.value))

    def visit_BinOpExpr(self, node: BinOpExpr) -> gtir.BinaryOp:
        return gtir.BinaryOp(
            left=self.visit(node.lhs),
            right=self.visit(node.rhs),
            op=self.GT4PY_OP_TO_GTIR_OP[node.op],
        )

    def visit_FieldRef(self, node: FieldRef):
        return gtir.FieldAccess(name=node.name, offset=transform_offset(node.offset))

    def visit_If(self, node: If):
        return gtir.FieldIfStmt(
            cond=self.visit(node.condition),
            true_branch=gtir.BlockStmt(body=self.visit(node.main_body)),
            false_branch=gtir.BlockStmt(body=self.visit(node.else_body)),
        )

    def visit_VarRef(self, node: VarRef):
        return gtir.ScalarAccess(
            name=node.name,
            # TODO index
        )

    def visit_AxisInterval(self, node: AxisInterval):
        return self.visit(node.start), self.visit(node.end)

    def visit_AxisBound(self, node: AxisBound):
        # TODO support VarRef
        return gtir.AxisBound(
            level=self.GT4PY_LEVELMARKER_TO_GTIR_LEVELMARKER[node.level], offset=node.offset
        )

    def visit_FieldDecl(self, node: FieldDecl):
        # datatype conversion works via same ID
        return gtir.FieldDecl(name=node.name, dtype=common.DataType(int(node.data_type.value)))

    def visit_VarDecl(self, node: VarDecl):
        # datatype conversion works via same ID
        return gtir.ScalarDecl(name=node.name, dtype=common.DataType(int(node.data_type.value)))
