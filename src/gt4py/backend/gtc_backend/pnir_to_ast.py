import ast
from typing import List, Union

from eve.visitors import NodeVisitor

from . import common, gtir, pnir


def offset_to_index_item(name: str, offset: int):
    if offset == 0:
        return ast.Name(id=name.upper())
    elif offset > 0:
        return ast.BinOp(left=ast.Name(id=name.upper()), right=ast.Num(n=offset), op=ast.Add())
    return ast.BinOp(left=ast.Name(id=name.upper()), right=ast.Num(n=abs(offset)), op=ast.Sub())


class PnirToAst(NodeVisitor):
    GTIR_OP_TO_AST_OP = {
        common.BinaryOperator.ADD: ast.Add,
        common.BinaryOperator.SUB: ast.Sub,
        common.BinaryOperator.MUL: ast.Mult,
        common.BinaryOperator.DIV: ast.Div,
    }

    def visit_AxisBound(self, node: gtir.AxisBound) -> Union[ast.Num, ast.Subscript, ast.BinOp]:
        if node.level == common.LevelMarker.START:
            return ast.Num(node.offset)
        subscript = ast.Subscript(value=ast.Name(id="_domain_"), slice=ast.Index(value=ast.Num(2)))
        if node.offset == 0:
            return subscript
        return ast.BinOp(left=subscript, op=ast.Sub(), right=ast.Num(n=node.offset))

    def visit_IJLoop(self, node: pnir.IJLoop) -> ast.For:
        return ast.For(
            target=ast.Name(id="J"),
            iter=ast.Call(
                func=ast.Name(id="range"),
                args=[
                    ast.Subscript(
                        value=ast.Name(id="_domain_"),
                        slice=ast.Index(value=ast.Num(n=1)),
                    )
                ],
                keywords=[],
            ),
            orelse=[],
            body=[
                ast.For(
                    target=ast.Name(id="I"),
                    iter=ast.Call(
                        func=ast.Name(id="range"),
                        args=[
                            ast.Subscript(
                                value=ast.Name(id="_domain_"),
                                slice=ast.Index(value=ast.Num(n=0)),
                            )
                        ],
                        keywords=[],
                    ),
                    orelse=[],
                    body=[self.visit(stmt) for stmt in node.body],
                )
            ],
        )

    def visit_AssignStmt(self, node: gtir.AssignStmt) -> ast.Assign:
        return ast.Assign(targets=[self.visit(node.left)], value=self.visit(node.right))

    def visit_BinaryOp(self, node: gtir.BinaryOp) -> ast.BinOp:
        return ast.BinOp(
            left=self.visit(node.left), right=self.visit(node.right), op=self.visit(node.op)
        )
        return ast.BinOp()

    def visit_BinaryOperator(
        self, node: common.BinaryOperator
    ) -> Union[ast.Add, ast.Sub, ast.Mult, ast.Div]:
        return self.GTIR_OP_TO_AST_OP[node]()

    def visit_FieldAccess(self, node: gtir.FieldAccess) -> ast.Subscript:
        return ast.Subscript(
            value=ast.Name(id=node.name),
            slice=ast.Index(value=ast.Tuple(elts=self.visit(node.offset))),
        )

    def visit_CartesianOffset(
        self, node: gtir.CartesianOffset
    ) -> List[Union[ast.Name, ast.BinOp]]:
        return [offset_to_index_item(name, offset) for name, offset in node.to_dict().items()]

    def visit_Literal(self, node: gtir.Literal) -> Union[ast.Num, ast.NameConstant]:
        if node.dtype == common.DataType.BOOL or node.value is True or node.value is False:
            return ast.NameConstant(value=bool(ast.literal_eval(node.value)))
        else:
            return ast.Num(n=ast.literal_eval(node.value))
