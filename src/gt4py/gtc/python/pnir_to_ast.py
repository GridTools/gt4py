import ast
from typing import Dict, List, Tuple, Union, cast

import astor
import black
from eve.visitors import NodeVisitor

from gt4py.backend.gtc_backend.stencil_module_builder import (
    FieldInfoBuilder,
    StencilClassBuilder,
    StencilModuleBuilder,
    parse_node,
)
from gt4py.gtc import common, gtir
from gt4py.gtc.python import pnir


GET_DOMAIN_TPL = """
def default_domain(*args):
    lengths = zip(*(i.shape for i in args))
    return tuple(max(*length) for length in lengths)
"""

DEFAULT_DOMAIN_TPL = """
if _domain_ is None:
    _domain_ = default_domain(fields_placeholder)
"""


def offset_to_index_item(name: str, offset: int):
    if offset == 0:
        return ast.Name(id=name.upper())
    elif offset > 0:
        return ast.BinOp(left=ast.Name(id=name.upper()), right=ast.Num(n=offset), op=ast.Add())
    return ast.BinOp(left=ast.Name(id=name.upper()), right=ast.Num(n=abs(offset)), op=ast.Sub())


def domain_for(
    loop_index_name: str,
    domain_index_or_range_args: Union[int, List[ast.AST]],
    body: List[ast.AST],
):
    range_args = domain_index_or_range_args
    if isinstance(domain_index_or_range_args, int):
        range_args = [
            ast.Subscript(
                value=ast.Name(id="_domain_"),
                slice=ast.Index(value=ast.Num(n=domain_index_or_range_args)),
            )
        ]
    return ast.For(
        target=ast.Name(id=loop_index_name),
        iter=ast.Call(
            func=ast.Name(id="range"),
            args=range_args,
            keywords=[],
        ),
        orelse=[],
        body=body,
    )


class PnirToAst(NodeVisitor):
    GTIR_OP_TO_AST_OP = {
        common.BinaryOperator.ADD: ast.Add,
        common.BinaryOperator.SUB: ast.Sub,
        common.BinaryOperator.MUL: ast.Mult,
        common.BinaryOperator.DIV: ast.Div,
    }

    def __init__(self, *, onemodule=False):
        self._onemodule = onemodule

    def visit_Stencil(
        self, node: pnir.Stencil
    ) -> Union[StencilModuleBuilder, Tuple[StencilModuleBuilder, ast.Module]]:
        comp_module = self.visit(node.computation)
        class_builder = self.visit(node.stencil_obj).source(
            black.format_str(astor.to_source(comp_module), mode=black.Mode())
        )
        module_builder = (
            StencilModuleBuilder().stencil_class(class_builder).name(node.stencil_obj.name)
        )
        if self._onemodule:
            for line in cast(ast.Call, comp_module.body[0]).body:
                class_builder.add_run_line(line)
            return module_builder
        else:
            module_builder.import_(parse_node("import computation")).add_relpaths(".")
            run_call = parse_node(
                f"computation.run({', '.join(class_builder.signature)}, _domain_=_domain_)"
            )
            class_builder.add_run_line(ast.Expr(value=run_call))
            return module_builder, comp_module

    def visit_StencilObject(self, node: pnir.StencilObject) -> StencilClassBuilder:
        builder = (
            StencilClassBuilder()
            .name(node.name)
            .field_names(*(decl.name for decl in node.params))
            .backend("gtc:py")
        )
        for name, field_info in self.visit(node.fields_metadata).items():
            builder.add_gt_field_info(name=name, field_info=field_info)
        return builder

    def visit_FieldsMetadata(self, node: gtir.FieldsMetadata) -> Dict[str, ast.Call]:
        return {name: self.visit(field_meta) for name, field_meta in node.metas.items()}

    def visit_FieldMetadata(self, node: gtir.FieldMetadata) -> ast.Call:
        return (
            FieldInfoBuilder()
            .access(node.access)
            .boundary((node.boundary.i, node.boundary.j, node.boundary.k))
            .dtype(node.dtype.name.lower())
            .build()
        )

    def visit_Module(self, node: pnir.Module) -> ast.Module:
        get_domain_def = parse_node(GET_DOMAIN_TPL)
        body = [] if self._onemodule else [get_domain_def]
        return ast.Module(body=[*body, self.visit(node.run)])

    def visit_RunFunction(self, node: pnir.RunFunction) -> ast.FunctionDef:
        default_domain_if = cast(ast.If, parse_node(DEFAULT_DOMAIN_TPL))
        cast(ast.Call, cast(ast.Assign, default_domain_if.body[0]).value).args = [
            cast(ast.expr, parse_node(name)) for name in node.field_params
        ]
        body = [] if self._onemodule else [default_domain_if]
        return ast.FunctionDef(
            name="run",
            args=ast.arguments(
                args=[ast.arg(arg=name, annotation=None) for name in node.field_params],
                vararg=None,
                kwonlyargs=[ast.arg(arg=name, annotation=None) for name in node.scalar_params]
                + [ast.arg(arg="_domain_", annotation=None)],
                kw_defaults=[parse_node("None")],
                kwarg=None,
                defaults=[],
            ),
            decorator_list=[],
            returns=None,
            body=body + [self.visit(k_loop) for k_loop in node.k_loops],
        )

    def visit_KLoop(self, node: pnir.KLoop) -> ast.For:
        return domain_for(
            loop_index_name="K",
            domain_index_or_range_args=[self.visit(node.lower), self.visit(node.upper)],
            body=[self.visit(stmt) for stmt in node.ij_loops],
        )

    def visit_AxisBound(self, node: gtir.AxisBound) -> Union[ast.Num, ast.Subscript, ast.BinOp]:
        if node.level == common.LevelMarker.START:
            return ast.Num(node.offset)
        subscript = ast.Subscript(value=ast.Name(id="_domain_"), slice=ast.Index(value=ast.Num(2)))
        if node.offset == 0:
            return subscript
        return ast.BinOp(left=subscript, op=ast.Sub(), right=ast.Num(n=node.offset))

    def visit_IJLoop(self, node: pnir.IJLoop) -> ast.For:
        return domain_for(
            loop_index_name="J",
            domain_index_or_range_args=1,
            body=[
                domain_for(
                    loop_index_name="I",
                    domain_index_or_range_args=0,
                    body=[self.visit(stmt) for stmt in node.body],
                )
            ],
        )

    def visit_AssignStmt(self, node: pnir.AssignStmt) -> ast.Assign:
        return ast.Assign(targets=[self.visit(node.left)], value=self.visit(node.right))

    def visit_BinaryOp(self, node: gtir.BinaryOp) -> ast.BinOp:
        return ast.BinOp(
            left=self.visit(node.left), right=self.visit(node.right), op=self.visit(node.op)
        )
        return ast.BinOp()

    def visit_BinaryOperator(self, node: common.BinaryOperator) -> ast.operator:
        return self.GTIR_OP_TO_AST_OP[node]()

    def visit_FieldAccess(self, node: gtir.FieldAccess) -> ast.Subscript:
        return ast.Subscript(
            value=ast.Name(id=node.name),
            slice=ast.Index(value=ast.Tuple(elts=self.visit(node.offset))),
        )

    def visit_ScalarAccess(self, node: gtir.ScalarAccess) -> ast.Name:
        return ast.Name(id=node.name)

    def visit_CartesianOffset(
        self, node: gtir.CartesianOffset
    ) -> List[Union[ast.Name, ast.BinOp]]:
        return [offset_to_index_item(name, offset) for name, offset in node.to_dict().items()]

    def visit_Literal(self, node: gtir.Literal) -> Union[ast.Num, ast.NameConstant]:
        if node.dtype == common.DataType.BOOL or node.value is True or node.value is False:
            return ast.NameConstant(value=bool(ast.literal_eval(node.value)))
        else:
            return ast.Num(n=ast.literal_eval(node.value))
