# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ast
import dataclasses
import typing
from typing import Any, cast

from gt4py.next import errors
from gt4py.next.ffront import (
    dialect_ast_enums,
    program_ast as past,
    source_utils,
    stages as ffront_stages,
    type_specifications as ts_ffront,
)
from gt4py.next.ffront.dialect_parser import DialectParser
from gt4py.next.ffront.past_passes.closure_var_type_deduction import ClosureVarTypeDeduction
from gt4py.next.ffront.past_passes.type_deduction import ProgramTypeDeduction
from gt4py.next.ffront.stages import AOT_DSL_PRG, AOT_PRG, DSL_PRG, PRG
from gt4py.next.otf import toolchain, workflow
from gt4py.next.type_system import type_specifications as ts, type_translation


def func_to_past(inp: DSL_PRG) -> PRG:
    """
    Turn a DSL program definition into a PAST Program definition, adding metadata.

    Examples:

        >>> from gt4py import next as gtx
        >>> IDim = gtx.Dimension("I")

        >>> @gtx.field_operator
        ... def copy(a: gtx.Field[[IDim], gtx.float32]) -> gtx.Field[[IDim], gtx.float32]:
        ...     return a

        >>> def dsl_program(a: gtx.Field[[IDim], gtx.float32], out: gtx.Field[[IDim], gtx.float32]):
        ...     copy(a, out=out)

        >>> dsl_definition = gtx.ffront.stages.ProgramDefinition(definition=dsl_program)
        >>> past_definition = func_to_past(dsl_definition)

        >>> print(past_definition.past_node.id)
        dsl_program

        >>> assert "copy" in past_definition.closure_vars
    """
    source_def = source_utils.SourceDefinition.from_function(inp.definition)
    closure_vars = source_utils.get_closure_vars_from_function(inp.definition)
    annotations = typing.get_type_hints(inp.definition)
    return ffront_stages.PastProgramDefinition(
        past_node=ProgramParser.apply(source_def, closure_vars, annotations),
        closure_vars=closure_vars,
        grid_type=inp.grid_type,
        debug=inp.debug,
    )


def func_to_past_factory(cached: bool = True) -> workflow.Workflow[DSL_PRG, PRG]:
    """
    Wrap `func_to_past` in a chainable and optionally cached workflow step.

    Caching is switched off by default, because whether recompiling is necessary can only be known after
    the closure variables have been collected (which is done in this step). In special cases where it can
    be guaranteed that the closure variables do not change, switching caching on should be safe.
    """
    wf = workflow.make_step(func_to_past)
    if cached:
        wf = workflow.CachedStep(wf, hash_function=ffront_stages.fingerprint_stage)
    return wf


def adapted_func_to_past_factory(**kwargs: Any) -> workflow.Workflow[AOT_DSL_PRG, AOT_PRG]:
    """
    Wrap an adapter around the DSL definition -> PAST definition step to fit into transform toolchains.
    """
    return toolchain.DataOnlyAdapter(func_to_past_factory(**kwargs))


@dataclasses.dataclass(frozen=True, kw_only=True)
class ProgramParser(DialectParser[past.Program]):
    """Parse program definition from Python source code into PAST."""

    @classmethod
    def _postprocess_dialect_ast(
        cls, output_node: past.Program, closure_vars: dict[str, Any], annotations: dict[str, Any]
    ) -> past.Program:
        output_node = ClosureVarTypeDeduction.apply(output_node, closure_vars)
        return ProgramTypeDeduction.apply(output_node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> past.Program:
        closure_symbols: list[past.Symbol] = [
            past.Symbol(
                id=name,
                type=type_translation.from_value(val),
                namespace=dialect_ast_enums.Namespace.CLOSURE,
                location=self.get_location(node),
            )
            for name, val in self.closure_vars.items()
        ]

        return past.Program(
            id=node.name,
            type=ts.DeferredType(constraint=ts_ffront.ProgramType),
            params=self.visit(node.args),
            body=[self.visit(node) for node in node.body],
            closure_vars=closure_symbols,
            location=self.get_location(node),
        )

    def visit_arguments(self, node: ast.arguments) -> list[past.DataSymbol]:
        return [self.visit_arg(arg) for arg in node.args]

    def visit_arg(self, node: ast.arg) -> past.DataSymbol:
        loc = self.get_location(node)
        if (annotation := self.annotations.get(node.arg, None)) is None:
            raise errors.MissingParameterAnnotationError(loc, node.arg)
        new_type = type_translation.from_type_hint(annotation)
        if not isinstance(new_type, ts.DataType):
            raise errors.InvalidParameterAnnotationError(loc, node.arg, new_type)
        return past.DataSymbol(id=node.arg, location=loc, type=new_type)

    def visit_Expr(self, node: ast.Expr) -> past.LocatedNode:
        return self.visit(node.value)

    def visit_Add(self, node: ast.Add, **kwargs: Any) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.ADD

    def visit_Sub(self, node: ast.Sub, **kwargs: Any) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.SUB

    def visit_Mult(self, node: ast.Mult, **kwargs: Any) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.MULT

    def visit_Div(self, node: ast.Div, **kwargs: Any) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.DIV

    def visit_FloorDiv(self, node: ast.FloorDiv, **kwargs: Any) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.FLOOR_DIV

    def visit_Pow(self, node: ast.Pow, **kwargs: Any) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.POW

    def visit_Mod(self, node: ast.Mod, **kwargs: Any) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.MOD

    def visit_BitAnd(self, node: ast.BitAnd, **kwargs: Any) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.BIT_AND

    def visit_BitOr(self, node: ast.BitOr, **kwargs: Any) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.BIT_OR

    def visit_BitXor(self, node: ast.BitXor, **kwargs: Any) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.BIT_XOR

    def visit_BinOp(self, node: ast.BinOp, **kwargs: Any) -> past.BinOp:
        return past.BinOp(
            op=self.visit(node.op),
            left=self.visit(node.left),
            right=self.visit(node.right),
            location=self.get_location(node),
        )

    def visit_Name(self, node: ast.Name) -> past.Name:
        return past.Name(id=node.id, location=self.get_location(node))

    def visit_Attribute(self, node: ast.Attribute) -> past.Attribute:
        if not isinstance(node.ctx, ast.Load):
            raise errors.DSLError(
                self.get_location(node), "`node.ctx` can only be of type ast.Load"
            )
        assert isinstance(node.value, (ast.Name, ast.Attribute))

        return past.Attribute(
            attr=node.attr,
            value=self.visit(node.value),
            location=self.get_location(node),
        )

    def visit_Dict(self, node: ast.Dict) -> past.Dict:
        return past.Dict(
            keys_=[self.visit(cast(ast.AST, param)) for param in node.keys],
            values_=[self.visit(param) for param in node.values],
            location=self.get_location(node),
        )

    def visit_Call(self, node: ast.Call) -> past.Call:
        loc = self.get_location(node)
        new_func = self.visit(node.func)
        if not isinstance(new_func, past.Name):
            raise errors.DSLError(
                loc, "Functions must be referenced by their name in function calls."
            )

        return past.Call(
            func=new_func,
            args=[self.visit(arg) for arg in node.args],
            kwargs={arg.arg: self.visit(arg.value) for arg in node.keywords},
            location=loc,
        )

    def visit_Subscript(self, node: ast.Subscript) -> past.Subscript:
        return past.Subscript(
            value=self.visit(node.value),
            slice_=self.visit(node.slice),
            location=self.get_location(node),
        )

    def visit_Tuple(self, node: ast.Tuple) -> past.TupleExpr:
        return past.TupleExpr(
            elts=[self.visit(item) for item in node.elts],
            location=self.get_location(node),
            type=ts.DeferredType(constraint=ts.TupleType),
        )

    def visit_Slice(self, node: ast.Slice) -> past.Slice:
        return past.Slice(
            lower=self.visit(node.lower) if node.lower is not None else None,
            upper=self.visit(node.upper) if node.upper is not None else None,
            step=self.visit(node.step) if node.step is not None else None,
            location=self.get_location(node),
        )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> past.Constant:
        loc = self.get_location(node)
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            symbol_type = type_translation.from_value(node.operand.value)
            return past.Constant(value=-node.operand.value, type=symbol_type, location=loc)
        raise errors.DSLError(loc, "Unary operators are only applicable to literals.")

    def visit_Constant(self, node: ast.Constant) -> past.Constant:
        symbol_type = type_translation.from_value(node.value)
        return past.Constant(value=node.value, type=symbol_type, location=self.get_location(node))
