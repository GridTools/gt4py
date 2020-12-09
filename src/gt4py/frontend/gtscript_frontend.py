# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import ast
import copy
import enum
import inspect
import itertools
import numbers
import textwrap
import types
from typing import List, Optional, Union

import numpy as np

from gt4py import definitions as gt_definitions
from gt4py import frontend as gt_frontend
from gt4py import gtscript
from gt4py import ir as gt_ir
from gt4py import utils as gt_utils
from gt4py.utils import NOTHING
from gt4py.utils import meta as gt_meta


class GTScriptSyntaxError(gt_definitions.GTSyntaxError):
    def __init__(self, message, *, loc=None):
        super().__init__(message, frontend=GTScriptFrontend.name)
        self.loc = loc


class GTScriptSymbolError(GTScriptSyntaxError):
    def __init__(self, name, message=None, *, loc=None):
        if message is None:
            if loc is None:
                message = "Unknown symbol '{name}' symbol".format(name=name)
            else:
                message = "Unknown symbol '{name}' symbol in '{scope}' (line: {line}, col: {col})".format(
                    name=name, scope=loc.scope, line=loc.line, col=loc.column
                )
        super().__init__(message, loc=loc)
        self.name = name


class GTScriptDefinitionError(GTScriptSyntaxError):
    def __init__(self, name, value, message=None, *, loc=None):
        if message is None:
            if loc is None:
                message = "Invalid definition for '{name}' symbol".format(name=name)
            else:
                message = "Invalid definition for '{name}' symbol in '{scope}' (line: {line}, col: {col})".format(
                    name=name, scope=loc.scope, line=loc.line, col=loc.column
                )
        super().__init__(message, loc=loc)
        self.name = name
        self.value = value


class GTScriptValueError(GTScriptDefinitionError):
    def __init__(self, name, value, message=None, *, loc=None):
        if message is None:
            if loc is None:
                message = "Invalid value for '{name}' symbol ".format(name=name)
            else:
                message = (
                    "Invalid value for '{name}' in '{scope}' (line: {line}, col: {col})".format(
                        name=name, scope=loc.scope, line=loc.line, col=loc.column
                    )
                )
        super().__init__(name, value, message, loc=loc)


# class GTScriptConstValueError(GTScriptValueError):
#     def __init__(self, name, value, message=None, *, loc=None):
#         if message is None:
#             if loc is None:
#                 message = "Non-constant definition of constant '{name}' symbol ".format(name=name)
#             else:
#                 message = "Non-constant definition of constant '{name}' in '{scope}' (line: {line}, col: {col})".format(
#                     name=name, scope=loc.scope, line=loc.line, col=loc.column
#                 )
#         super().__init__(name, value, message, loc=loc)
#
#
class GTScriptDataTypeError(GTScriptSyntaxError):
    def __init__(self, name, data_type, message=None, *, loc=None):
        if message is None:
            if loc is None:
                message = "Invalid data type for '{name}' numeric symbol ".format(name=name)
            else:
                message = "Invalid data type for '{name}' numeric symbol in '{scope}' (line: {line}, col: {col})".format(
                    name=name, scope=loc.scope, line=loc.line, col=loc.column
                )
        super().__init__(message, loc=loc)
        self.name = name
        self.data_type = data_type


class GTScriptAssertionError(gt_definitions.GTSpecificationError):
    def __init__(self, source, *, loc=None):
        if loc:
            message = f"Assertion failed at line {loc.line}, col {loc.column}:\n{source}"
        else:
            message = f"Assertion failed.\n{source}"
        super().__init__(message)
        self.loc = loc


class AssertionChecker(ast.NodeTransformer):
    """Check assertions and remove from the AST for further parsing."""

    @classmethod
    def apply(cls, func_node: ast.FunctionDef, context: dict, source: str):
        checker = cls(context, source)
        checker(func_node)

    def __init__(self, context, source):
        self.context = context
        self.source = source

    def __call__(self, func_node: ast.FunctionDef):
        self.visit(func_node)

    def visit_Assert(self, assert_node: ast.Assert) -> None:
        if assert_node.test.func.id != "__INLINED":
            raise GTScriptSyntaxError("Run-time assertions are not supported.")
        eval_node = assert_node.test.args[0]

        condition_value = gt_utils.meta.ast_eval(eval_node, self.context, default=NOTHING)
        if condition_value is not NOTHING:
            if not condition_value:
                source_lines = textwrap.dedent(self.source).split("\n")
                loc = gt_ir.Location.from_ast_node(assert_node)
                raise GTScriptAssertionError(source_lines[loc.line - 1], loc=loc)
        else:
            raise GTScriptSyntaxError(
                "Evaluation of compile-time assertion condition failed at the preprocessing step."
            )

        return None


class AxisIntervalParser(ast.NodeVisitor):
    """Parse Python AST interval syntax in the form of a Slice.

    Corner cases: `ast.Ellipsis` refers to the entire interval, and
    if an `ast.Subscript` is passed, this parses its slice attribute.
    """

    @classmethod
    def apply(
        cls,
        node: Union[ast.Slice, ast.Ellipsis, ast.Subscript],
        axis_name: str,
        context: Optional[dict] = None,
        loc: Optional[gt_ir.Location] = None,
    ) -> gt_ir.AxisInterval:
        parser = cls(axis_name, context, loc)

        if isinstance(node, ast.Ellipsis):
            interval = gt_ir.AxisInterval.full_interval()
            interval.loc = loc
            return interval

        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Index):
                slice_node = cls.single_index_slice(node.slice.value)
            else:
                slice_node = node.slice
        elif isinstance(node, ast.Slice):
            slice_node = node
        else:
            slice_node = cls.single_index_slice(node)

        if slice_node.lower is not None:
            start = parser.visit(slice_node.lower)
        else:
            start = gt_ir.AxisBound(level=gt_ir.LevelMarker.START, offset=0, extend=True)

        if slice_node.upper is not None:
            end = parser.visit(slice_node.upper)
        else:
            end = gt_ir.AxisBound(level=gt_ir.LevelMarker.END, offset=0, extend=True)

        return gt_ir.AxisInterval(start=start, end=end, loc=loc)

    def __init__(
        self,
        axis_name: str,
        context: Optional[dict] = None,
        loc: Optional[gt_ir.Location] = None,
    ):
        self.axis_name = axis_name
        self.context = context or dict()
        self.loc = loc

        # initialize possible exceptions
        self.interval_error = ValueError(
            f"Invalid 'interval' specification at line {loc.line} (column {loc.column})"
        )

    @staticmethod
    def single_index_slice(node: ast.Expr):
        slice_node = ast.Slice(
            lower=node, upper=ast.BinOp(left=node, op=ast.Add(), right=ast.Constant(value=1))
        )
        ast.copy_location(slice_node, node)
        return slice_node

    @staticmethod
    def make_axis_bound(offset: int, loc: gt_ir.Location = None) -> gt_ir.AxisBound:
        return gt_ir.AxisBound(
            level=gt_ir.LevelMarker.START if offset >= 0 else gt_ir.LevelMarker.END,
            offset=offset,
            loc=loc,
        )

    def visit_Name(self, node: ast.Name) -> gt_ir.AxisBound:
        symbol = node.id
        if symbol in self.context:
            value = self.context[symbol]
            if isinstance(value, gtscript._AxisOffset):
                if value.axis != self.axis_name:
                    raise self.interval_error
                offset = value.offset
            elif isinstance(value, int):
                offset = value
            else:
                raise self.interval_error
            return self.make_axis_bound(offset, self.loc)
        else:
            return gt_ir.AxisBound(level=gt_ir.VarRef(name=symbol), loc=self.loc)

    def visit_Constant(self, node: ast.Constant) -> gt_ir.AxisBound:
        if node.value is not None:
            if isinstance(node.value, gtscript._AxisOffset):
                if node.value.axis != self.axis_name:
                    raise self.interval_error
                offset = node.value.offset
            elif isinstance(node.value, int):
                offset = node.value
            else:
                raise self.interval_error
            return self.make_axis_bound(offset, self.loc)
        else:
            return gt_ir.AxisBound(level=gt_ir.LevelMarker.END, offset=0, loc=self.loc)

    def visit_NameConstant(self, node: ast.NameConstant) -> gt_ir.AxisBound:
        """Python < 3.8 uses ast.NameConstant for 'None'."""
        if node.value is not None:
            raise self.interval_error
        else:
            return gt_ir.AxisBound(level=gt_ir.LevelMarker.END, offset=0, loc=self.loc)

    def visit_Num(self, node: ast.Num) -> gt_ir.AxisBound:
        """Equivalent to visit_Constant. Required for Python < 3.8."""
        return self.make_axis_bound(node.n, self.loc)

    def visit_BinOp(self, node: ast.BinOp) -> gt_ir.AxisBound:
        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(node.op, ast.Add):
            op = lambda x, y: x + y
        elif isinstance(node.op, ast.Sub):
            op = lambda x, y: x - y
        elif isinstance(node.op, ast.Mult):
            if left.level != right.level or not isinstance(left.level, gt_ir.LevelMarker):
                raise self.interval_error
            op = lambda x, y: x * y
        else:
            raise self.interval_error

        if right.level == gt_ir.LevelMarker.END:
            level = gt_ir.LevelMarker.END
        else:
            level = left.level

        return gt_ir.AxisBound(level=level, offset=op(left.offset, right.offset), loc=self.loc)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> gt_ir.AxisBound:
        axis_bound = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            new_level = (
                gt_ir.LevelMarker.END
                if axis_bound.level == gt_ir.LevelMarker.START
                else gt_ir.LevelMarker.START
            )
            return gt_ir.AxisBound(level=new_level, offset=-axis_bound.offset, loc=self.loc)
        else:
            raise self.interval_error

        return gt_ir.AxisBound(level=gt_ir.LevelMarker.END, offset=0, loc=self.loc)

    def visit_Subscript(self, node: ast.Subscript) -> gt_ir.AxisBound:
        if node.value.id != self.axis_name:
            raise self.interval_error

        if not isinstance(node.slice, ast.Index):
            raise self.interval_error

        return self.visit(node.slice.value)


parse_interval_node = AxisIntervalParser.apply


class ValueInliner(ast.NodeTransformer):
    @classmethod
    def apply(cls, func_node: ast.FunctionDef, context: dict):
        inliner = cls(context)
        inliner(func_node)

    def __init__(self, context):
        self.context = context
        self.prefix = ""

    def __call__(self, func_node: ast.FunctionDef):
        self.visit(func_node)

    def _replace_node(self, name_or_attr_node):
        new_node = name_or_attr_node
        qualified_name = gt_meta.get_qualified_name_from_node(name_or_attr_node)
        if qualified_name in self.context:
            value = self.context[qualified_name]
            if value is None or isinstance(value, bool):
                new_node = ast.NameConstant(value=value)
            elif isinstance(value, numbers.Number):
                new_node = ast.Num(n=value)
            elif isinstance(value, gtscript._AxisOffset):
                new_node = ast.Constant(value=value)
            elif hasattr(value, "_gtscript_"):
                pass
            else:
                assert False
        return new_node

    def visit_ImportFrom(self, node: ast.ImportFrom):
        return node

    def visit_Attribute(self, node: ast.Attribute):
        return self._replace_node(node)

    def visit_Name(self, node: ast.Name):
        return self._replace_node(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.body = [self.visit(n) for n in node.body]
        return node


class ReturnReplacer(gt_utils.meta.ASTTransformPass):
    @classmethod
    def apply(cls, ast_object: ast.AST, target_node: ast.AST) -> None:
        """Ensure that there is only a single return statement (can still return a tuple)."""
        ret_count = sum(isinstance(node, ast.Return) for node in ast.walk(ast_object))
        if ret_count != 1:
            raise GTScriptSyntaxError("GTScript Functions should have a single return statement")
        cls().visit(ast_object, target_node=target_node)

    @staticmethod
    def _get_num_values(node: ast.AST) -> int:
        return len(node.elts) if isinstance(node, ast.Tuple) else 1

    def visit_Return(self, node: ast.Return, *, target_node: ast.AST) -> ast.Assign:
        rhs_length = self._get_num_values(node.value)
        lhs_length = self._get_num_values(target_node)

        if lhs_length == rhs_length:
            return ast.Assign(
                targets=[target_node],
                value=node.value,
                lineno=node.lineno,
                col_offset=node.col_offset,
            )
        else:
            raise GTScriptSyntaxError(
                "Number of returns values does not match arguments on left side"
            )


class CallInliner(ast.NodeTransformer):
    """Inlines calls to gtscript.function calls.

    Calls to NativeFunctions (intrinsic math functions) are kept in the IR and
    dealt with in the IRMaker.
    """

    @classmethod
    def apply(cls, func_node: ast.FunctionDef, context: dict):
        inliner = cls(context)
        inliner(func_node)
        return inliner.all_skip_names

    def __init__(self, context: dict):
        self.context = context
        self.current_block = None
        self.all_skip_names = set(gtscript.builtins | {"gt4py", "gtscript"})

    def __call__(self, func_node: ast.FunctionDef):
        self.visit(func_node)

    def visit(self, node, **kwargs):
        """Visit a node."""
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, **kwargs)

    def _process_stmts(self, stmts):
        new_stmts = []
        outer_block = self.current_block
        self.current_block = new_stmts
        for s in stmts:
            if not isinstance(s, (ast.Import, ast.ImportFrom)):
                if self.visit(s):
                    new_stmts.append(s)
        self.current_block = outer_block

        return new_stmts

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.body = self._process_stmts(node.body)
        return node

    def visit_With(self, node: ast.With):
        node.body = self._process_stmts(node.body)
        return node

    def visit_If(self, node: ast.If):
        node.body = self._process_stmts(node.body)
        if node.orelse:
            node.orelse = self._process_stmts(node.orelse)
        return node

    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, ast.Call) and node.value.func.id not in gtscript.MATH_BUILTINS:
            assert len(node.targets) == 1
            self.visit(node.value, target_node=node.targets[0])
            # This node can be now removed since the trivial assignment has been already done
            # in the Call visitor
            return None
        else:
            return self.generic_visit(node)

    def visit_Call(self, node: ast.Call, *, target_node=None):
        call_name = node.func.id

        if call_name in gtscript.MATH_BUILTINS:
            node.args = [self.visit(arg) for arg in node.args]
            return node
        elif any(
            isinstance(arg, ast.Call) and arg.func.id not in gtscript.MATH_BUILTINS
            for arg in node.args
        ):
            raise GTScriptSyntaxError(
                "Function calls are not supported in arguments to function calls",
                loc=gt_ir.Location.from_ast_node(node),
            )

        elif call_name not in self.context and not hasattr(self.context[call_name], "_gtscript_"):
            raise GTScriptSyntaxError("Unknown call", loc=gt_ir.Location.from_ast_node(node))

        # Recursively inline any possible nested subroutine call
        call_info = self.context[call_name]._gtscript_
        call_ast = copy.deepcopy(call_info["ast"])
        CallInliner.apply(call_ast, call_info["local_context"])

        # Extract call arguments
        call_signature = call_info["api_signature"]
        arg_infos = {arg.name: arg.default for arg in call_signature}
        try:
            assert len(node.args) <= len(call_signature)
            call_args = {}
            for i, arg_value in enumerate(node.args):
                assert not call_signature[i].is_keyword
                call_args[call_signature[i].name] = arg_value
            for kwarg in node.keywords:
                assert kwarg.arg in arg_infos
                call_args[kwarg.arg] = kwarg.value

            # Add default values for missing args when possible
            for name in arg_infos:
                if name not in call_args:
                    assert arg_infos[name] != gt_ir.Empty
                    if (
                        (arg_infos[name] is True)
                        or arg_infos[name] is False
                        or arg_infos[name] is None
                    ):
                        call_args[name] = ast.Num(n=0.0)  # ast.NameConstant(value=arg_infos[name])
                    else:
                        call_args[name] = ast.Num(n=arg_infos[name])
        except Exception:
            raise GTScriptSyntaxError(
                message="Invalid call signature", loc=gt_ir.Location.from_ast_node(node)
            )

        # Rename local names in subroutine to avoid conflicts with caller context names
        try:
            assign_targets = gt_meta.collect_assign_targets(call_ast, allow_multiple_targets=False)
        except RuntimeError as e:
            raise GTScriptSyntaxError(
                message="Assignment to more than one target is not supported."
            ) from e

        assigned_symbols = set()
        for target in assign_targets:
            if not isinstance(target, ast.Name):
                raise GTScriptSyntaxError(message="Unsupported assignment target.", loc=target)

            assigned_symbols.add(target.id)

        name_mapping = {
            name: value.id
            for name, value in call_args.items()
            if isinstance(value, ast.Name) and name not in assigned_symbols
        }

        call_id = gt_utils.shashed_id(call_name)[:3]
        call_id_suffix = f"{call_id}_{node.lineno}_{node.col_offset}"
        template_fmt = "{name}__" + call_id_suffix

        gt_meta.map_symbol_names(
            call_ast, name_mapping, template_fmt=template_fmt, skip_names=self.all_skip_names
        )

        # Replace returns by assignments in subroutine
        if target_node is None:
            target_node = ast.Name(
                ctx=ast.Store(),
                lineno=node.lineno,
                col_offset=node.col_offset,
                id=template_fmt.format(name="RETURN_VALUE"),
            )

        assert isinstance(target_node, (ast.Name, ast.Tuple)) and isinstance(
            target_node.ctx, ast.Store
        )

        ReturnReplacer.apply(call_ast, target_node)

        # Add subroutine sources prepending the required arg assignments
        inlined_stmts = []
        for arg_name, arg_value in call_args.items():
            if arg_name not in name_mapping:
                inlined_stmts.append(
                    ast.Assign(
                        lineno=node.lineno,
                        col_offset=node.col_offset,
                        targets=[
                            ast.Name(
                                ctx=ast.Store(),
                                lineno=node.lineno,
                                col_offset=node.col_offset,
                                id=template_fmt.format(name=arg_name),
                            )
                        ],
                        value=arg_value,
                    )
                )

        # Add inlined statements to the current block and return name node with the result
        inlined_stmts.extend(call_ast.body)
        self.current_block.extend(inlined_stmts)
        if isinstance(target_node, ast.Name):
            result_node = ast.Name(
                ctx=ast.Load(),
                lineno=target_node.lineno,
                col_offset=target_node.col_offset,
                id=target_node.id,
            )
        else:
            result_node = ast.Tuple(
                ctx=ast.Load(),
                lineno=target_node.lineno,
                col_offset=target_node.col_offset,
                elts=target_node.elts,
            )

        return result_node

    def visit_Expr(self, node: ast.Expr):
        """ignore pure string statements in callee"""
        if not isinstance(node.value, ast.Str):
            return super().visit(node)


class CompiledIfInliner(ast.NodeTransformer):
    @classmethod
    def apply(cls, ast_object, context):
        preprocessor = cls(context)
        preprocessor(ast_object)

    def __init__(self, context):
        self.context = context

    def __call__(self, ast_object):
        self.visit(ast_object)

    def visit_If(self, node: ast.If):
        # Compile-time evaluation of "if" conditions
        node = self.generic_visit(node)
        if (
            isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Name)
            and node.test.func.id == "__INLINED"
            and len(node.test.args) == 1
        ):
            eval_node = node.test.args[0]
            condition_value = gt_utils.meta.ast_eval(eval_node, self.context, default=NOTHING)
            if condition_value is not NOTHING:
                node = node.body if condition_value else node.orelse
            else:
                raise GTScriptSyntaxError(
                    "Evaluation of compile-time 'IF' condition failed at the preprocessing step"
                )

        return node if node else None


#
# class Cleaner(gt_ir.IRNodeVisitor):
#     @classmethod
#     def apply(cls, ast_object):
#         cleaner = cls()
#         cleaner(ast_object)
#
#     def __init__(self):
#         self.defines = {}
#         self.writes = {}
#         self.reads = {}
#         self.aliases = {}
#
#     def __call__(self, ast_object):
#         self.visit(ast_object)
#
#     def visit_FieldRef(self, node: gt_ir.FieldRef):
#         assert node.name in self.defines
#         self.reads[node.name] = self.reads.get(node.name, 0) + 1
#
#     def visit_FieldDecl(self, node: gt_ir.FieldDecl):
#         assert node.is_api is False
#         self.defines[node.name] = "temp"
#
#     def visit_Assign(self, node: gt_ir.Assign):
#         if isinstance(node.target, gt_ir.FieldRef):
#             for alias in self.aliases.get(node.target.name, set()):
#                 self.aliases[alias] -= {node.target.name}
#                 self.aliases[node.target.name] = set()
#
#             if isinstance(node.value, gt_ir.FieldRef):
#                 if node.target.name not in itertools.chain(self.writes.keys(), self.reads.keys()):
#                     aliases = set(self.aliases.setdefault(node.value.name, set()))
#                     self.aliases.setdefault(node.target.name, set())
#                     self.aliases[node.target.name] |= aliases | {node.value.name}
#                     for alias in self.aliases[node.target.name]:
#                         self.aliases[alias] |= {node.target.name}
#
#         self.visit(node.value)
#         self.writes[node.target.name] = self.reads.get(node.target.name, 0) + 1
#
#     def visit_StencilDefinition(self, node: gt_ir.StencilDefinition):
#         for decl in node.api_fields:
#             self.defines[decl.name] = "api"
#         for computation in node.computations:
#             self.visit(computation)
#
#         for a in ["defines", "reads", "writes"]:
#             print(f"{a}: ", getattr(self, a))
#


@enum.unique
class ParsingContext(enum.Enum):
    CONTROL_FLOW = 1
    COMPUTATION = 2


class IRMaker(ast.NodeVisitor):
    def __init__(
        self,
        fields: dict,
        parameters: dict,
        local_symbols: dict,
        *,
        domain: gt_ir.Domain,
        extra_temp_decls: dict,
    ):
        fields = fields or {}
        parameters = parameters or {}
        assert all(isinstance(name, str) for name in parameters.keys())
        local_symbols = local_symbols or {}
        assert all(isinstance(name, str) for name in local_symbols.keys()) and all(
            isinstance(value, (type, np.dtype)) for value in local_symbols.values()
        )

        self.fields = fields
        self.parameters = parameters
        self.local_symbols = local_symbols
        self.domain = domain or gt_ir.Domain.LatLonGrid()
        self.extra_temp_decls = extra_temp_decls or {}
        self.parsing_context = None
        self.if_decls_stack = []
        gt_ir.NativeFunction.PYTHON_SYMBOL_TO_IR_OP = {
            "abs": gt_ir.NativeFunction.ABS,
            "min": gt_ir.NativeFunction.MIN,
            "max": gt_ir.NativeFunction.MAX,
            "mod": gt_ir.NativeFunction.MOD,
            "sin": gt_ir.NativeFunction.SIN,
            "cos": gt_ir.NativeFunction.COS,
            "tan": gt_ir.NativeFunction.TAN,
            "asin": gt_ir.NativeFunction.ARCSIN,
            "acos": gt_ir.NativeFunction.ARCCOS,
            "atan": gt_ir.NativeFunction.ARCTAN,
            "sqrt": gt_ir.NativeFunction.SQRT,
            "exp": gt_ir.NativeFunction.EXP,
            "log": gt_ir.NativeFunction.LOG,
            "isfinite": gt_ir.NativeFunction.ISFINITE,
            "isinf": gt_ir.NativeFunction.ISINF,
            "isnan": gt_ir.NativeFunction.ISNAN,
            "floor": gt_ir.NativeFunction.FLOOR,
            "ceil": gt_ir.NativeFunction.CEIL,
            "trunc": gt_ir.NativeFunction.TRUNC,
        }

    def __call__(self, ast_root: ast.AST):
        assert (
            isinstance(ast_root, ast.Module)
            and "body" in ast_root._fields
            and len(ast_root.body) == 1
            and isinstance(ast_root.body[0], ast.FunctionDef)
        )
        func_ast = ast_root.body[0]
        self.parsing_context = ParsingContext.CONTROL_FLOW
        computations = self.visit(func_ast)

        return computations

    # Helpers functions
    def _is_field(self, name: str):
        return name in self.fields

    def _is_parameter(self, name: str):
        return name in self.parameters

    def _is_local_symbol(self, name: str):
        return name in self.local_symbols

    def _is_known(self, name: str):
        return self._is_field(name) or self._is_parameter(name) or self._is_local_symbol(name)

    def _get_qualified_name(self, node: ast.AST, *, joiner="."):
        if isinstance(node, ast.Name):
            result = node.id
        elif isinstance(node, ast.Attribute):
            prefix = self._get_qualified_name(node.value)
            result = joiner.join([prefix, node.attr])
        else:
            result = None

        return result

    def _are_blocks_sorted(self, compute_blocks: List[gt_ir.ComputationBlock]):
        def sort_blocks_key(comp_block):
            start = comp_block.interval.start
            assert isinstance(start.level, gt_ir.LevelMarker)
            key = 0 if start.level == gt_ir.LevelMarker.START else 100000
            key += start.offset
            return key

        if len(compute_blocks) < 1:
            return True

        # validate invariant
        assert all(
            comp_block.iteration_order == compute_blocks[0].iteration_order
            for comp_block in compute_blocks
        )

        # extract iteration order
        iteration_order = compute_blocks[0].iteration_order

        # sort blocks
        compute_blocks_sorted = sorted(
            compute_blocks,
            key=sort_blocks_key,
            reverse=iteration_order == gt_ir.IterationOrder.BACKWARD,
        )

        # if sorting didn't change anything it was already sorted
        return compute_blocks == compute_blocks_sorted

    def _visit_iteration_order_node(self, node: ast.withitem, loc: gt_ir.Location):
        syntax_error = GTScriptSyntaxError(
            f"Invalid 'computation' specification at line {loc.line} (column {loc.column})",
            loc=loc,
        )
        comp_node = node.context_expr
        if len(comp_node.args) + len(comp_node.keywords) != 1 or any(
            keyword.arg not in ["order"] for keyword in comp_node.keywords
        ):
            raise syntax_error

        if comp_node.args:
            iteration_order_node = comp_node.args[0]
        else:
            iteration_order_node = comp_node.keywords[0].value
        if (
            not isinstance(iteration_order_node, ast.Name)
            or iteration_order_node.id not in gt_ir.IterationOrder.__members__
        ):
            raise syntax_error

        return gt_ir.IterationOrder[iteration_order_node.id]

    def _visit_interval_node(self, node: ast.withitem, loc: gt_ir.Location):
        range_error = GTScriptSyntaxError(
            f"Invalid interval range specification at line {loc.line} (column {loc.column})",
            loc=loc,
        )

        if node.context_expr.args:
            args = node.context_expr.args
        else:
            args = [keyword.value for keyword in node.context_expr.keywords]
            if len(args) != 2:
                raise range_error

        if len(args) == 2:
            if any(isinstance(arg, ast.Subscript) for arg in args):
                raise GTScriptSyntaxError(
                    "Two-argument syntax should not use AxisOffsets or AxisIntervals"
                )
            interval_node = ast.Slice(lower=args[0], upper=args[1])
            ast.copy_location(interval_node, node)
        else:
            interval_node = args[0]

        interval = parse_interval_node(interval_node, "K", loc=loc)

        if (
            interval.start.level == gt_ir.LevelMarker.END
            and interval.end.level == gt_ir.LevelMarker.START
        ) or (
            interval.start.level == interval.end.level
            and interval.end.offset <= interval.start.offset
        ):
            raise range_error

        return interval

    def _visit_computation_node(self, node: ast.With) -> List[gt_ir.ComputationBlock]:
        loc = gt_ir.Location.from_ast_node(node)
        syntax_error = GTScriptSyntaxError(
            f"Invalid 'computation' specification at line {loc.line} (column {loc.column})",
            loc=loc,
        )

        # Parse computation specification, i.e. `withItems` nodes
        iteration_order = None
        interval = None

        try:
            for item in node.items:
                if (
                    isinstance(item.context_expr, ast.Call)
                    and item.context_expr.func.id == "computation"
                ):
                    assert iteration_order is None  # only one spec allowed
                    iteration_order = self._visit_iteration_order_node(item, loc)
                elif (
                    isinstance(item.context_expr, ast.Call)
                    and item.context_expr.func.id == "interval"
                ):
                    assert interval is None  # only one spec allowed
                    interval = self._visit_interval_node(item, loc)
                else:
                    raise syntax_error
        except AssertionError as e:
            raise syntax_error from e

        if iteration_order is None or interval is None:
            raise syntax_error

        #  Parse `With` body into computation blocks
        self.parsing_context = ParsingContext.COMPUTATION
        stmts = []
        for stmt in node.body:
            stmts.extend(gt_utils.listify(self.visit(stmt)))
        self.parsing_context = ParsingContext.CONTROL_FLOW

        result = gt_ir.ComputationBlock(
            interval=interval, iteration_order=iteration_order, body=gt_ir.BlockStmt(stmts=stmts)
        )

        return result

    # Visitor methods
    # -- Special nodes --
    def visit_Raise(self):
        return gt_ir.InvalidBranch()

    # -- Literal nodes --
    def visit_Num(self, node: ast.Num) -> numbers.Number:
        value = node.n
        return value

    def visit_Tuple(self, node: ast.Tuple) -> tuple:
        value = tuple(self.visit(elem) for elem in node.elts)
        return value

    def visit_NameConstant(self, node: ast.NameConstant):
        value = gt_ir.BuiltinLiteral(value=gt_ir.Builtin[str(node.value).upper()])
        return value

    # -- Symbol nodes --
    def visit_Attribute(self, node: ast.Attribute):
        qualified_name = self._get_qualified_name(node)
        return self.visit(ast.Name(id=qualified_name, ctx=node.ctx))

    def visit_Name(self, node: ast.Name) -> gt_ir.Ref:
        symbol = node.id
        if self._is_field(symbol):
            result = gt_ir.FieldRef(
                name=symbol,
                offset={axis: value for axis, value in zip(self.domain.axes_names, (0, 0, 0))},
            )
        elif self._is_parameter(symbol):
            result = gt_ir.VarRef(name=symbol)
        elif self._is_local_symbol(symbol):
            assert False  # result = gt_ir.VarRef(name=symbol)
        else:
            assert False, "Missing '{}' symbol definition".format(symbol)

        return result

    def visit_Index(self, node: ast.Index):
        index = self.visit(node.value)
        return index

    def visit_Subscript(self, node: ast.Subscript):
        assert isinstance(node.ctx, (ast.Load, ast.Store))
        index = self.visit(node.slice)
        result = self.visit(node.value)
        if isinstance(result, gt_ir.VarRef):
            result.index = index
        else:
            result.offset = {axis.name: value for axis, value in zip(self.domain.axes, index)}

        return result

    # -- Expressions nodes --
    def visit_UnaryOp(self, node: ast.UnaryOp):
        op = self.visit(node.op)
        arg = self.visit(node.operand)
        if isinstance(arg, numbers.Number):
            result = eval("{op}{arg}".format(op=op.python_symbol, arg=arg))
        else:
            result = gt_ir.UnaryOpExpr(op=op, arg=arg)

        return result

    def visit_UAdd(self, node: ast.UAdd) -> gt_ir.UnaryOperator:
        return gt_ir.UnaryOperator.POS

    def visit_USub(self, node: ast.USub) -> gt_ir.UnaryOperator:
        return gt_ir.UnaryOperator.NEG

    def visit_Not(self, node: ast.Not) -> gt_ir.UnaryOperator:
        return gt_ir.UnaryOperator.NOT

    def visit_BinOp(self, node: ast.BinOp) -> gt_ir.BinOpExpr:
        op = self.visit(node.op)
        rhs = gt_ir.utils.make_expr(self.visit(node.right))
        lhs = gt_ir.utils.make_expr(self.visit(node.left))
        result = gt_ir.BinOpExpr(op=op, lhs=lhs, rhs=rhs)

        return result

    def visit_Add(self, node: ast.Add) -> gt_ir.BinaryOperator:
        return gt_ir.BinaryOperator.ADD

    def visit_Sub(self, node: ast.Sub) -> gt_ir.BinaryOperator:
        return gt_ir.BinaryOperator.SUB

    def visit_Mult(self, node: ast.Mult) -> gt_ir.BinaryOperator:
        return gt_ir.BinaryOperator.MUL

    def visit_Div(self, node: ast.Div) -> gt_ir.BinaryOperator:
        return gt_ir.BinaryOperator.DIV

    def visit_Pow(self, node: ast.Pow) -> gt_ir.BinaryOperator:
        return gt_ir.BinaryOperator.POW

    def visit_And(self, node: ast.And) -> gt_ir.BinaryOperator:
        return gt_ir.BinaryOperator.AND

    def visit_Or(self, node: ast.And) -> gt_ir.BinaryOperator:
        return gt_ir.BinaryOperator.OR

    def visit_Eq(self, node: ast.Eq) -> gt_ir.BinaryOperator:
        return gt_ir.BinaryOperator.EQ

    def visit_NotEq(self, node: ast.NotEq) -> gt_ir.BinaryOperator:
        return gt_ir.BinaryOperator.NE

    def visit_Lt(self, node: ast.Lt) -> gt_ir.BinaryOperator:
        return gt_ir.BinaryOperator.LT

    def visit_LtE(self, node: ast.LtE) -> gt_ir.BinaryOperator:
        return gt_ir.BinaryOperator.LE

    def visit_Gt(self, node: ast.Gt) -> gt_ir.BinaryOperator:
        return gt_ir.BinaryOperator.GT

    def visit_GtE(self, node: ast.GtE) -> gt_ir.BinaryOperator:
        return gt_ir.BinaryOperator.GE

    def visit_BoolOp(self, node: ast.BoolOp) -> gt_ir.BinOpExpr:
        op = self.visit(node.op)
        rhs = gt_ir.utils.make_expr(self.visit(node.values[-1]))
        for value in reversed(node.values[:-1]):
            lhs = gt_ir.utils.make_expr(self.visit(value))
            rhs = gt_ir.BinOpExpr(op=op, lhs=lhs, rhs=rhs)
            res = rhs

        return res

    def visit_Compare(self, node: ast.Compare) -> gt_ir.BinOpExpr:
        lhs = gt_ir.utils.make_expr(self.visit(node.left))
        args = [lhs]

        assert len(node.comparators) >= 1
        op = self.visit(node.ops[-1])
        rhs = gt_ir.utils.make_expr(self.visit(node.comparators[-1]))
        args.append(rhs)

        for i in range(len(node.comparators) - 2, -1, -1):
            lhs = gt_ir.utils.make_expr(self.visit(node.values[i]))
            rhs = gt_ir.BinOpExpr(op=op, lhs=lhs, rhs=rhs)
            op = self.visit(node.ops[i])
            args.append(lhs)

        result = gt_ir.BinOpExpr(op=op, lhs=lhs, rhs=rhs)

        return result

    def visit_IfExp(self, node: ast.IfExp) -> gt_ir.TernaryOpExpr:
        result = gt_ir.TernaryOpExpr(
            condition=gt_ir.utils.make_expr(self.visit(node.test)),
            then_expr=gt_ir.utils.make_expr(self.visit(node.body)),
            else_expr=gt_ir.utils.make_expr(self.visit(node.orelse)),
        )

        return result

    def visit_If(self, node: ast.If) -> list:
        self.if_decls_stack.append([])

        main_stmts = []
        for stmt in node.body:
            main_stmts.extend(gt_utils.listify(self.visit(stmt)))
        assert all(isinstance(item, gt_ir.Statement) for item in main_stmts)

        else_stmts = []
        if node.orelse:
            for stmt in node.orelse:
                else_stmts.extend(gt_utils.listify(self.visit(stmt)))
            assert all(isinstance(item, gt_ir.Statement) for item in else_stmts)

        result = []
        if len(self.if_decls_stack) == 1:
            result.extend(self.if_decls_stack.pop())
        elif len(self.if_decls_stack) > 1:
            self.if_decls_stack[-2].extend(self.if_decls_stack[-1])
            self.if_decls_stack.pop()

        result.append(
            gt_ir.If(
                condition=gt_ir.utils.make_expr(self.visit(node.test)),
                main_body=gt_ir.BlockStmt(stmts=main_stmts),
                else_body=gt_ir.BlockStmt(stmts=else_stmts) if else_stmts else None,
            )
        )

        return result

    def visit_Call(self, node: ast.Call):
        native_fcn = gt_ir.NativeFunction.PYTHON_SYMBOL_TO_IR_OP[node.func.id]

        args = [gt_ir.utils.make_expr(self.visit(arg)) for arg in node.args]
        if len(args) != native_fcn.arity:
            raise GTScriptSyntaxError(
                "Invalid native function call", loc=gt_ir.Location.from_ast_node(node)
            )

        return gt_ir.NativeFuncCall(
            func=native_fcn,
            args=args,
            data_type=gt_ir.DataType.AUTO,
            loc=gt_ir.Location.from_ast_node(node),
        )

    # -- Statement nodes --
    def visit_Assign(self, node: ast.Assign) -> list:
        result = []

        # assert len(node.targets) == 1
        # Create decls for temporary fields
        target = []
        if len(node.targets) > 1:
            raise GTScriptSyntaxError(
                message="Assignment to multiple variables (e.g. var1 = var2 = value) not supported."
            )

        for t in node.targets[0].elts if isinstance(node.targets[0], ast.Tuple) else node.targets:
            if isinstance(t, ast.Subscript):
                if isinstance(t.slice, ast.Index) and (
                    isinstance(t.slice.value, ast.Ellipsis)
                    or (
                        isinstance(t.slice.value, ast.Tuple)
                        and all(v.n == 0 for v in t.slice.value.elts)
                    )
                ):
                    if t.value.id not in {
                        name for name, field in self.fields.items() if field.is_api
                    }:

                        raise GTScriptSyntaxError(
                            message="No subscript allowed in assignment to temporaries",
                            loc=gt_ir.Location.from_ast_node(t),
                        )
                    t = t.value

                else:
                    raise GTScriptSyntaxError(
                        message="Assignment to non-zero offsets is not supported.",
                        loc=gt_ir.Location.from_ast_node(t),
                    )
            if isinstance(t, ast.Name):
                if not self._is_known(t.id):
                    field_decl = gt_ir.FieldDecl(
                        name=t.id,
                        data_type=gt_ir.DataType.AUTO,
                        axes=[ax.name for ax in gt_ir.Domain.LatLonGrid().axes],
                        # layout_id=t.id,
                        is_api=False,
                    )
                    if len(self.if_decls_stack):
                        self.if_decls_stack[-1].append(field_decl)
                    else:
                        result.append(field_decl)
                    self.fields[field_decl.name] = field_decl
            else:
                raise GTScriptSyntaxError(message="Invalid target in assignment.", loc=target)

            target.append(self.visit(t))

        value = self.visit(node.value)
        if len(target) == 1:
            value = [gt_ir.utils.make_expr(value)]
        else:
            value = [gt_ir.utils.make_expr(item) for item in value]

        assert len(target) == len(value)
        for left, right in zip(target, value):
            result.append(gt_ir.Assign(target=left, value=right))

        return result

    def visit_AugAssign(self, node: ast.AugAssign):
        """Implement left <op>= right in terms of left = left <op> right."""
        binary_operation = ast.BinOp(left=node.target, op=node.op, right=node.value)
        assignment = ast.Assign(targets=[node.target], value=binary_operation)
        ast.copy_location(binary_operation, node)
        ast.copy_location(assignment, node)
        return self.visit_Assign(assignment)

    def visit_With(self, node: ast.With):
        loc = gt_ir.Location.from_ast_node(node)
        syntax_error = GTScriptSyntaxError(
            f"Invalid 'with' statement at line {loc.line} (column {loc.column})", loc=loc
        )

        # If we find nested `with` blocks flatten them, i.e. transform
        #  with computation(PARALLEL):
        #   with interval(...):
        #     ...
        # into
        #  with computation(PARALLEL), interval(...):
        #    ...
        # otherwise just parse the node
        if self.parsing_context == ParsingContext.CONTROL_FLOW and all(
            isinstance(child_node, ast.With) for child_node in node.body
        ):
            # Ensure top level `with` specifies the iteration order
            if not any(
                with_item.context_expr.func.id == "computation"
                for with_item in node.items
                if isinstance(with_item.context_expr, ast.Call)
            ):
                raise syntax_error

            # Parse nested `with` blocks
            compute_blocks = []
            for with_node in node.body:
                with_node = copy.deepcopy(with_node)  # Copy to avoid altering original ast
                # Splice `withItems` of current/primary with statement into nested with
                with_node.items.extend(node.items)

                compute_blocks.append(self._visit_computation_node(with_node))

            # Validate block specification order
            #  the nested computation blocks must be specified in their order of execution. The order of execution is
            #  such that the lowest (highest) interval is processed first if the iteration order is forward (backward).
            if not self._are_blocks_sorted(compute_blocks):
                raise GTScriptSyntaxError(
                    f"Invalid 'with' statement at line {loc.line} (column {loc.column}). Intervals must be specified in order of execution."
                )

            return compute_blocks
        elif self.parsing_context == ParsingContext.CONTROL_FLOW and not any(
            isinstance(child_node, ast.With) for child_node in node.body
        ):
            return self._visit_computation_node(node)
        else:
            # Mixing nested `with` blocks with stmts not allowed
            raise syntax_error

    def visit_FunctionDef(self, node: ast.FunctionDef) -> list:
        blocks = []
        docstring = ast.get_docstring(node)
        for stmt in node.body:
            blocks.extend(gt_utils.listify(self.visit(stmt)))

        if not all(isinstance(item, gt_ir.ComputationBlock) for item in blocks):
            raise GTScriptSyntaxError(
                "Invalid stencil definition", loc=gt_ir.Location.from_ast_node(node)
            )

        return blocks


class CollectLocalSymbolsAstVisitor(ast.NodeVisitor):
    def __call__(self, node: ast.FunctionDef):
        self.local_symbols = set()
        self.visit(node)
        result = self.local_symbols
        del self.local_symbols
        return result

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            targets = target.elts if isinstance(target, ast.Tuple) else [target]
            for t in targets:
                if isinstance(t, ast.Name):
                    self.local_symbols.add(t.id)
                elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                    self.local_symbols.add(t.value.id)
                else:
                    raise GTScriptSyntaxError(
                        message="invalid target in assign", loc=gt_ir.Location.from_ast_node(node)
                    )


class GTScriptParser(ast.NodeVisitor):

    CONST_VALUE_TYPES = (
        *gtscript._VALID_DATA_TYPES,
        types.FunctionType,
        type(None),
        gtscript._AxisOffset,
    )

    def __init__(self, definition, *, options, externals=None):
        assert isinstance(definition, types.FunctionType)
        self.definition = definition
        self.filename = inspect.getfile(definition)
        self.source, decorators_source = gt_meta.split_def_decorators(self.definition)
        self.ast_root = ast.parse(self.source)
        self.options = options
        self.build_info = options.build_info
        self.main_name = options.qualified_name
        self.definition_ir = None
        self.external_context = externals or {}
        self.resolved_externals = {}
        self.block = None

    def __str__(self):
        result = "<GT4Py.GTScriptParser> {\n"
        result += "\n".join("\t{}: {}".format(name, getattr(self, name)) for name in vars(self))
        result += "\n}"
        return result

    @staticmethod
    def annotate_definition(definition):
        api_signature = []
        api_annotations = []

        qualified_name = "{}.{}".format(definition.__module__, definition.__name__)
        sig = inspect.signature(definition)
        for param in sig.parameters.values():
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                raise GTScriptDefinitionError(
                    name=qualified_name,
                    value=definition,
                    message="'*args' tuple parameter is not supported in GTScript definitions",
                )
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                raise GTScriptDefinitionError(
                    name=qualified_name,
                    value=definition,
                    message="'*kwargs' dict parameter is not supported in GTScript definitions",
                )
            else:
                is_keyword = param.kind == inspect.Parameter.KEYWORD_ONLY

                default = gt_ir.Empty
                if param.default != inspect.Parameter.empty:
                    if not isinstance(param.default, GTScriptParser.CONST_VALUE_TYPES):
                        raise GTScriptValueError(
                            name=param.name,
                            value=param.default,
                            message=f"Invalid default value for argument '{param.name}': {param.default}",
                        )
                    default = param.default

                if isinstance(param.annotation, (str, gtscript._FieldDescriptor)):
                    dtype_annotation = param.annotation
                elif (
                    isinstance(param.annotation, type)
                    and param.annotation in gtscript._VALID_DATA_TYPES
                ):
                    dtype_annotation = np.dtype(param.annotation)
                elif param.annotation is inspect.Signature.empty:
                    dtype_annotation = None
                else:
                    raise GTScriptValueError(
                        name=param.name,
                        value=param.annotation,
                        message=f"Invalid annotated dtype value for argument '{param.name}': {param.annotation}",
                    )

                api_signature.append(
                    gt_ir.ArgumentInfo(name=param.name, is_keyword=is_keyword, default=default)
                )

                api_annotations.append(dtype_annotation)

        nonlocal_symbols, imported_symbols = GTScriptParser.collect_external_symbols(definition)
        canonical_ast = gt_meta.ast_dump(definition)

        definition._gtscript_ = dict(
            qualified_name=qualified_name,
            api_signature=api_signature,
            api_annotations=api_annotations,
            canonical_ast=canonical_ast,
            nonlocals=nonlocal_symbols,
            imported=imported_symbols,
        )

        return definition

    @staticmethod
    def collect_external_symbols(definition):
        bare_imports, from_imports, relative_imports = gt_meta.collect_imported_symbols(definition)
        wrong_imports = list(bare_imports.keys()) + list(relative_imports.keys())
        imported_names = set()
        for key, value in from_imports.items():
            if key != value:
                # Aliasing imported names is not allowed
                wrong_imports.append(key)
            else:
                for prefix in [
                    "__externals__.",
                    "gt4py.__externals__.",
                    "__gtscript__.",
                    "gt4py.__gtscript__.",
                ]:
                    if key.startswith(prefix):
                        if "__externals__" in key:
                            imported_names.add(value.replace(prefix, "", 1))
                        break
                else:
                    wrong_imports.append(key)

        if wrong_imports:
            raise GTScriptSyntaxError("Invalid 'import' statements ({})".format(wrong_imports))

        imported_symbols = {name: {} for name in imported_names}

        context, unbound = gt_meta.get_closure(
            definition, included_nonlocals=True, include_builtins=False
        )

        gtscript_ast = ast.parse(gt_meta.get_ast(definition)).body[0]
        local_symbol_collector = CollectLocalSymbolsAstVisitor()
        local_symbols = local_symbol_collector(gtscript_ast)

        nonlocal_symbols = {}

        name_nodes = gt_meta.collect_names(definition)
        for collected_name in name_nodes.keys():
            if collected_name not in gtscript.builtins:
                root_name = collected_name.split(".")[0]
                if root_name in imported_symbols:
                    imported_symbols[root_name].setdefault(
                        collected_name, name_nodes[collected_name]
                    )
                elif root_name in context:
                    nonlocal_symbols[collected_name] = GTScriptParser.eval_external(
                        collected_name,
                        context,
                        gt_ir.Location.from_ast_node(name_nodes[collected_name][0]),
                    )
                elif root_name not in local_symbols and root_name in unbound:
                    raise GTScriptSymbolError(
                        name=collected_name,
                        loc=gt_ir.Location.from_ast_node(name_nodes[collected_name][0]),
                    )

        return nonlocal_symbols, imported_symbols

    @staticmethod
    def eval_external(name: str, context: dict, loc=None):
        try:
            value = eval(name, context)
            if isinstance(value, types.FunctionType) and not hasattr(value, "_gtscript_"):
                GTScriptParser.annotate_definition(value)

            assert (
                value is None
                or isinstance(value, GTScriptParser.CONST_VALUE_TYPES)
                or hasattr(value, "_gtscript_")
            )

        except Exception as e:
            raise GTScriptDefinitionError(
                name=name,
                value="<unknown>",
                message="Missing or invalid value for external symbol {name}".format(name=name),
                loc=loc,
            ) from e
        return value

    @staticmethod
    def resolve_external_symbols(
        nonlocals: dict, imported: dict, context: dict, *, exhaustive=True
    ):
        result = {}
        accepted_imports = set(imported.keys())
        resolved_imports = {**imported}
        resolved_values_list = list(nonlocals.items())

        # Resolve function-like imports
        func_externals = {
            key: value
            for key, value in itertools.chain(context.items(), resolved_values_list)
            if isinstance(value, types.FunctionType)
        }
        for name, value in func_externals.items():
            if isinstance(value, types.FunctionType) and not hasattr(value, "_gtscript_"):
                GTScriptParser.annotate_definition(value)
            for imported_name, imported_value in value._gtscript_["imported"].items():
                resolved_imports[imported_name] = imported_value

        # Collect all imported and inlined values recursively through all the external symbols
        while resolved_imports or resolved_values_list:
            new_imports = {}
            for name, accesses in resolved_imports.items():
                if accesses:
                    for attr_name, attr_nodes in accesses.items():
                        resolved_values_list.append(
                            (
                                attr_name,
                                GTScriptParser.eval_external(
                                    attr_name, context, gt_ir.Location.from_ast_node(attr_nodes[0])
                                ),
                            )
                        )

                elif not exhaustive:
                    resolved_values_list.append(
                        (name, GTScriptParser.eval_external(name, context))
                    )

            for name, value in resolved_values_list:
                if hasattr(value, "_gtscript_") and exhaustive:
                    assert callable(value)
                    nested_inlined_values = {
                        "{}.{}".format(value._gtscript_["qualified_name"], item_name): item_value
                        for item_name, item_value in value._gtscript_["nonlocals"].items()
                    }
                    resolved_values_list.extend(nested_inlined_values.items())

                    for imported_name, imported_name_accesses in value._gtscript_[
                        "imported"
                    ].items():
                        if imported_name in accepted_imports:
                            # Only check names explicitly imported in the main caller context
                            new_imports.setdefault(imported_name, {})
                            for attr_name, attr_nodes in imported_name_accesses.items():
                                new_imports[imported_name].setdefault(attr_name, [])
                                new_imports[imported_name][attr_name].extend(attr_nodes)

            result.update(dict(resolved_values_list))
            resolved_imports = new_imports
            resolved_values_list = []

        return result

    def extract_arg_descriptors(self):
        api_signature = self.definition._gtscript_["api_signature"]
        api_annotations = self.definition._gtscript_["api_annotations"]
        assert len(api_signature) == len(api_annotations)
        fields_decls, parameter_decls = {}, {}

        for arg_info, arg_annotation in zip(api_signature, api_annotations):
            try:
                assert arg_annotation in gtscript._VALID_DATA_TYPES or isinstance(
                    arg_annotation, (gtscript._SequenceDescriptor, gtscript._FieldDescriptor)
                ), "Invalid parameter annotation"

                if arg_annotation in gtscript._VALID_DATA_TYPES:
                    dtype = np.dtype(arg_annotation)
                    if arg_info.default not in [gt_ir.Empty, None]:
                        assert np.dtype(type(arg_info.default)) == dtype
                    data_type = gt_ir.DataType.from_dtype(dtype)
                    parameter_decls[arg_info.name] = gt_ir.VarDecl(
                        name=arg_info.name, data_type=data_type, length=0, is_api=True
                    )
                elif isinstance(arg_annotation, gtscript._SequenceDescriptor):
                    assert arg_info.default in [gt_ir.Empty, None]
                    data_type = gt_ir.DataType.from_dtype(np.dtype(arg_annotation))
                    length = arg_annotation.length
                    parameter_decls[arg_info.name] = gt_ir.VarDecl(
                        name=arg_info.name, data_type=data_type, length=length, is_api=True
                    )
                else:
                    assert isinstance(arg_annotation, gtscript._FieldDescriptor)
                    assert arg_info.default in [gt_ir.Empty, None]
                    data_type = gt_ir.DataType.from_dtype(np.dtype(arg_annotation.dtype))
                    axes = [ax.name for ax in arg_annotation.axes]
                    fields_decls[arg_info.name] = gt_ir.FieldDecl(
                        name=arg_info.name,
                        data_type=data_type,
                        axes=axes,
                        is_api=True,
                        layout_id=arg_info.name,
                    )

                if data_type is gt_ir.DataType.INVALID:
                    raise GTScriptDataTypeError(name=arg_info.name, data_type=data_type)

            except Exception as e:
                raise GTScriptDefinitionError(
                    name=arg_info.name,
                    value=arg_annotation,
                    message=f"Invalid definition of argument '{arg_info.name}': {arg_annotation}",
                ) from e

        for item in itertools.chain(fields_decls.values(), parameter_decls.values()):
            if item.data_type is gt_ir.DataType.INVALID:
                raise GTScriptDataTypeError(name=item.name, data_type=item.data_type)

        return api_signature, fields_decls, parameter_decls

    # def eval_type(self, expr: gt_ir.Expr):
    #     kind = gt_ir.SymbolKind.SCALAR
    #     sctype = gt_ir.DataType.DEFAULT
    #     for ref in gt_ir.refs_from(expr):
    #         if isinstance(ref, gt_ir.Ref):
    #             ref = self.scope[ref.name]
    #         if isinstance(ref, gt_ir.Decl):
    #             if ref.kind == gt_ir.SymbolKind.FIELD:
    #                 kind = gt_ir.SymbolKind.FIELD
    #
    #         sctype = gt_ir.ScalarType.merge(sctype, ref.sctype)
    #
    #     return (kind, sctype)

    def run(self):
        assert (
            isinstance(self.ast_root, ast.Module)
            and "body" in self.ast_root._fields
            and len(self.ast_root.body) == 1
            and isinstance(self.ast_root.body[0], ast.FunctionDef)
        )
        main_func_node = self.ast_root.body[0]

        assert hasattr(self.definition, "_gtscript_")
        # self.resolved_externals = self.resolve_external_symbols(
        #     self.definition._gtscript_["nonlocals"],
        #     self.definition._gtscript_["imported"],
        #     self.external_context,
        # )
        self.resolved_externals = self.definition._gtscript_["externals"]
        api_signature, fields_decls, parameter_decls = self.extract_arg_descriptors()

        # Inline constant values
        for name, value in self.resolved_externals.items():
            if hasattr(value, "_gtscript_"):
                assert callable(value)
                func_node = ast.parse(gt_meta.get_ast(value)).body[0]
                local_context = self.resolve_external_symbols(
                    value._gtscript_["nonlocals"],
                    value._gtscript_["imported"],
                    self.external_context,
                    exhaustive=False,
                )
                ValueInliner.apply(func_node, context=local_context)
                value._gtscript_["ast"] = func_node
                value._gtscript_["local_context"] = local_context

        local_context = self.resolve_external_symbols(
            self.definition._gtscript_["nonlocals"],
            self.definition._gtscript_["imported"],
            self.external_context,
            exhaustive=False,
        )
        AssertionChecker.apply(main_func_node, context=local_context, source=self.source)

        ValueInliner.apply(main_func_node, context=local_context)

        # Inline function calls
        CallInliner.apply(main_func_node, context=local_context)

        # Evaluate and inline compile-time conditionals
        CompiledIfInliner.apply(main_func_node, context=local_context)
        # Cleaner.apply(self.definition_ir)

        # Generate definition IR
        domain = gt_ir.Domain.LatLonGrid()
        computations = IRMaker(
            fields=fields_decls,
            parameters=parameter_decls,
            local_symbols={},  # Not used
            domain=domain,
            extra_temp_decls={},  # Not used
        )(self.ast_root)

        self.definition_ir = gt_ir.StencilDefinition(
            name=self.main_name,
            domain=domain,
            api_signature=api_signature,
            api_fields=[
                fields_decls[item.name] for item in api_signature if item.name in fields_decls
            ],
            parameters=[
                parameter_decls[item.name]
                for item in api_signature
                if item.name in parameter_decls
            ],
            computations=computations,
            externals=self.resolved_externals,
            docstring=inspect.getdoc(self.definition) or "",
            loc=gt_ir.Location.from_ast_node(self.ast_root.body[0]),
        )

        return self.definition_ir


@gt_frontend.register
class GTScriptFrontend(gt_frontend.Frontend):
    name = "gtscript"

    @classmethod
    def get_stencil_id(cls, qualified_name, definition, externals, options_id):
        cls.prepare_stencil_definition(definition, externals or {})
        fingerprint = {
            "__main__": definition._gtscript_["canonical_ast"],
            "docstring": inspect.getdoc(definition),
            "api_annotations": f"[{', '.join(str(item) for item in definition._gtscript_['api_annotations'])}]",
        }
        for name, value in definition._gtscript_["externals"].items():
            fingerprint[name] = (
                value._gtscript_["canonical_ast"] if hasattr(value, "_gtscript_") else value
            )

        definition_id = gt_utils.shashed_id(fingerprint)
        version = gt_utils.shashed_id(definition_id, options_id)
        stencil_id = gt_definitions.StencilID(qualified_name, version)

        return stencil_id

    @classmethod
    def prepare_stencil_definition(cls, definition, externals):
        GTScriptParser.annotate_definition(definition)
        resolved_externals = GTScriptParser.resolve_external_symbols(
            definition._gtscript_["nonlocals"], definition._gtscript_["imported"], externals
        )
        definition._gtscript_["externals"] = resolved_externals
        return definition

    @classmethod
    def generate(cls, definition, externals, options):
        if not hasattr(definition, "_gtscript_"):
            cls.prepare_stencil_definition(definition, externals)
        translator = GTScriptParser(definition, externals=externals, options=options)
        return translator.run()
