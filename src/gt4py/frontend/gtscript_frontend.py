# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2022, ETH Zurich
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
import time
import types
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from gt4py import definitions as gt_definitions
from gt4py import gtscript
from gt4py import utils as gt_utils
from gt4py.frontend import node_util, nodes
from gt4py.frontend.defir_to_gtir import DefIRToGTIR, UnrollVectorAssignments
from gt4py.utils import NOTHING
from gt4py.utils import meta as gt_meta
from gtc import utils as gtc_utils

from .base import Frontend, register
from .exceptions import (
    GTScriptAssertionError,
    GTScriptDataTypeError,
    GTScriptDefinitionError,
    GTScriptSymbolError,
    GTScriptSyntaxError,
    GTScriptValueError,
)


class AssertionChecker(ast.NodeTransformer):
    """Check assertions and remove from the AST for further parsing."""

    @classmethod
    def apply(cls, func_node: ast.FunctionDef, context: Dict[str, Any], source: str):
        checker = cls(context, source)
        checker.visit(func_node)

    def __init__(self, context: Dict[str, Any], source: str):
        self.context = context
        self.source = source

    def _process_assertion(self, expr_node: ast.Expr) -> None:
        condition_value = gt_utils.meta.ast_eval(expr_node, self.context, default=NOTHING)
        if condition_value is not NOTHING:
            if not condition_value:
                source_lines = textwrap.dedent(self.source).split("\n")
                loc = nodes.Location.from_ast_node(expr_node)
                raise GTScriptAssertionError(source_lines[loc.line - 1], loc=loc)
        else:
            raise GTScriptSyntaxError(
                "Evaluation of compile_assert condition failed at the preprocessing step."
            )
        return None

    def _process_call(self, node: ast.Call) -> Optional[ast.Call]:
        name = gt_meta.get_qualified_name_from_node(node.func)
        if name != "compile_assert":
            return node
        else:
            if len(node.args) != 1:
                raise GTScriptSyntaxError(
                    "Invalid assertion. Correct syntax: compile_assert(condition)"
                )
            return self._process_assertion(node.args[0])

    def visit_Expr(self, node: ast.Expr) -> Optional[ast.AST]:
        if isinstance(node.value, ast.Call):
            ret = self._process_call(node.value)
            return ast.Expr(value=ret) if ret else None
        else:
            return node


class AxisIntervalParser(gt_meta.ASTPass):
    """Parse Python AST interval syntax in the form of a Slice.

    Corner cases: `ast.Ellipsis` refers to the entire interval, and
    if an `ast.Subscript` is passed, this parses its slice attribute.

    """

    @classmethod
    def apply(
        cls,
        node: Union[ast.Ellipsis, ast.Slice, ast.Subscript, ast.Constant],
        axis_name: str,
        loc: Optional[nodes.Location] = None,
    ) -> nodes.AxisInterval:
        parser = cls(axis_name, loc)

        if isinstance(node, ast.Ellipsis):
            interval = nodes.AxisInterval.full_interval()
            interval.loc = loc
            return interval

        if isinstance(node, ast.Slice):
            slice_node = node
        elif isinstance(getattr(node, "slice", None), ast.Slice):
            slice_node = node.slice
        else:
            slice_node = cls.slice_from_value(node)

        if slice_node.lower is None:
            slice_node.lower = ast.Constant(value=None)

        if (
            isinstance(slice_node.lower, ast.Constant)
            and slice_node.lower.value is None
            and axis_name == nodes.Domain.LatLonGrid().sequential_axis.name
        ):
            raise parser.interval_error

        if slice_node.upper is None:
            slice_node.upper = ast.Constant(value=None)

        lower = parser.visit(slice_node.lower)
        upper = parser.visit(slice_node.upper)

        start = parser._make_axis_bound(lower, nodes.LevelMarker.START)
        end = parser._make_axis_bound(upper, nodes.LevelMarker.END)

        return nodes.AxisInterval(start=start, end=end, loc=loc)

    def __init__(self, axis_name: str, loc: Optional[nodes.Location] = None):
        self.axis_name = axis_name
        self.loc = loc

        error_msg = "Invalid interval range specification"

        if self.loc is not None:
            error_msg = f"{error_msg} at line {loc.line} (column: {loc.column})"

        self.interval_error = GTScriptSyntaxError(error_msg)

    @staticmethod
    def slice_from_value(node: ast.Expr) -> ast.Slice:
        """Create an ast.Slice node from a general ast.Expr node."""
        slice_node = ast.Slice(
            lower=node, upper=ast.BinOp(left=node, op=ast.Add(), right=ast.Constant(value=1))
        )
        slice_node = ast.copy_location(slice_node, node)
        return slice_node

    def _make_axis_bound(
        self,
        value: Union[int, None, gtscript.AxisIndex, nodes.AxisBound, nodes.VarRef],
        endpt: nodes.LevelMarker,
    ) -> nodes.AxisBound:
        if isinstance(value, nodes.AxisBound):
            return value
        else:
            if isinstance(value, int):
                level = nodes.LevelMarker.END if value < 0 else nodes.LevelMarker.START
                offset = value
            elif isinstance(value, nodes.VarRef):
                level = value
                offset = 0
            elif isinstance(value, gtscript.AxisIndex):
                level = nodes.LevelMarker.START if value.index >= 0 else nodes.LevelMarker.END
                offset = value.index + value.offset
            elif value is None:
                LARGE_NUM = 10000
                seq_name = nodes.Domain.LatLonGrid().sequential_axis.name
                level = endpt
                if self.axis_name == seq_name:
                    offset = 0
                else:
                    offset = -LARGE_NUM if level == nodes.LevelMarker.START else LARGE_NUM
            else:
                raise self.interval_error

            return nodes.AxisBound(level=level, offset=offset, loc=self.loc)

    def visit_Name(self, node: ast.Name) -> nodes.VarRef:
        return nodes.VarRef(name=node.id, loc=nodes.Location.from_ast_node(node))

    def visit_Constant(self, node: ast.Constant) -> Union[int, gtscript.AxisIndex, None]:
        if isinstance(node.value, gtscript.AxisIndex):
            return node.value
        elif isinstance(node.value, numbers.Number):
            return int(node.value)
        elif node.value is None:
            return None
        else:
            raise GTScriptSyntaxError(
                f"Unexpected type found {type(node.value)}. Expected one of: int, AxisIndex, string (var ref), or None.",
                loc=self.loc,
            )

    def visit_BinOp(self, node: ast.BinOp) -> Union[gtscript.AxisIndex, nodes.AxisBound, int]:
        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(node.op, ast.Add):
            bin_op = lambda x, y: x + y  # noqa: E731
            u_op = lambda x: x  # noqa: E731
        elif isinstance(node.op, ast.Sub):
            bin_op = lambda x, y: x - y  # noqa: E731
            u_op = lambda x: -x  # noqa: E731
        elif isinstance(node.op, ast.Mult):
            if left.level != right.level or not isinstance(left.level, nodes.LevelMarker):
                raise self.interval_error
            bin_op = lambda x, y: x * y  # noqa: E731
            u_op = None
        else:
            raise GTScriptSyntaxError("Unexpected binary operator found in interval expression")

        incompatible_types_error = GTScriptSyntaxError(
            "Incompatible types found in interval expression"
        )

        if isinstance(left, gtscript.AxisIndex):
            if not isinstance(right, numbers.Number):
                raise incompatible_types_error
            return gtscript.AxisIndex(
                axis=left.axis, index=left.index, offset=bin_op(left.offset, right)
            )
        elif isinstance(left, nodes.VarRef):
            if not isinstance(right, numbers.Number):
                raise incompatible_types_error
            return nodes.AxisBound(level=left, offset=u_op(right), loc=self.loc)
        elif isinstance(left, nodes.AxisBound):
            if not isinstance(right, numbers.Number):
                raise incompatible_types_error
            return nodes.AxisBound(
                level=left.level, offset=bin_op(left.offset, right), loc=self.loc
            )
        elif isinstance(left, numbers.Number) and isinstance(right, numbers.Number):
            return bin_op(left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> nodes.AxisBound:
        if isinstance(node.op, ast.USub):
            op = lambda x: -x  # noqa: E731
        else:
            raise self.interval_error

        value = self.visit(node.operand)
        if isinstance(value, numbers.Number):
            return op(value)
        else:
            raise self.interval_error

    def visit_Subscript(self, node: ast.Subscript) -> nodes.AxisBound:
        if node.value.id != self.axis_name:
            raise self.interval_error

        if isinstance(node.slice, ast.Index):
            index = self.visit(node.slice.value)
        else:
            index = self.visit(node.slice)

        return gtscript.AxisIndex(axis=self.axis_name, index=index)


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
            if value is None or isinstance(value, (bool, numbers.Number, gtscript.AxisIndex)):
                new_node = ast.Constant(value=value)
            elif hasattr(value, "_gtscript_"):
                pass
            else:
                raise ValueError(f"Failed to inline {qualified_name}.")
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
        self.all_skip_names = set(gtscript.builtins) | {"gt4py", "gtscript"}

    def __call__(self, func_node: ast.FunctionDef):
        self.current_name = func_node.name
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

    def visit_Assert(self, node: ast.Assert):
        # Assertions are removed in the AssertionChecker later.
        return node

    def visit_Assign(self, node: ast.Assign):
        if (
            isinstance(node.value, ast.Call)
            and gt_meta.get_qualified_name_from_node(node.value.func) not in gtscript.MATH_BUILTINS
        ):
            assert len(node.targets) == 1
            self.visit(node.value, target_node=node.targets[0])
            # This node can be now removed since the trivial assignment has been already done
            # in the Call visitor
            return None
        else:
            return self.generic_visit(node)

    def visit_Call(  # noqa: C901 # Cyclomatic complexity too high
        self, node: ast.Call, *, target_node=None
    ):
        call_name = gt_meta.get_qualified_name_from_node(node.func)

        if call_name in gtscript.IGNORE_WHEN_INLINING:
            # Not a function to inline. Visit arguments and return as-is.
            node.args = [self.visit(arg) for arg in node.args]
            return node
        elif any(
            isinstance(arg, ast.Call) and arg.func.id not in gtscript.MATH_BUILTINS
            for arg in node.args
        ):
            raise GTScriptSyntaxError(
                "Function calls are not supported in arguments to function calls",
                loc=nodes.Location.from_ast_node(node),
            )
        elif call_name not in self.context or not hasattr(self.context[call_name], "_gtscript_"):
            raise GTScriptSyntaxError("Unknown call", loc=nodes.Location.from_ast_node(node))

        # Recursively inline any possible nested subroutine call
        call_info = self.context[call_name]._gtscript_
        call_ast = copy.deepcopy(call_info["ast"])
        self.current_name = call_name
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
                    assert arg_infos[name] != nodes.Empty
                    call_args[name] = ast.Constant(value=arg_infos[name])
        except Exception:
            raise GTScriptSyntaxError(
                message="Invalid call signature", loc=nodes.Location.from_ast_node(node)
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

        # Add the temp_annotations and temp_init_values to the parent
        current_info = self.context[self.current_name]._gtscript_
        for name, desc in call_info["temp_annotations"].items():
            new_name = name_mapping[name]
            current_info["temp_annotations"][new_name] = desc

        for name, value in call_info["temp_init_values"].items():
            new_name = name_mapping[name]
            current_info["temp_init_values"][new_name] = value

        return result_node

    def visit_Expr(self, node: ast.Expr):
        """Ignore pure string statements in callee."""
        if not isinstance(node.value, (ast.Constant, ast.Str)):
            return super().visit(node.value)


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


def _make_temp_decls(
    descriptors: Dict[str, gtscript._FieldDescriptor]
) -> Dict[str, nodes.FieldDecl]:
    return {
        name: nodes.FieldDecl(
            name=name,
            data_type=nodes.DataType.from_dtype(desc.dtype),
            axes=[str(ax) for ax in desc.axes],
            is_api=False,
            data_dims=list(desc.data_dims),
        )
        for name, desc in descriptors.items()
    }


def _make_init_computations(
    temp_decls: Dict[str, nodes.FieldDecl], init_values: Dict[str, Any], func_node: ast.AST
) -> List[nodes.ComputationBlock]:
    if not temp_decls:
        return []

    stmts: List[nodes.Assign] = []
    for name in init_values:
        decl = temp_decls[name]
        stmts.append(decl)
        if decl.data_dims:
            for index in itertools.product(*(range(i) for i in decl.data_dims)):
                literal_index = [
                    nodes.ScalarLiteral(value=i, data_type=nodes.DataType.INT32) for i in index
                ]
                stmts.append(
                    nodes.Assign(
                        target=nodes.FieldRef.at_center(
                            name, axes=decl.axes, data_index=literal_index
                        ),
                        value=nodes.ScalarLiteral(
                            value=init_values[name], data_type=decl.data_type
                        ),
                    )
                )
        else:
            stmts.append(
                nodes.Assign(
                    target=nodes.FieldRef.at_center(name, axes=decl.axes),
                    value=init_values[name],
                )
            )

    return [
        nodes.ComputationBlock(
            interval=nodes.AxisInterval.full_interval(),
            iteration_order=nodes.IterationOrder.PARALLEL,
            body=nodes.BlockStmt(stmts=stmts),
        )
    ]


def _find_accesses_with_offsets(node: nodes.Node) -> Set[str]:
    names: Set[str] = set()

    class FindRefs(node_util.IRNodeVisitor):
        def visit_FieldRef(self, node: nodes.FieldRef) -> None:
            if node.offset.get("I", 0) != 0 or node.offset.get("J", 0) != 0:
                names.add(node.name)

    FindRefs().visit(node)
    return names


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
        domain: nodes.Domain,
        temp_decls: Optional[Dict[str, nodes.FieldDecl]] = None,
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
        self.domain = domain or nodes.Domain.LatLonGrid()
        self.temp_decls = temp_decls or {}
        self.parsing_context = None
        self.iteration_order = None
        self.decls_stack = []
        self.parsing_horizontal_region = False
        self.written_vars: Set[str] = set()
        nodes.NativeFunction.PYTHON_SYMBOL_TO_IR_OP = {
            "abs": nodes.NativeFunction.ABS,
            "min": nodes.NativeFunction.MIN,
            "max": nodes.NativeFunction.MAX,
            "mod": nodes.NativeFunction.MOD,
            "sin": nodes.NativeFunction.SIN,
            "cos": nodes.NativeFunction.COS,
            "tan": nodes.NativeFunction.TAN,
            "asin": nodes.NativeFunction.ARCSIN,
            "acos": nodes.NativeFunction.ARCCOS,
            "atan": nodes.NativeFunction.ARCTAN,
            "sinh": nodes.NativeFunction.SINH,
            "cosh": nodes.NativeFunction.COSH,
            "tanh": nodes.NativeFunction.TANH,
            "asinh": nodes.NativeFunction.ARCSINH,
            "acosh": nodes.NativeFunction.ARCCOSH,
            "atanh": nodes.NativeFunction.ARCTANH,
            "sqrt": nodes.NativeFunction.SQRT,
            "exp": nodes.NativeFunction.EXP,
            "log": nodes.NativeFunction.LOG,
            "gamma": nodes.NativeFunction.GAMMA,
            "cbrt": nodes.NativeFunction.CBRT,
            "isfinite": nodes.NativeFunction.ISFINITE,
            "isinf": nodes.NativeFunction.ISINF,
            "isnan": nodes.NativeFunction.ISNAN,
            "floor": nodes.NativeFunction.FLOOR,
            "ceil": nodes.NativeFunction.CEIL,
            "trunc": nodes.NativeFunction.TRUNC,
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

    def _are_blocks_sorted(self, compute_blocks: List[nodes.ComputationBlock]):
        def sort_blocks_key(comp_block):
            start = comp_block.interval.start
            assert isinstance(start.level, nodes.LevelMarker)
            key = 0 if start.level == nodes.LevelMarker.START else 100000
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
            reverse=iteration_order == nodes.IterationOrder.BACKWARD,
        )

        # if sorting didn't change anything it was already sorted
        return compute_blocks == compute_blocks_sorted

    def _parse_region_intervals(
        self, node: Union[ast.ExtSlice, ast.Index, ast.Tuple], loc: nodes.Location = None
    ) -> Dict[str, nodes.AxisInterval]:
        if isinstance(node, ast.Index):
            # Python 3.8 wraps a Tuple in an Index for region[0, 1]
            tuple_node = node.value
            list_of_exprs = tuple_node.elts
        elif isinstance(node, ast.ExtSlice) or isinstance(node, ast.Tuple):
            # Python 3.8 returns an ExtSlice for region[0, :]
            # Python 3.9 directly returns a Tuple for region[0, 1]
            node_list = node.dims if isinstance(node, ast.ExtSlice) else node.elts
            list_of_exprs = [
                axis_node.value if isinstance(axis_node, ast.Index) else axis_node
                for axis_node in node_list
            ]
        else:
            raise GTScriptSyntaxError(
                f"Invalid 'region' index at line {loc.line} (column {loc.column})", loc=loc
            )
        axes_names = [axis.name for axis in self.domain.parallel_axes]
        return {
            name: AxisIntervalParser.apply(axis_node, name)
            for axis_node, name in zip(list_of_exprs, axes_names)
        }

    def _visit_with_horizontal(
        self, node: ast.withitem, loc: nodes.Location
    ) -> List[Dict[str, nodes.AxisInterval]]:
        syntax_error = GTScriptSyntaxError(
            f"Invalid 'with' statement at line {loc.line} (column {loc.column})", loc=loc
        )

        call_args = node.context_expr.args
        if any(not isinstance(arg, ast.Subscript) for arg in call_args) or any(
            arg.value.id != "region" for arg in call_args
        ):
            raise syntax_error

        return [self._parse_region_intervals(arg.slice, loc) for arg in call_args]

    def _are_intervals_nonoverlapping(self, compute_blocks: List[nodes.ComputationBlock]):
        for i, block in enumerate(compute_blocks[1:]):
            other = compute_blocks[i]
            if not block.interval.disjoint_from(other.interval):
                return False
        return True

    def _visit_iteration_order_node(self, node: ast.withitem, loc: nodes.Location):
        syntax_error = GTScriptSyntaxError(
            f"Invalid 'computation' specification at line {loc.line} (column {loc.column})", loc=loc
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
            or iteration_order_node.id not in nodes.IterationOrder.__members__
        ):
            raise syntax_error

        self.iteration_order = nodes.IterationOrder[iteration_order_node.id]

        return self.iteration_order

    def _visit_interval_node(self, node: ast.withitem, loc: nodes.Location):
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
                    "Two-argument syntax should not use AxisIndexs or AxisIntervals"
                )
            interval_node = ast.Slice(lower=args[0], upper=args[1])
            ast.copy_location(interval_node, node)
        else:
            interval_node = args[0]

        seq_name = nodes.Domain.LatLonGrid().sequential_axis.name
        interval = AxisIntervalParser.apply(interval_node, seq_name, loc=loc)

        if (
            interval.start.level == nodes.LevelMarker.END
            and interval.end.level == nodes.LevelMarker.START
        ) or (
            interval.start.level == interval.end.level
            and interval.end.offset <= interval.start.offset
        ):
            raise range_error

        return interval

    def _visit_computation_node(self, node: ast.With) -> nodes.ComputationBlock:
        loc = nodes.Location.from_ast_node(node)
        syntax_error = GTScriptSyntaxError(
            f"Invalid 'computation' specification at line {loc.line} (column {loc.column})", loc=loc
        )

        # Parse computation specification, i.e. `withItems` nodes
        iteration_order = None
        interval = None
        intervals_dicts = None

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
                elif (
                    isinstance(item.context_expr, ast.Call)
                    and item.context_expr.func.id == "horizontal"
                ):
                    intervals_dicts = self._visit_with_horizontal(item, loc)
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
            stmts.extend(gtc_utils.listify(self.visit(stmt)))
        self.parsing_context = ParsingContext.CONTROL_FLOW

        if intervals_dicts:
            stmts = [
                nodes.HorizontalIf(
                    intervals=intervals_dict,
                    body=nodes.BlockStmt(stmts=stmts, loc=loc),
                )
                for intervals_dict in intervals_dicts
            ]

        return nodes.ComputationBlock(
            interval=interval,
            iteration_order=iteration_order,
            loc=nodes.Location.from_ast_node(node),
            body=nodes.BlockStmt(stmts=stmts, loc=loc),
        )

    # Visitor methods
    # -- Special nodes --
    def visit_Raise(self):
        return nodes.InvalidBranch()

    # -- Literal nodes --
    def visit_Constant(
        self, node: ast.Constant
    ) -> Union[nodes.ScalarLiteral, nodes.BuiltinLiteral, nodes.Cast]:
        value = node.value
        if value is None:
            return nodes.BuiltinLiteral(value=nodes.Builtin.from_value(value))
        elif isinstance(value, bool):
            return nodes.Cast(
                data_type=nodes.DataType.BOOL,
                expr=nodes.BuiltinLiteral(
                    value=nodes.Builtin.from_value(value),
                ),
                loc=nodes.Location.from_ast_node(node),
            )
        elif isinstance(value, numbers.Number):
            data_type = nodes.DataType.from_dtype(np.dtype(type(value)))
            return nodes.ScalarLiteral(value=value, data_type=data_type)
        else:
            raise GTScriptSyntaxError(
                f"Unknown constant value found: {value}. Expected boolean or number.",
                loc=nodes.Location.from_ast_node(node),
            )

    def visit_Tuple(self, node: ast.Tuple) -> tuple:
        value = tuple(self.visit(elem) for elem in node.elts)
        return value

    # -- Symbol nodes --
    def visit_Attribute(self, node: ast.Attribute):
        # Matrix Transposed
        if node.attr == "T":
            return nodes.UnaryOpExpr(
                op=nodes.UnaryOperator.TRANSPOSED,
                arg=self.visit(node.value),
                loc=nodes.Location.from_ast_node(node),
            )
        else:
            qualified_name = gt_meta.get_qualified_name_from_node(node)
            return self.visit(ast.Name(id=qualified_name, ctx=node.ctx))

    def visit_Name(self, node: ast.Name) -> nodes.Ref:
        symbol = node.id
        if self._is_field(symbol):
            result = nodes.FieldRef.at_center(
                symbol, axes=self.fields[symbol].axes, loc=nodes.Location.from_ast_node(node)
            )
        elif self._is_parameter(symbol):
            result = nodes.VarRef(name=symbol, loc=nodes.Location.from_ast_node(node))
        elif self._is_local_symbol(symbol):
            raise AssertionError("Logic error")
        else:
            raise AssertionError(f"Missing '{symbol}' symbol definition")

        return result

    def visit_Index(self, node: ast.Index):
        index = self.visit(node.value)
        return index

    def _eval_index(self, node: ast.Subscript) -> Optional[List[int]]:
        invalid_target = GTScriptSyntaxError(message="Invalid target in assignment.", loc=node)

        tuple_or_expr = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
        index_nodes = gtc_utils.listify(
            tuple_or_expr.elts if isinstance(tuple_or_expr, ast.Tuple) else tuple_or_expr
        )

        if any(isinstance(cn, ast.Slice) for cn in index_nodes):
            raise invalid_target
        if any(isinstance(cn, ast.Ellipsis) for cn in index_nodes):
            return None
        else:
            index = []
            for index_node in index_nodes:
                try:
                    offset = ast.literal_eval(index_node)
                    index.append(offset)
                except Exception:
                    index.append(self.visit(index_node))
            return index

    def visit_Subscript(self, node: ast.Subscript):
        assert isinstance(node.ctx, (ast.Load, ast.Store))

        index = self._eval_index(node)
        result = self.visit(node.value)
        if isinstance(result, nodes.VarRef):
            assert index is not None
            result.index = index[0]
        else:
            if isinstance(node.value, ast.Name):
                field_axes = self.fields[result.name].axes
                if index is not None:
                    if len(field_axes) != len(index):
                        raise GTScriptSyntaxError(
                            f"Incorrect offset specification detected. Found {index}, "
                            f"but the field has dimensions ({', '.join(field_axes)})"
                        )
                    result.offset = {axis: value for axis, value in zip(field_axes, index)}
            elif isinstance(node.value, ast.Subscript):
                result.data_index = [
                    nodes.ScalarLiteral(value=value, data_type=nodes.DataType.INT32)
                    if isinstance(value, numbers.Integral)
                    else value
                    for value in index
                ]
                if len(result.data_index) != len(self.fields[result.name].data_dims):
                    raise GTScriptSyntaxError(
                        f"Incorrect data index length {len(result.data_index)}. "
                        f"Invalid data dimension index. Field {result.name} has {len(self.fields[result.name].data_dims)} data dimensions.",
                        loc=nodes.Location.from_ast_node(node),
                    )
                if any(
                    isinstance(index, nodes.ScalarLiteral)
                    and not (0 <= int(index.value) < axis_length)
                    for index, axis_length in zip(
                        result.data_index, self.fields[result.name].data_dims
                    )
                ):
                    raise GTScriptSyntaxError(
                        f"Data index out of bounds. "
                        f"Found index {result.data_index}, but field {result.name} has {self.fields[result.name].data_dims} data-dimensions",
                        loc=nodes.Location.from_ast_node(node),
                    )
            else:
                raise GTScriptSyntaxError(
                    "Unrecognized subscript expression", loc=nodes.Location.from_ast_node(node)
                )

        return result

    # -- Expressions nodes --
    def visit_UnaryOp(self, node: ast.UnaryOp):
        op = self.visit(node.op)
        arg = self.visit(node.operand)
        if isinstance(arg, numbers.Number):
            result = eval("{op}{arg}".format(op=op.python_symbol, arg=arg))
        else:
            result = nodes.UnaryOpExpr(op=op, arg=arg, loc=nodes.Location.from_ast_node(node))

        return result

    def visit_UAdd(self, node: ast.UAdd) -> nodes.UnaryOperator:
        return nodes.UnaryOperator.POS

    def visit_USub(self, node: ast.USub) -> nodes.UnaryOperator:
        return nodes.UnaryOperator.NEG

    def visit_Not(self, node: ast.Not) -> nodes.UnaryOperator:
        return nodes.UnaryOperator.NOT

    def visit_BinOp(self, node: ast.BinOp) -> nodes.BinOpExpr:
        return nodes.BinOpExpr(
            op=self.visit(node.op),
            rhs=self.visit(node.right),
            lhs=self.visit(node.left),
            loc=nodes.Location.from_ast_node(node),
        )

    def visit_Add(self, node: ast.Add) -> nodes.BinaryOperator:
        return nodes.BinaryOperator.ADD

    def visit_Sub(self, node: ast.Sub) -> nodes.BinaryOperator:
        return nodes.BinaryOperator.SUB

    def visit_Mult(self, node: ast.Mult) -> nodes.BinaryOperator:
        return nodes.BinaryOperator.MUL

    def visit_Div(self, node: ast.Div) -> nodes.BinaryOperator:
        return nodes.BinaryOperator.DIV

    def visit_Mod(self, node: ast.Mod) -> nodes.BinaryOperator:
        return nodes.BinaryOperator.MOD

    def visit_Pow(self, node: ast.Pow) -> nodes.BinaryOperator:
        return nodes.BinaryOperator.POW

    def visit_MatMult(self, node: ast.MatMult) -> nodes.BinaryOperator:
        return nodes.BinaryOperator.MATMULT

    def visit_And(self, node: ast.And) -> nodes.BinaryOperator:
        return nodes.BinaryOperator.AND

    def visit_Or(self, node: ast.And) -> nodes.BinaryOperator:
        return nodes.BinaryOperator.OR

    def visit_Eq(self, node: ast.Eq) -> nodes.BinaryOperator:
        return nodes.BinaryOperator.EQ

    def visit_NotEq(self, node: ast.NotEq) -> nodes.BinaryOperator:
        return nodes.BinaryOperator.NE

    def visit_Lt(self, node: ast.Lt) -> nodes.BinaryOperator:
        return nodes.BinaryOperator.LT

    def visit_LtE(self, node: ast.LtE) -> nodes.BinaryOperator:
        return nodes.BinaryOperator.LE

    def visit_Gt(self, node: ast.Gt) -> nodes.BinaryOperator:
        return nodes.BinaryOperator.GT

    def visit_GtE(self, node: ast.GtE) -> nodes.BinaryOperator:
        return nodes.BinaryOperator.GE

    def visit_BoolOp(self, node: ast.BoolOp) -> nodes.BinOpExpr:
        op = self.visit(node.op)
        rhs = self.visit(node.values[-1])
        for value in reversed(node.values[:-1]):
            lhs = self.visit(value)
            rhs = nodes.BinOpExpr(op=op, lhs=lhs, rhs=rhs, loc=nodes.Location.from_ast_node(node))
            res = rhs

        return res

    def visit_Compare(self, node: ast.Compare) -> nodes.BinOpExpr:
        lhs = self.visit(node.left)
        args = [lhs]

        assert len(node.comparators) >= 1
        op = self.visit(node.ops[-1])
        rhs = self.visit(node.comparators[-1])
        args.append(rhs)

        for i in range(len(node.comparators) - 2, -1, -1):
            lhs = self.visit(node.values[i])
            rhs = nodes.BinOpExpr(op=op, lhs=lhs, rhs=rhs, loc=nodes.Location.from_ast_node(node))
            op = self.visit(node.ops[i])
            args.append(lhs)

        result = nodes.BinOpExpr(op=op, lhs=lhs, rhs=rhs, loc=nodes.Location.from_ast_node(node))

        return result

    def visit_IfExp(self, node: ast.IfExp) -> nodes.TernaryOpExpr:
        result = nodes.TernaryOpExpr(
            condition=self.visit(node.test),
            then_expr=self.visit(node.body),
            else_expr=self.visit(node.orelse),
        )

        return result

    def visit_If(self, node: ast.If) -> list:
        self.decls_stack.append([])

        main_stmts = []
        for stmt in node.body:
            main_stmts.extend(gtc_utils.listify(self.visit(stmt)))
        assert all(isinstance(item, nodes.Statement) for item in main_stmts)

        else_stmts = []
        if node.orelse:
            for stmt in node.orelse:
                else_stmts.extend(gtc_utils.listify(self.visit(stmt)))
            assert all(isinstance(item, nodes.Statement) for item in else_stmts)

        result = []
        if len(self.decls_stack) == 1:
            result.extend(self.decls_stack.pop())
        elif len(self.decls_stack) > 1:
            self.decls_stack[-2].extend(self.decls_stack[-1])
            self.decls_stack.pop()

        result.append(
            nodes.If(
                condition=self.visit(node.test),
                loc=nodes.Location.from_ast_node(node),
                main_body=nodes.BlockStmt(stmts=main_stmts, loc=nodes.Location.from_ast_node(node)),
                else_body=nodes.BlockStmt(stmts=else_stmts, loc=nodes.Location.from_ast_node(node))
                if else_stmts
                else None,
            )
        )

        return result

    def visit_While(self, node: ast.While) -> list:
        loc = nodes.Location.from_ast_node(node)

        self.decls_stack.append([])
        stmts = gt_utils.flatten([self.visit(stmt) for stmt in node.body])
        assert all(isinstance(item, nodes.Statement) for item in stmts)

        result = [
            nodes.While(
                condition=self.visit(node.test),
                loc=nodes.Location.from_ast_node(node),
                body=nodes.BlockStmt(stmts=stmts, loc=loc),
            )
        ]

        if len(self.decls_stack) == 1:
            result.extend(self.decls_stack.pop())
        elif len(self.decls_stack) > 1:
            self.decls_stack[-2].extend(self.decls_stack[-1])
            self.decls_stack.pop()

        return result

    def visit_Call(self, node: ast.Call):
        native_fcn = nodes.NativeFunction.PYTHON_SYMBOL_TO_IR_OP[node.func.id]

        args = [self.visit(arg) for arg in node.args]
        if len(args) != native_fcn.arity:
            raise GTScriptSyntaxError(
                "Invalid native function call", loc=nodes.Location.from_ast_node(node)
            )

        return nodes.NativeFuncCall(
            func=native_fcn,
            args=args,
            data_type=nodes.DataType.AUTO,
            loc=nodes.Location.from_ast_node(node),
        )

    # -- Statement nodes --
    def _parse_assign_target(
        self, target_node: Union[ast.Subscript, ast.Name]
    ) -> Tuple[str, Optional[List[int]], Optional[List[int]]]:
        invalid_target = GTScriptSyntaxError(
            message="Invalid target in assignment.", loc=target_node
        )
        spatial_offset = None
        data_index = None
        if isinstance(target_node, ast.Name):
            name = target_node.id
        elif isinstance(target_node, ast.Subscript):
            if isinstance(target_node.value, ast.Name):
                name = target_node.value.id
                spatial_offset = self._eval_index(target_node)
            elif isinstance(target_node.value, ast.Subscript) and isinstance(
                target_node.value.value, ast.Name
            ):
                name = target_node.value.value.id
                spatial_offset = self._eval_index(target_node.value)
                data_index = self._eval_index(target_node)
            else:
                raise invalid_target
            if spatial_offset is None:
                num_axes = len(self.fields[name].axes) if name in self.fields else 3
                spatial_offset = [0] * num_axes
        else:
            raise invalid_target

        return name, spatial_offset, data_index

    def visit_Assign(self, node: ast.Assign) -> list:
        result = []

        # Create decls for temporary fields
        target = []
        if len(node.targets) > 1:
            raise GTScriptSyntaxError(
                message="Assignment to multiple variables (e.g. var1 = var2 = value) not supported.",
                loc=nodes.Location.from_ast_node(node),
            )

        for t in node.targets[0].elts if isinstance(node.targets[0], ast.Tuple) else node.targets:
            name, spatial_offset, data_index = self._parse_assign_target(t)
            if spatial_offset:
                if any(offset != 0 for offset in spatial_offset):
                    raise GTScriptSyntaxError(
                        message="Assignment to non-zero offsets is not supported.",
                        loc=nodes.Location.from_ast_node(t),
                    )

            if not self._is_known(name):
                if name in self.temp_decls:
                    field_decl = self.temp_decls[name]
                else:
                    if data_index is not None and data_index:
                        raise GTScriptSyntaxError(
                            message="Temporaries with data dimensions need to be declared explicitly.",
                            loc=nodes.Location.from_ast_node(t),
                        )
                    field_decl = nodes.FieldDecl(
                        name=name,
                        data_type=nodes.DataType.AUTO,
                        axes=nodes.Domain.LatLonGrid().axes_names,
                        is_api=False,
                        loc=nodes.Location.from_ast_node(t),
                    )

                if len(self.decls_stack):
                    self.decls_stack[-1].append(field_decl)
                else:
                    result.append(field_decl)
                self.fields[field_decl.name] = field_decl

            if not self.parsing_horizontal_region:
                self.written_vars.add(name)

            axes = self.fields[name].axes
            par_axes_names = [axis.name for axis in nodes.Domain.LatLonGrid().parallel_axes]
            if self.iteration_order == nodes.IterationOrder.PARALLEL:
                par_axes_names.append(nodes.Domain.LatLonGrid().sequential_axis.name)
            if set(par_axes_names) - set(axes):
                raise GTScriptSyntaxError(
                    message=f"Cannot assign to field '{node.targets[0].id}' as all parallel axes '{par_axes_names}' are not present.",
                    loc=nodes.Location.from_ast_node(t),
                )

            target.append(self.visit(t))

        value = gtc_utils.listify(self.visit(node.value))

        assert len(target) == len(value)
        for left, right in zip(target, value):
            result.append(
                nodes.Assign(target=left, value=right, loc=nodes.Location.from_ast_node(node))
            )

        return result

    def visit_AugAssign(self, node: ast.AugAssign):
        """Implement left <op>= right in terms of left = left <op> right."""
        binary_operation = ast.BinOp(left=node.target, op=node.op, right=node.value)
        assignment = ast.Assign(targets=[node.target], value=binary_operation)
        ast.copy_location(binary_operation, node)
        ast.copy_location(assignment, node)
        return self.visit_Assign(assignment)

    def visit_With(self, node: ast.With):
        loc = nodes.Location.from_ast_node(node)
        syntax_error = GTScriptSyntaxError(
            f"Invalid 'with' statement at line {loc.line} (column {loc.column})", loc=loc
        )

        if (
            len(node.items) == 1
            and isinstance(node.items[0].context_expr, ast.Call)
            and node.items[0].context_expr.func.id == "horizontal"
        ):
            if any(isinstance(child_node, ast.With) for child_node in node.body):
                raise GTScriptSyntaxError("Cannot nest `with` node inside horizontal region")

            self.parsing_horizontal_region = True
            intervals_dicts = self._visit_with_horizontal(node.items[0], loc)
            all_stmts = gt_utils.flatten(
                [gtc_utils.listify(self.visit(stmt)) for stmt in node.body]
            )
            self.parsing_horizontal_region = False
            stmts = list(filter(lambda stmt: isinstance(stmt, nodes.Decl), all_stmts))
            body_block = nodes.BlockStmt(
                stmts=list(filter(lambda stmt: not isinstance(stmt, nodes.Decl), all_stmts)),
                loc=loc,
            )
            names = _find_accesses_with_offsets(body_block)
            written_then_offset = names.intersection(self.written_vars)
            if written_then_offset:
                raise GTScriptSyntaxError(
                    "The following variables are"
                    f"written before being referenced with an offset in a horizontal region: {', '.join(written_then_offset)}"
                )
            stmts.extend(
                [
                    nodes.HorizontalIf(intervals=intervals_dict, body=body_block)
                    for intervals_dict in intervals_dicts
                ]
            )
            return stmts
        else:
            # If we find nested `with` blocks flatten them, i.e. transform
            #  with computation(PARALLEL):
            #   with interval(...):
            #     ...
            # into
            #  with computation(PARALLEL), interval(...):
            #    ...
            # otherwise just parse the node
            if self.parsing_context == ParsingContext.CONTROL_FLOW and all(
                isinstance(child_node, ast.With)
                and child_node.items[0].context_expr.func.id == "interval"
                for child_node in node.body
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
                if not self._are_intervals_nonoverlapping(compute_blocks):
                    raise GTScriptSyntaxError(
                        f"Overlapping intervals detected at line {loc.line} (column {loc.column})"
                    )

                return compute_blocks
            elif self.parsing_context == ParsingContext.CONTROL_FLOW:
                return gtc_utils.listify(self._visit_computation_node(node))
            else:
                # Mixing nested `with` blocks with stmts not allowed
                raise syntax_error

    def visit_FunctionDef(self, node: ast.FunctionDef) -> List[nodes.ComputationBlock]:
        blocks = []
        for stmt in filter(lambda s: not isinstance(s, ast.AnnAssign), node.body):
            blocks.extend(self.visit(stmt))

        if not all(isinstance(item, nodes.ComputationBlock) for item in blocks):
            raise GTScriptSyntaxError(
                "Invalid stencil definition", loc=nodes.Location.from_ast_node(node)
            )

        return blocks


class CollectLocalSymbolsAstVisitor(ast.NodeVisitor):
    @classmethod
    def apply(cls, node: ast.FunctionDef):
        return cls()(node)

    def __call__(self, node: ast.FunctionDef):
        self.local_symbols = set()
        self.visit(node)
        result = self.local_symbols
        del self.local_symbols
        return result

    def visit_Assign(self, node: ast.Assign):
        invalid_target = GTScriptSyntaxError(
            message="invalid target in assign", loc=nodes.Location.from_ast_node(node)
        )
        for target in node.targets:
            targets = target.elts if isinstance(target, ast.Tuple) else [target]
            for t in targets:
                if isinstance(t, ast.Name):
                    self.local_symbols.add(t.id)
                elif isinstance(t, ast.Subscript):
                    if isinstance(t.value, ast.Name):
                        name_node = t.value
                    elif isinstance(t.value, ast.Subscript) and isinstance(t.value.value, ast.Name):
                        name_node = t.value.value
                    else:
                        raise invalid_target
                    self.local_symbols.add(name_node.id)
                else:
                    raise invalid_target


class GTScriptParser(ast.NodeVisitor):

    CONST_VALUE_TYPES = (
        *gtscript._VALID_DATA_TYPES,
        types.FunctionType,
        type(None),
        gtscript.AxisIndex,
    )

    def __init__(self, definition, *, options, externals=None):
        assert isinstance(definition, types.FunctionType)
        self.definition = definition
        self.filename = inspect.getfile(definition)
        self.source, decorators_source = gt_meta.split_def_decorators(self.definition)
        self.ast_root = ast.parse(self.source, feature_version=(3, 9))
        self.options = options
        self.build_info = options.build_info
        self.main_name = options.name
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
    def annotate_definition(definition, externals=None):
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
                    message="'**kwargs' dict parameter is not supported in GTScript definitions",
                )
            else:
                is_keyword = param.kind == inspect.Parameter.KEYWORD_ONLY

                default = nodes.Empty
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
                    nodes.ArgumentInfo(name=param.name, is_keyword=is_keyword, default=default)
                )

                api_annotations.append(dtype_annotation)

        nonlocal_symbols, imported_symbols = GTScriptParser.collect_external_symbols(definition)
        ast_func_def = gt_meta.get_ast(definition).body[0]
        canonical_ast = gt_meta.ast_dump(ast_func_def)

        # resolve externals
        if externals is not None:
            resolved_externals = GTScriptParser.resolve_external_symbols(
                nonlocal_symbols, imported_symbols, externals
            )

        # Gather temporary
        temp_annotations: Dict[str, gtscript._FieldDescriptor] = {}
        temp_init_values: Dict[str, numbers.Number] = {}

        ann_assign_context = {
            "Field": gtscript.Field,
            "IJK": gtscript.IJK,
            "IJ": gtscript.IJ,
            "I": gtscript.I,
            "J": gtscript.J,
            "K": gtscript.K,
            "IK": gtscript.IK,
            "JK": gtscript.JK,
            "np": np,
            **(resolved_externals if externals is not None else nonlocal_symbols),
        }
        ann_assigns = tuple(filter(lambda stmt: isinstance(stmt, ast.AnnAssign), ast_func_def.body))
        for ann_assign in ann_assigns:
            assert isinstance(ann_assign.target, ast.Name)
            name = ann_assign.target.id

            source = gt_meta.ast_unparse(ann_assign.annotation)
            descriptor = eval(source, ann_assign_context)
            temp_annotations[name] = descriptor
            if descriptor.axes != gtscript.IJK:
                axes = "".join(str(ax) for ax in descriptor.axes)
                raise GTScriptSyntaxError(
                    f"Found {axes}, but only IJK is currently supported for temporaries",
                    loc=nodes.Location.from_ast_node(ast_func_def),
                )

            if hasattr(ann_assign, "value") and ann_assign.value is not None:
                assert isinstance(ann_assign.value, ast.Constant)
                temp_init_values[name] = ann_assign.value.value

        definition._gtscript_ = dict(
            qualified_name=qualified_name,
            api_signature=api_signature,
            api_annotations=api_annotations,
            temp_annotations=temp_annotations,
            temp_init_values=temp_init_values,
            canonical_ast=canonical_ast,
            nonlocals=nonlocal_symbols,
            imported=imported_symbols,
            externals=resolved_externals if externals is not None else {},
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

        gtscript_ast = ast.parse(gt_meta.get_ast(definition), feature_version=(3, 9)).body[0]
        local_symbols = CollectLocalSymbolsAstVisitor.apply(gtscript_ast)

        nonlocal_symbols = {}

        name_nodes = gt_meta.collect_names(definition, skip_annotations=False)
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
                        nodes.Location.from_ast_node(name_nodes[collected_name][0]),
                    )
                    if hasattr(nonlocal_symbols[collected_name], "_gtscript_"):
                        # Recursively add nonlocals and imported symbols
                        nonlocal_symbols.update(
                            nonlocal_symbols[collected_name]._gtscript_["nonlocals"]
                        )
                        imported_symbols.update(
                            nonlocal_symbols[collected_name]._gtscript_["imported"]
                        )
                elif root_name not in local_symbols and root_name in unbound:
                    raise GTScriptSymbolError(
                        name=collected_name,
                        loc=nodes.Location.from_ast_node(name_nodes[collected_name][0]),
                    )

        return nonlocal_symbols, imported_symbols

    @staticmethod
    def eval_external(name: str, context: dict, loc=None):
        try:
            value = eval(name, context)

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
        for value in func_externals.values():
            if not hasattr(value, "_gtscript_"):
                raise TypeError(f"{value.__name__} is not a gtscript function")
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
                                    attr_name, context, nodes.Location.from_ast_node(attr_nodes[0])
                                ),
                            )
                        )

                elif not exhaustive:
                    resolved_values_list.append((name, GTScriptParser.eval_external(name, context)))

            for _, value in resolved_values_list:
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
                    if arg_info.default not in [nodes.Empty, None]:
                        assert np.dtype(type(arg_info.default)) == dtype
                    data_type = nodes.DataType.from_dtype(dtype)
                    parameter_decls[arg_info.name] = nodes.VarDecl(
                        name=arg_info.name, data_type=data_type, length=0, is_api=True
                    )
                elif isinstance(arg_annotation, gtscript._SequenceDescriptor):
                    assert arg_info.default in [nodes.Empty, None]
                    data_type = nodes.DataType.from_dtype(np.dtype(arg_annotation))
                    length = arg_annotation.length
                    parameter_decls[arg_info.name] = nodes.VarDecl(
                        name=arg_info.name, data_type=data_type, length=length, is_api=True
                    )
                else:
                    assert isinstance(arg_annotation, gtscript._FieldDescriptor)
                    assert arg_info.default in [nodes.Empty, None]
                    data_type = nodes.DataType.from_dtype(np.dtype(arg_annotation.dtype))
                    axes = [ax.name for ax in arg_annotation.axes]
                    data_dims = list(arg_annotation.data_dims)
                    fields_decls[arg_info.name] = nodes.FieldDecl(
                        name=arg_info.name,
                        data_type=data_type,
                        axes=axes,
                        data_dims=data_dims,
                        is_api=True,
                        layout_id=arg_info.name,
                    )

                if data_type is nodes.DataType.INVALID:
                    raise GTScriptDataTypeError(name=arg_info.name, data_type=data_type)

            except Exception as e:
                raise GTScriptDefinitionError(
                    name=arg_info.name,
                    value=arg_annotation,
                    message=f"Invalid definition of argument '{arg_info.name}': {arg_annotation}",
                ) from e

        for item in itertools.chain(fields_decls.values(), parameter_decls.values()):
            if item.data_type is nodes.DataType.INVALID:
                raise GTScriptDataTypeError(name=item.name, data_type=item.data_type)

        return api_signature, fields_decls, parameter_decls

    def run(self):
        assert (
            isinstance(self.ast_root, ast.Module)
            and "body" in self.ast_root._fields
            and len(self.ast_root.body) == 1
            and isinstance(self.ast_root.body[0], ast.FunctionDef)
        )
        main_func_node = self.ast_root.body[0]

        assert hasattr(self.definition, "_gtscript_")
        self.resolved_externals = self.definition._gtscript_["externals"]
        api_signature, fields_decls, parameter_decls = self.extract_arg_descriptors()

        # Inline constant values
        for value in self.resolved_externals.values():
            if hasattr(value, "_gtscript_"):
                assert callable(value)
                func_node = ast.parse(gt_meta.get_ast(value), feature_version=(3, 9)).body[0]
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

        ValueInliner.apply(main_func_node, context=local_context)

        # Inline function calls
        CallInliner.apply(main_func_node, context=local_context)

        # Evaluate and inline compile-time conditionals
        CompiledIfInliner.apply(main_func_node, context=local_context)

        AssertionChecker.apply(main_func_node, context=local_context, source=self.source)

        temp_decls = _make_temp_decls(self.definition._gtscript_["temp_annotations"])
        fields_decls.update(temp_decls)

        init_computations = _make_init_computations(
            temp_decls, self.definition._gtscript_["temp_init_values"], func_node=main_func_node
        )

        # Generate definition IR
        domain = nodes.Domain.LatLonGrid()
        computations = IRMaker(
            fields=fields_decls,
            parameters=parameter_decls,
            local_symbols={},  # Not used
            domain=domain,
            temp_decls=temp_decls,
        )(self.ast_root)

        self.definition_ir = nodes.StencilDefinition(
            name=self.main_name,
            domain=domain,
            api_signature=api_signature,
            api_fields=[
                fields_decls[item.name] for item in api_signature if item.name in fields_decls
            ],
            parameters=[
                parameter_decls[item.name] for item in api_signature if item.name in parameter_decls
            ],
            computations=init_computations + computations if init_computations else computations,
            externals=self.resolved_externals,
            docstring=inspect.getdoc(self.definition) or "",
            loc=nodes.Location.from_ast_node(self.ast_root.body[0]),
        )

        self.definition_ir = UnrollVectorAssignments.apply(
            self.definition_ir, fields_decls=fields_decls
        )
        return self.definition_ir


@register
class GTScriptFrontend(Frontend):
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
        return GTScriptParser.annotate_definition(definition, externals)

    @classmethod
    def generate(cls, definition, externals, options):
        if options.build_info is not None:
            start_time = time.perf_counter()

        if not hasattr(definition, "_gtscript_"):
            cls.prepare_stencil_definition(definition, externals)
        translator = GTScriptParser(definition, externals=externals, options=options)
        definition_ir = translator.run()

        # GTIR only supports LatLonGrids
        if definition_ir.domain != nodes.Domain.LatLonGrid():
            raise TypeError("GTIR does not support grids other than LatLong.")
        gtir_stencil = DefIRToGTIR.apply(definition_ir)

        if options.build_info is not None:
            options.build_info["parse_time"] = time.perf_counter() - start_time

        return gtir_stencil
