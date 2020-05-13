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
import inspect
import numbers
import textwrap
import types

from .nodes import *
from gt4py.utils import NOTHING

# --- Definition IR ---
DEFAULT_LAYOUT_ID = "_default_layout_id_"


def make_expr(value):
    if isinstance(value, Expr):
        result = value
    elif isinstance(value, numbers.Number):
        data_type = DataType.from_dtype(np.dtype(type(value)))
        result = ScalarLiteral(value=value, data_type=data_type)
    else:
        raise ValueError("Invalid expression value '{}'".format(value))
    return result


def make_field_decl(
    name: str,
    dtype=np.float_,
    masked_axes=None,
    is_api=True,
    layout_id=DEFAULT_LAYOUT_ID,
    loc=None,
    *,
    axes_dict=None,
):
    axes_dict = axes_dict or {ax.name: ax for ax in Domain.LatLonGrid().axes}
    masked_axes = masked_axes or []
    return FieldDecl(
        name=name,
        data_type=DataType.from_dtype(dtype),
        axes=[name for name, axis in axes_dict.items() if name not in masked_axes],
        is_api=is_api,
        layout_id=layout_id,
        loc=loc,
    )


def make_field_ref(name: str, offset=(0, 0, 0), *, axes_names=None):
    axes_names = axes_names or [ax.name for ax in Domain.LatLonGrid().axes]
    offset = {axes_names[i]: value for i, value in enumerate(offset) if value is not None}
    return FieldRef(name=name, offset=offset)


def make_axis_interval(bounds: tuple, *, offset_limit: int = 2):
    assert isinstance(bounds[0], (VarRef, UnaryOpExpr, BinOpExpr)) or (
        isinstance(bounds[0], int) and abs(bounds[0]) <= offset_limit
    )
    if isinstance(bounds[0], (VarRef, BinOpExpr)):
        # assuming: "var_name[index] + offset"
        if isinstance(bounds[0], VarRef):
            start = AxisBound(level=bounds[0])
        else:
            if bounds[0].op == BinaryOperator.ADD:
                offset = bounds[0].rhs.value
            elif bounds[0].op == BinaryOperator.SUB:
                offset = -bounds[0].rhs.value
            else:
                assert False, "Invalid bound"
            start = AxisBound(level=bounds[0].lhs, offset=offset)
    elif bounds[0] >= 0:
        start = AxisBound(level=LevelMarker.START, offset=bounds[0])
    else:
        start = AxisBound(level=LevelMarker.END, offset=bounds[0])

    assert isinstance(bounds[1], (VarRef, BinOpExpr, BuiltinLiteral)) or (
        isinstance(bounds[1], int) and abs(bounds[1])
    )
    if isinstance(bounds[1], (VarRef, BinOpExpr)):
        # assuming: "var_name[index] + offset"
        if isinstance(bounds[1], VarRef):
            end = AxisBound(level=bounds[1])
        else:
            if bounds[1].op == BinaryOperator.ADD:
                offset = bounds[1].rhs.value
            elif bounds[1].op == BinaryOperator.SUB:
                offset = -bounds[1].rhs.value
            else:
                assert False, "Invalid bound"
            end = AxisBound(level=bounds[1].lhs, offset=offset)
    elif isinstance(bounds[1], BuiltinLiteral):  # None
        end = AxisBound(level=LevelMarker.END)
    elif bounds[1] >= 0:
        end = AxisBound(level=LevelMarker.START, offset=bounds[1])
    else:
        end = AxisBound(level=LevelMarker.END, offset=bounds[1])

    # TODO: more thorough verifications
    assert (
        start.level == LevelMarker.START
        and (start.offset < end.offset or start.level != end.level)
    ) or (
        start.level == LevelMarker.END
        and end.level == LevelMarker.END
        and start.offset < end.offset
    )

    interval = AxisInterval(start=start, end=end)

    return interval


def make_api_signature(args_list: list):
    api_signature = []
    for item in args_list:
        if isinstance(item, str):
            api_signature.append(ArgumentInfo(name=item, is_keyword=False, default=None))
        elif isinstance(item, tuple):
            api_signature.append(
                ArgumentInfo(
                    name=item[0],
                    is_keyword=item[1] if len(item) > 1 else False,
                    default=item[2] if len(item) > 2 else None,
                )
            )
        else:
            assert isinstance(item, ArgumentInfo), "Invalid api_signature"
    return api_signature


def make_definition(
    stencil_name: str,
    definition_func: types.FunctionType,
    args_list: list,
    fields_with_storage_descriptor: dict,
    temp_fields_with_type: dict,
    parameters_with_type: dict,
    domain=None,
    externals=None,
    sources=None,
):
    from gt4py import gtscript

    api_signature = make_api_signature(args_list)
    domain = domain or Domain.LatLonGrid()
    externals = externals or {}
    sources = sources or {}

    fields_decls = []
    for name, descriptor in fields_with_storage_descriptor.items():
        fields_decls.append(
            make_field_decl(
                name=name,
                dtype=descriptor.dtype,
                masked_axes=[
                    i
                    for i, masked in enumerate(gtscript.mask_from_axes(descriptor.axes))
                    if masked
                ],
                is_api=True,
                layout_id=name,
            )
        )

    temp_fields_decls = {
        name: make_field_decl(name=name, dtype=dtype, is_api=False)
        for name, dtype in temp_fields_with_type.items()
    }

    parameter_decls = []
    for key, value in parameters_with_type.items():
        if isinstance(value, tuple):
            assert len(value) == 2
            data_type = value[0]
            length = value[1]
        else:
            data_type = value
            length = 0

        parameter_decls.append(
            VarDecl(name=key, data_type=DataType.from_dtype(data_type), length=length, is_api=True)
        )

    computations = make_computations(
        definition_func,
        fields={decl.name: decl for decl in fields_decls},
        parameters={decl.name: decl for decl in parameter_decls},
        local_symbols=None,
        externals=externals,
        domain=domain,
        extra_temp_decls=temp_fields_decls,
    )

    definition = StencilDefinition(
        name=stencil_name,
        domain=domain,
        api_signature=api_signature,
        api_fields=fields_decls,
        parameters=parameter_decls,
        computations=computations,
        externals=externals,
        sources=sources,
    )

    return definition


# --- Implementation IR ---
def make_field_accessor(name: str, intent=False, extent=((0, 0), (0, 0), (0, 0))):
    if not isinstance(intent, AccessIntent):
        assert isinstance(intent, bool)
        intent = AccessIntent.READ_WRITE if intent else AccessIntent.READ_ONLY
    return FieldAccessor(symbol=name, intent=intent, extent=Extent(extent))

    # Apply blocks and decls


def make_stage(
    stage_func,
    compute_extent: list = None,
    fields_with_access: dict = None,
    parameters: dict = None,
    local_symbols: dict = None,
    externals: dict = None,
    *,
    domain: Domain = None,
):
    compute_extent = Extent(compute_extent or [(0, 0), (0, 0), (0, 0)])
    computations = make_computations(
        stage_func,
        fields=fields_with_access,
        parameters=parameters,
        local_symbols=local_symbols,
        externals=externals,
        domain=domain,
    )

    # Signature convention:
    #   - regular args are fields
    #   - kwonly args are parameters
    ast_root = ast.parse(textwrap.dedent(inspect.getsource(stage_func)))
    func_ast = ast_root.body[0]
    accessors = []
    for arg in func_ast.args.args:
        name = arg.arg
        is_read_write, extent = fields_with_access[name]
        accessors.append(make_field_accessor(name, is_read_write, extent))

    for arg in func_ast.args.kwonlyargs:
        name = arg.arg
        accessors.append(ParameterAccessor(symbol=name))
    local_symbols = (
        {
            key: VarDecl(name=key, data_type=DataType.from_dtype(value), length=0, is_api=False)
            for key, value in local_symbols.items()
        }
        if local_symbols
        else {}
    )
    apply_blocks = [
        ApplyBlock(
            interval=computation.interval, body=computation.body, local_symbols=local_symbols
        )
        for computation in computations
    ]

    stage = Stage(
        name=func_ast.name,
        accessors=accessors,
        apply_blocks=apply_blocks,
        compute_extent=compute_extent,
    )

    return stage


def make_multi_stage(name: str, iteration_order: IterationOrder, stages: list):
    groups = []
    for grouped_stages in stages:
        grouped_stages = gt_utils.listify(grouped_stages)
        groups.append(StageGroup(stages=grouped_stages))
    multi_stage = MultiStage(name=name, iteration_order=iteration_order, groups=groups)
    return multi_stage


def make_implementation(
    name: str,
    args_list: list,
    fields_with_description: dict,
    parameters_with_type: dict,
    multi_stages: list,
    domain=None,
    k_axis_splitters=None,
    externals=None,
    sources=None,
):
    from gt4py.analysis.passes import DataTypePass

    api_signature = make_api_signature(args_list)

    domain = domain or Domain.LatLonGrid()
    # if k_axis_splitters is not None:
    #     # Assume: ["var_name"] or  [("var_name", index)]
    #     refs = []
    #     for item in k_axis_splitters:
    #         if isinstance(item, tuple):
    #             refs.append(VarRef(name=item[0], index=Index([item[1]])))
    #         else:
    #             refs.append(VarRef(name=item))
    #     axis_splitters = {domain.sequential_axis.name: refs}
    # else:
    #     axis_splitters = {}
    axis_splitters = None

    fields_decls = {}
    fields_extents = {}
    for field_name, description in fields_with_description.items():
        extent = description.pop("extent", Extent.zeros())
        description.setdefault("layout_id", repr(extent))
        fields_extents[field_name] = Extent(extent)
        fields_decls[field_name] = make_field_decl(name=field_name, **description)

    parameter_decls = {}
    for key, value in parameters_with_type.items():
        if isinstance(value, tuple):
            assert len(value) == 2
            data_type = value[0]
            length = value[1]
        else:
            data_type = value
            length = 0
        parameter_decls[key] = VarDecl(
            name=key, data_type=DataType.from_dtype(data_type), length=length, is_api=True
        )

    implementation = StencilImplementation(
        name=name,
        api_signature=api_signature,
        domain=domain,
        axis_splitters_var=axis_splitters,
        fields=fields_decls,
        parameters=parameter_decls,
        multi_stages=multi_stages,
        fields_extents=fields_extents,
        externals=externals,
        sources=sources,
    )
    #
    data_type_visitor = DataTypePass.CollectDataTypes()
    data_type_visitor(implementation)

    return implementation


class AST2IRVisitor(ast.NodeVisitor):
    @classmethod
    def apply(
        cls,
        stage_func,
        fields: dict = None,
        parameters: dict = None,
        local_symbols: dict = None,
        externals: dict = None,
        *,
        domain: Domain = None,
        extra_temp_decls: dict = None,
    ):
        return cls(
            fields,
            parameters,
            local_symbols,
            externals,
            domain=domain,
            extra_temp_decls=extra_temp_decls,
        )(ast.parse(textwrap.dedent(inspect.getsource(stage_func))))

    def __init__(
        self,
        fields: dict,
        parameters: dict,
        local_symbols: dict,
        externals: dict,
        *,
        domain: Domain,
        extra_temp_decls: dict,
    ):
        fields = fields or {}
        parameters = parameters or {}
        assert all(isinstance(name, str) for name in parameters.keys())
        local_symbols = local_symbols or {}
        assert all(isinstance(name, str) for name in local_symbols.keys()) and all(
            isinstance(value, (type, np.dtype)) for value in local_symbols.values()
        )
        externals = externals or {}
        assert all(isinstance(name, str) for name in externals.keys())

        self.fields = fields
        self.parameters = parameters
        self.local_symbols = local_symbols
        self.externals = externals
        self.domain = domain or Domain.LatLonGrid()
        self.extra_temp_decls = extra_temp_decls or {}

    def __call__(self, ast_root: ast.AST):
        assert (
            isinstance(ast_root, ast.Module)
            and "body" in ast_root._fields
            and len(ast_root.body) == 1
            and isinstance(ast_root.body[0], ast.FunctionDef)
        )
        func_ast = ast_root.body[0]
        computations = self.visit(func_ast)

        return computations

    def _is_field(self, name: str):
        return name in self.fields

    def _is_parameter(self, name: str):
        return name in self.parameters

    def _is_local_symbol(self, name: str):
        return name in self.local_symbols

    def _is_external(self, name: str):
        return name in self.externals

    def _is_known(self, name: str):
        return (
            self._is_field(name)
            or self._is_parameter(name)
            or self._is_local_symbol(name)
            or self._is_external(name)
        )

    def _get_qualified_name(self, node: ast.AST, *, joiner="."):
        if isinstance(node, ast.Name):
            result = node.id
        elif isinstance(node, ast.Attribute):
            prefix = self._get_qualified_name(node.value)
            result = joiner.join([prefix, node.attr])
        else:
            result = None

        return result

    def visit_Raise(self):
        return InvalidBranch()

    # -- Literal nodes --
    def visit_Num(self, node: ast.Num) -> numbers.Number:
        value = node.n
        return value

    def visit_Tuple(self, node: ast.Tuple) -> tuple:
        value = tuple(self.visit(elem) for elem in node.elts)
        return value

    def visit_NameConstant(self, node: ast.NameConstant):
        value = BuiltinLiteral(value=Builtin[str(node.value).upper()])
        return value

    # -- Symbol nodes --
    def visit_Attribute(self, node: ast.Attribute):
        qualified_name = self._get_qualified_name(node)
        return self.visit(ast.Name(id=qualified_name, ctx=node.ctx))

    def visit_Name(self, node: ast.Name) -> Ref:
        symbol = node.id
        if self._is_field(symbol):
            result = FieldRef(
                name=symbol,
                offset={axis: value for axis, value in zip(self.domain.axes_names, (0, 0, 0))},
            )
        elif self._is_parameter(symbol):
            result = VarRef(name=symbol)
        elif self._is_local_symbol(symbol):
            result = VarRef(name=symbol)
        elif self._is_external(symbol):
            result = make_expr(self.externals[symbol])
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
        if isinstance(result, VarRef):
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
            result = UnaryOpExpr(op=op, arg=arg)

        return result

    def visit_UAdd(self, node: ast.UAdd) -> UnaryOperator:
        return UnaryOperator.POS

    def visit_USub(self, node: ast.USub) -> UnaryOperator:
        return UnaryOperator.NEG

    def visit_Not(self, node: ast.Not) -> UnaryOperator:
        return UnaryOperator.NOT

    def visit_BinOp(self, node: ast.BinOp) -> BinOpExpr:
        op = self.visit(node.op)
        rhs = make_expr(self.visit(node.right))
        lhs = make_expr(self.visit(node.left))
        result = BinOpExpr(op=op, lhs=lhs, rhs=rhs)

        return result

    def visit_Add(self, node: ast.Add) -> BinaryOperator:
        return BinaryOperator.ADD

    def visit_Sub(self, node: ast.Sub) -> BinaryOperator:
        return BinaryOperator.SUB

    def visit_Mult(self, node: ast.Mult) -> BinaryOperator:
        return BinaryOperator.MUL

    def visit_Div(self, node: ast.Div) -> BinaryOperator:
        return BinaryOperator.DIV

    def visit_Pow(self, node: ast.Pow) -> BinaryOperator:
        return BinaryOperator.POW

    def visit_And(self, node: ast.And) -> BinaryOperator:
        return BinaryOperator.AND

    def visit_Or(self, node: ast.And) -> BinaryOperator:
        return BinaryOperator.OR

    def visit_Eq(self, node: ast.Eq) -> BinaryOperator:
        return BinaryOperator.EQ

    def visit_NotEq(self, node: ast.NotEq) -> BinaryOperator:
        return BinaryOperator.NE

    def visit_Lt(self, node: ast.Lt) -> BinaryOperator:
        return BinaryOperator.LT

    def visit_LtE(self, node: ast.LtE) -> BinaryOperator:
        return BinaryOperator.LE

    def visit_Gt(self, node: ast.Gt) -> BinaryOperator:
        return BinaryOperator.GT

    def visit_GtE(self, node: ast.GtE) -> BinaryOperator:
        return BinaryOperator.GE

    def visit_BoolOp(self, node: ast.BoolOp) -> BinOpExpr:
        op = self.visit(node.op)
        lhs = make_expr(self.visit(node.values[0]))
        args = [lhs]

        assert len(node.values) >= 2
        rhs = make_expr(self.visit(node.values[-1]))
        args.append(rhs)

        for i in range(len(node.values) - 2, 0, -1):
            lhs = make_expr(self.visit(node.values[i]))
            rhs = BinOpExpr(op=op, lhs=lhs, rhs=rhs)
            args.append(lhs)

        result = BinOpExpr(op=op, lhs=lhs, rhs=rhs)

        return result

    def visit_Compare(self, node: ast.Compare) -> BinOpExpr:
        lhs = make_expr(self.visit(node.left))
        args = [lhs]

        assert len(node.comparators) >= 1
        op = self.visit(node.ops[-1])
        rhs = make_expr(self.visit(node.comparators[-1]))
        args.append(rhs)

        for i in range(len(node.comparators) - 2, -1, -1):
            lhs = make_expr(self.visit(node.values[i]))
            rhs = BinOpExpr(op=op, lhs=lhs, rhs=rhs)
            op = self.visit(node.ops[i])
            args.append(lhs)

        result = BinOpExpr(op=op, lhs=lhs, rhs=rhs)

        return result

    def visit_IfExp(self, node: ast.IfExp) -> TernaryOpExpr:
        result = TernaryOpExpr(
            condition=make_expr(self.visit(node.test)),
            then_expr=make_expr(self.visit(node.body)),
            else_expr=make_expr(self.visit(node.orelse)),
        )

        return result

    def visit_If(self, node: ast.If) -> If:

        condition_value = gt_utils.meta.ast_eval(node.test, self.externals, default=NOTHING)
        if condition_value is not NOTHING:
            # Compile-time evaluation
            stmts = []
            if condition_value:
                for stmt in node.body:
                    stmts.extend(gt_utils.listify(self.visit(stmt)))
            elif node.orelse:
                for stmt in node.orelse:
                    stmts.extend(gt_utils.listify(self.visit(stmt)))
            result = stmts
        else:
            # run-time evaluation
            main_stmts = []
            for stmt in node.body:
                main_stmts.extend(gt_utils.listify(self.visit(stmt)))
            assert all(isinstance(item, Statement) for item in main_stmts)

            else_stmts = []
            if node.orelse:
                for stmt in node.orelse:
                    else_stmts.extend(gt_utils.listify(self.visit(stmt)))
                assert all(isinstance(item, Statement) for item in else_stmts)

            result = If(
                condition=make_expr(self.visit(node.test)),
                main_body=BlockStmt(stmts=main_stmts),
                else_body=BlockStmt(stmts=else_stmts) if else_stmts else None,
            )

        return result

    # -- Statement nodes --
    def visit_Assign(self, node: ast.Assign) -> list:
        result = []

        # assert len(node.targets) == 1
        # Create decls for temporary fields
        target = []
        for t in node.targets[0].elts if isinstance(node.targets[0], ast.Tuple) else node.targets:
            if isinstance(t, ast.Name) and not self._is_known(t.id):
                field_decl = FieldDecl(
                    name=t.id,
                    data_type=DataType.AUTO,
                    axes=[ax.name for ax in Domain.LatLonGrid().axes],
                    layout_id=DEFAULT_LAYOUT_ID,
                    is_api=False,
                )
                result.append(field_decl)
                self.fields[field_decl.name] = field_decl

            target.append(self.visit(t))

        value = self.visit(node.value)
        if len(target) == 1:
            value = [make_expr(value)]
        else:
            value = [make_expr(item) for item in value]

        assert len(target) == len(value)
        for left, right in zip(target, value):
            result.append(Assign(target=left, value=right))

        return result

    def visit_With(self, node: ast.With) -> list:
        assert len(node.items) == 1
        with_item = node.items[0]
        assert with_item.optional_vars is None
        expr = with_item.context_expr
        assert isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute)
        assert (
            expr.func.attr == "region"
            and isinstance(expr.func.value, ast.Name)
            and expr.func.value.id == "gt"
        )
        assert (
            len(expr.keywords) == 2
            and expr.keywords[0].arg == "iteration"
            and expr.keywords[1].arg == "k_interval"
        )

        stmts = []
        for stmt in node.body:
            stmts.extend(gt_utils.listify(self.visit(stmt)))

        interval_axis = self.domain.sequential_axis.name
        params = [item.arg for item in expr.keywords]
        iteration = IterationOrder[expr.keywords[params.index("iteration")].value.attr]
        bounds = self.visit(expr.keywords[params.index("k_interval")].value)
        interval = make_axis_interval(bounds)
        result = [
            ComputationBlock(
                interval=interval, iteration_order=iteration, body=BlockStmt(stmts=stmts)
            )
        ]

        return result

    def visit_FunctionDef(self, node: ast.FunctionDef) -> list:
        stmts = []
        for stmt in node.body:
            stmts.extend(gt_utils.listify(self.visit(stmt)))
        if not all(isinstance(item, ComputationBlock) for item in stmts):
            assert all(isinstance(item, Statement) for item in stmts)
            result = [
                ComputationBlock(
                    interval=AxisInterval.full_interval(),
                    iteration_order=IterationOrder.PARALLEL,
                    body=BlockStmt(stmts=stmts),
                )
            ]

        else:
            result = stmts
        return result


make_computations = AST2IRVisitor.apply
