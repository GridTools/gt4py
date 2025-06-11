# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, List, TypeAlias

from dace import data, dtypes, nodes, symbolic

from gt4py import eve
from gt4py.cartesian.gtc import common, definitions, oir
from gt4py.cartesian.gtc.dace import daceir as dcir, oir_to_tasklet, treeir as tir
from gt4py.cartesian.gtc.dace.symbol_utils import data_type_to_dace_typeclass
from gt4py.cartesian.gtc.dace.utils import get_dace_debuginfo
from gt4py.cartesian.gtc.passes.oir_optimizations import utils as oir_utils


ControlFlow: TypeAlias = (
    oir.HorizontalExecution | oir.While | oir.MaskStmt | oir.HorizontalRestriction
)
"""All control flow OIR nodes"""


@dataclass
class Context:
    root: tir.TreeRoot
    current_scope: tir.TreeScope

    field_extents: dict[str, definitions.Extent]  # field_name -> Extent
    block_extents: dict[int, definitions.Extent]  # id(horizontal execution) -> Extent


# This could (should?) be a NodeTranslator
# (doesn't really matter for now)
class OIRToTreeIR(eve.NodeVisitor):
    class ContextPushPop:
        """Append the node to the scope, then Push/Pop the scope"""

        def __init__(self, ctx: Context, node: Any):
            self._ctx = ctx
            self._parent_scope = ctx.current_scope
            self._node = node

        def __enter__(self):
            self._node.parent = self._parent_scope
            self._parent_scope.children.append(self._node)
            self._ctx.current_scope = self._node

        def __exit__(self, _exc_type, _exc_value, _traceback):
            self._ctx.current_scope = self._parent_scope

    def visit_CodeBlock(self, node: oir.CodeBlock, ctx: Context) -> None:
        code, inputs, outputs = oir_to_tasklet.generate(node, tree=ctx.root)
        dace_tasklet = nodes.Tasklet(
            label=node.label,
            code=code,
            inputs=inputs.keys(),
            outputs=outputs.keys(),
        )

        tasklet = tir.Tasklet(
            tasklet=dace_tasklet,
            inputs=inputs,
            outputs=outputs,
            parent=ctx.current_scope,
        )
        ctx.current_scope.children.append(tasklet)

    def _group_statements(
        self, node: ControlFlow
    ) -> list[oir.CodeBlock | ControlFlow | common.Stmt]:
        """Group the body of a control flow node into CodeBlocks and other ControlFlow

        Visitor on statements is left to the caller.
        """
        statements: List[ControlFlow | oir.CodeBlock | common.Stmt] = []
        groups: List[ControlFlow | oir.CodeBlock | common.Stmt] = []
        for stmt in node.body:
            if isinstance(stmt, (oir.MaskStmt, oir.While, oir.HorizontalRestriction)):
                if statements != []:
                    groups.append(
                        oir.CodeBlock(label=f"he_{id(node)}_{len(groups)}", body=statements)
                    )
                groups.append(stmt)
                statements = []
            else:
                statements.append(stmt)
        if statements != []:
            groups.append(oir.CodeBlock(label=f"he_{id(node)}_{len(groups)}", body=statements))
        return groups

    def visit_HorizontalExecution(self, node: oir.HorizontalExecution, ctx: Context) -> None:
        # TODO
        # How do we get the domain in here?!
        # use a dace symbol:
        # axis = daceir.Axis.dims_horizontal() #noqa
        # axis.iteration_dace_symbol() axis.domain_dace_symbol()
        # start is always 0
        axis_start_i = "0"
        axis_end_i = dcir.Axis.I.domain_dace_symbol()
        axis_start_j = "0"
        axis_end_j = dcir.Axis.J.domain_dace_symbol()

        extent = ctx.block_extents[id(node)]

        loop = tir.HorizontalLoop(
            bounds_i=tir.Bounds(
                start=f"{axis_start_i} + {extent[0][0]}",
                end=f"{axis_end_i} + {extent[0][1]}",
            ),
            bounds_j=tir.Bounds(
                start=f"{axis_start_j} + {extent[1][0]}",
                end=f"{axis_end_j} + {extent[1][1]}",
            ),
            children=[],
            parent=ctx.current_scope,
        )

        with OIRToTreeIR.ContextPushPop(ctx, loop):
            # Push local scalars to the tree repository
            for local_scalar in node.declarations:
                ctx.root.containers[local_scalar.name] = data.Scalar(
                    data_type_to_dace_typeclass(local_scalar.dtype),  # dtype
                    transient=True,
                    debuginfo=get_dace_debuginfo(local_scalar),
                )

            groups = self._group_statements(node)
            self.visit(groups, ctx=ctx)

    def visit_MaskStmt(self, node: oir.MaskStmt, ctx: Context) -> None:
        if_else = tir.IfElse(
            if_condition_code=self.visit(node.mask), children=[], parent=ctx.current_scope
        )

        with OIRToTreeIR.ContextPushPop(ctx, if_else):
            groups = self._group_statements(node)
            self.visit(groups, ctx=ctx)

    def visit_HorizontalRestriction(self, node: oir.HorizontalRestriction, ctx: Context) -> None:
        """Translate `region` concept into If control flow in TreeIR"""
        condition_code = self.visit(node.mask, ctx=ctx)
        if_else = tir.IfElse(
            if_condition_code=condition_code, children=[], parent=ctx.current_scope
        )
        with OIRToTreeIR.ContextPushPop(ctx, if_else):
            groups = self._group_statements(node)
            self.visit(groups, ctx=ctx)

    def visit_HorizontalMask(self, node: common.HorizontalMask, ctx: Context) -> str:
        # TODO: probably a nope
        loop_i = dcir.Axis.I.iteration_symbol()
        axis_start_i = "0"
        axis_end_i = dcir.Axis.I.domain_symbol()
        loop_j = dcir.Axis.J.iteration_symbol()
        axis_start_j = "0"
        axis_end_j = dcir.Axis.J.domain_symbol()

        cond = ""
        if node.i.start is not None:
            cond += f"{loop_i} >= {self.visit(node.i.start, axis_start=axis_start_i, axis_end=axis_end_i)}"
        if node.i.end is not None:
            if cond != "":
                cond += " and "
            cond += (
                f"{loop_i} < {self.visit(node.i.end, axis_start=axis_start_i, axis_end=axis_end_i)}"
            )
        if node.j.start is not None:
            if cond != "":
                cond += " and "
            cond += f"{loop_j} >= {self.visit(node.j.start, axis_start=axis_start_j, axis_end=axis_end_j)}"
        if node.j.start is not None:
            if cond != "":
                cond += " and "
            cond += f"{loop_j} >= {self.visit(node.j.end, axis_start=axis_start_j, axis_end=axis_end_j)}"

        return cond

    def visit_While(self, node: oir.While, ctx: Context) -> None:
        while_ = tir.While(
            condition_code=self.visit(node.cond, ctx=ctx),
            children=[],
            parent=ctx.current_scope,
        )

        with OIRToTreeIR.ContextPushPop(ctx, while_):
            groups = self._group_statements(node)
            self.visit(groups, ctx=ctx)

    def visit_AxisBound(self, node: oir.AxisBound, axis_start: str, axis_end: str) -> str:
        if node.level == common.LevelMarker.START:
            return f"{axis_start} + {node.offset}"

        return f"{axis_end} + {node.offset}"

    def visit_Interval(
        self, node: oir.Interval, loop_order: common.LoopOrder, axis_start: str, axis_end: str
    ) -> tir.Bounds:
        start = self.visit(node.start, axis_start=axis_start, axis_end=axis_end)
        end = self.visit(node.end, axis_start=axis_start, axis_end=axis_end)

        if loop_order == common.LoopOrder.BACKWARD:
            return tir.Bounds(start=f"{end} - 1", end=start)

        return tir.Bounds(start=start, end=end)

    def visit_VerticalLoopSection(
        self, node: oir.VerticalLoopSection, ctx: Context, loop_order: common.LoopOrder
    ) -> None:
        # TODO
        # How do we get the domain in here?!
        # Axis.domain_dace_symbol() #noqa
        # start is always 0
        bounds = self.visit(
            node.interval,
            loop_order=loop_order,
            axis_start="0",
            axis_end=dcir.Axis.K.domain_dace_symbol(),
        )

        loop = tir.VerticalLoop(
            loop_order=loop_order, bounds_k=bounds, children=[], parent=ctx.current_scope
        )

        with OIRToTreeIR.ContextPushPop(ctx, loop):
            self.visit(node.horizontal_executions, ctx=ctx)

    def visit_VerticalLoop(self, node: oir.VerticalLoop, ctx: Context) -> None:
        if node.caches:
            raise NotImplementedError("we don't do caches in this prototype")

        self.visit(node.sections, ctx=ctx, loop_order=node.loop_order)

    def visit_Stencil(self, node: oir.Stencil) -> tir.TreeRoot:
        # setup the descriptor repository
        containers: dict[str, data.Data] = {}
        dimensions: dict[str, tuple[bool, bool, bool]] = {}
        symbols: tir.SymbolDict = {}
        shift: dict[str, tuple[int, int]] = {}

        # this is ij blocks = horizontal execution
        field_extents, block_extents = oir_utils.compute_extents(node)

        for param in node.params:
            if isinstance(param, oir.ScalarDecl):
                containers[param.name] = data.Scalar(
                    data_type_to_dace_typeclass(param.dtype),  # dtype
                    debuginfo=get_dace_debuginfo(param),
                )
                continue

            if isinstance(param, oir.FieldDecl):
                extent = field_extents[param.name]
                shift[param.name] = (-extent[0][0], -extent[1][0])
                containers[param.name] = data.Array(
                    data_type_to_dace_typeclass(param.dtype),  # dtype
                    get_dace_shape(param, extent, symbols),  # shape
                    strides=get_dace_strides(param, symbols),
                    debuginfo=get_dace_debuginfo(param),
                )
                dimensions[param.name] = param.dimensions
                continue

            raise ValueError(f"Unexpected parameter type {type(param)}.")

        for field in node.declarations:
            extent = field_extents[field.name]
            shift[field.name] = (-extent[0][0], -extent[1][0])
            containers[field.name] = data.Array(
                data_type_to_dace_typeclass(field.dtype),  # dtype
                get_dace_shape(field, extent, symbols),  # shape
                strides=get_dace_strides(field, symbols),
                transient=True,
                lifetime=dtypes.AllocationLifetime.Persistent,
                debuginfo=get_dace_debuginfo(field),
            )
            dimensions[field.name] = field.dimensions

        tree = tir.TreeRoot(
            name=node.name,
            containers=containers,
            dimensions=dimensions,
            shift=shift,
            symbols=symbols,
            children=[],
            parent=None,
        )

        ctx = Context(
            root=tree,
            current_scope=tree,
            field_extents=field_extents,
            block_extents=block_extents,
        )

        self.visit(node.vertical_loops, ctx=ctx)

        return ctx.root

    # Visit expressions for condition code in ControlFlow
    def visit_Cast(self, node: oir.Cast, **kwargs: Any) -> str:
        dtype = data_type_to_dace_typeclass(node.dtype)
        expression = self.visit(node.expr, **kwargs)

        return f"{dtype}({expression})"

    def visit_CartesianOffset(
        self, node: common.CartesianOffset, field: oir.FieldAccess, ctx: Context, **_kwargs: Any
    ) -> str:
        shift = ctx.root.shift[field.name]
        indices: list[str] = []

        offset_dict = node.to_dict()
        for index, axis in enumerate(dcir.Axis.dims_3d()):
            if ctx.root.dimensions[field.name][index]:
                shift_str = f" + {shift[index]}" if index < 2 and shift[index] != 0 else ""
                indices.append(
                    f"{axis.iteration_symbol()}{shift_str} + {offset_dict[axis.lower()]}"
                )

        return ", ".join(indices)

    def visit_VariableKOffset(
        self, node: oir.VariableKOffset, field: oir.FieldAccess, ctx: Context, **kwargs: Any
    ) -> str:
        shift = ctx.root.shift[field.name]
        i_shift = f" + {shift[0]}" if shift[0] != 0 else ""
        j_shift = f" + {shift[1]}" if shift[1] != 0 else ""
        return (
            f"{dcir.Axis.I.iteration_symbol()}{i_shift}, "
            f"{dcir.Axis.J.iteration_symbol()}{j_shift}, "
            f"{dcir.Axis.K.iteration_symbol()} + {self.visit(node.k, **kwargs)}"
        )

    def visit_ScalarAccess(self, node: oir.ScalarAccess, **_kwargs: Any) -> str:
        return f"{node.name}"

    def visit_FieldAccess(self, node: oir.FieldAccess, **kwargs: Any) -> str:
        if node.data_index:
            raise NotImplementedError("Data dimensions aren't supported yet.")

        if "field" in kwargs:
            kwargs.pop("field")

        field_name = node.name
        offsets = self.visit(node.offset, field=node, **kwargs)
        return f"{field_name}[{offsets}]"

    def visit_Literal(self, node: oir.Literal, **kwargs: Any) -> str:
        if type(node.value) is str:
            # Note: isinstance(node.value, str) also matches the string enum `BuiltInLiteral`
            # which we don't want to match because it returns lower-case `true`, which isn't
            # defined in (python) tasklet code.
            return node.value

        return self.visit(node.value, **kwargs)

    def visit_BuiltInLiteral(self, node: common.BuiltInLiteral, **_kwargs: Any) -> str:
        if node == common.BuiltInLiteral.TRUE:
            return "True"
        if node == common.BuiltInLiteral.FALSE:
            return "False"
        raise NotImplementedError(f"Not implemented BuiltInLiteral '{node}' encountered.")

    def visit_UnaryOp(self, node: oir.UnaryOp, **kwargs: Any) -> str:
        expression = self.visit(node.expr, **kwargs)

        return f"{node.op}({expression})"

    def visit_BinaryOp(self, node: oir.BinaryOp, **kwargs: Any) -> str:
        left = self.visit(node.left, **kwargs)
        right = self.visit(node.right, **kwargs)

        return f"({left} {node.op.value} {right})"

    def visit_TernaryOp(self, node: oir.TernaryOp, **kwargs):
        condition = self.visit(node.cond, **kwargs)
        if_code = self.visit(node.true_expr, **kwargs)
        else_code = self.visit(node.false_expr, **kwargs)

        return f"({if_code} if {condition} else {else_code})"

    # visitor that should _not_ be called

    def visit_Decl(self, node: oir.Decl):
        raise RuntimeError("visit_Decl should not be called")

    def visit_FieldDecl(self, node: oir.FieldDecl):
        raise RuntimeError("visit_FieldDecl should not be called")

    def visit_LocalScalar(self, node: oir.LocalScalar):
        raise RuntimeError("visit_LocalScalar should not be called")


def get_dace_shape(
    field: oir.FieldDecl, extent: definitions.Extent, symbols: tir.SymbolDict
) -> list:
    shape = []
    for index, axis in enumerate(dcir.Axis.dims_3d()):
        if field.dimensions[index]:
            symbol = axis.domain_dace_symbol()
            symbols[axis.domain_symbol()] = dtypes.int32

            if axis == dcir.Axis.I:
                i_padding = extent[0][1] - extent[0][0]
                if i_padding != 0:
                    shape.append(symbol + i_padding)
                    continue

            if axis == dcir.Axis.J:
                j_padding = extent[1][1] - extent[1][0]
                if j_padding != 0:
                    shape.append(symbol + j_padding)
                    continue

            shape.append(symbol)

    shape.extend([d for d in field.data_dims])
    return shape


def get_dace_strides(field: oir.FieldDecl, symbols: tir.SymbolDict) -> list[symbolic.symbol]:
    dimension_strings = [d for i, d in enumerate("IJK") if field.dimensions[i]]
    data_dimension_strings = [f"d{ddim}" for ddim in range(len(field.data_dims))]

    strides = []
    for dim in dimension_strings + data_dimension_strings:
        stride = f"__{field.name}_{dim}_stride"
        symbol = symbolic.pystr_to_symbolic(stride)
        symbols[stride] = dtypes.int32
        strides.append(symbol)
    return strides
