# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, List, TypeAlias

from dace import data, dtypes, symbolic

from gt4py import eve
from gt4py.cartesian.gtc import common, definitions, oir
from gt4py.cartesian.gtc.dace import oir_to_tasklet, treeir as tir, utils
from gt4py.cartesian.gtc.passes.gtir_k_boundary import compute_k_boundary
from gt4py.cartesian.gtc.passes.oir_optimizations import utils as oir_utils
from gt4py.cartesian.stencil_builder import StencilBuilder


ControlFlow: TypeAlias = (
    oir.HorizontalExecution | oir.While | oir.MaskStmt | oir.HorizontalRestriction
)
"""All control flow OIR nodes"""

DEFAULT_STORAGE_TYPE = {
    dtypes.DeviceType.CPU: dtypes.StorageType.Default,
    dtypes.DeviceType.GPU: dtypes.StorageType.GPU_Global,
}
"""Default dace residency types per device type."""

DEFAULT_MAP_SCHEDULE = {
    dtypes.DeviceType.CPU: dtypes.ScheduleType.Default,
    dtypes.DeviceType.GPU: dtypes.ScheduleType.GPU_Device,
}
"""Default kernel target per device type."""


class OIRToTreeIR(eve.NodeVisitor):
    """
    Translate the GT4Py OIR into a Dace-centric TreeIR.

    TreeIR is built to be a minimum representation of DaCe's Schedule
    Tree. No transformation is done on TreeIR, though should be done
    once the TreeIR has been properly turned into a Schedule Tree.

    This class _does not_ deal with Tasklet representation, it defers that
    work to the OIRToTasklet visitor.
    """

    def __init__(self, builder: StencilBuilder) -> None:
        device_type_translate = {
            "CPU": dtypes.DeviceType.CPU,
            "GPU": dtypes.DeviceType.GPU,
        }

        device_type = builder.backend.storage_info["device"]
        if device_type.upper() not in device_type_translate:
            raise ValueError(f"Unknown device type {device_type}.")

        self._device_type = device_type_translate[device_type.upper()]
        self._api_signature = builder.gtir.api_signature
        self._k_bounds = compute_k_boundary(builder.gtir)
        self._vloop_sections = 0

    def visit_CodeBlock(self, node: oir.CodeBlock, ctx: tir.Context) -> None:
        dace_tasklet, inputs, outputs = oir_to_tasklet.OIRToTasklet().visit_CodeBlock(
            node, root=ctx.root, scope=ctx.current_scope
        )

        tasklet = tir.Tasklet(
            tasklet=dace_tasklet,
            inputs=inputs,
            outputs=outputs,
            parent=ctx.current_scope,
        )
        ctx.current_scope.children.append(tasklet)

    def _group_statements(self, node: ControlFlow) -> list[oir.CodeBlock | ControlFlow]:
        """
        Group the body of a control flow node into CodeBlocks and other ControlFlow.

        This function only groups statements. The job of visiting the groups statements is
        left to the caller.
        """
        statements: List[ControlFlow | oir.CodeBlock | common.Stmt] = []
        groups: List[ControlFlow | oir.CodeBlock] = []

        for statement in node.body:
            if isinstance(statement, ControlFlow):
                if statements != []:
                    groups.append(
                        oir.CodeBlock(label=f"he_{id(node)}_{len(groups)}", body=statements)
                    )
                groups.append(statement)
                statements = []
            else:
                statements.append(statement)

        if statements != []:
            groups.append(oir.CodeBlock(label=f"he_{id(node)}_{len(groups)}", body=statements))

        return groups

    def _insert_evaluation_tasklet(
        self, node: oir.MaskStmt | oir.While, ctx: tir.Context
    ) -> tuple[str, oir.AssignStmt]:
        """Evaluate condition in a separate tasklet to avoid sympy problems down the line."""

        prefix = "while" if isinstance(node, oir.While) else "if"
        condition_name = f"{prefix}_condition_{id(node)}"

        ctx.root.containers[condition_name] = data.Scalar(
            utils.data_type_to_dace_typeclass(common.DataType.BOOL),
            transient=True,
            storage=dtypes.StorageType.Register,
            debuginfo=utils.get_dace_debuginfo(node),
        )

        assignment = oir.AssignStmt(
            left=oir.ScalarAccess(name=condition_name),
            right=node.cond if isinstance(node, oir.While) else node.mask,
        )

        code_block = oir.CodeBlock(label=f"masklet_{id(node)}", body=[assignment])
        self.visit(code_block, ctx=ctx)

        return (condition_name, assignment)

    def visit_HorizontalExecution(self, node: oir.HorizontalExecution, ctx: tir.Context) -> None:
        block_extent = ctx.block_extents[id(node)]

        axis_start_i = f"{block_extent[0][0]}"
        axis_start_j = f"{block_extent[1][0]}"
        axis_end_i = f"({tir.Axis.I.domain_dace_symbol()})  + ({block_extent[0][1]})"
        axis_end_j = f"({tir.Axis.J.domain_dace_symbol()})  + ({block_extent[1][1]})"

        loop = tir.HorizontalLoop(
            bounds_i=tir.Bounds(start=axis_start_i, end=axis_end_i),
            bounds_j=tir.Bounds(start=axis_start_j, end=axis_end_j),
            schedule=DEFAULT_MAP_SCHEDULE[self._device_type],
            children=[],
            parent=ctx.current_scope,
        )

        with loop.scope(ctx):
            # Push local scalars to the tree repository
            for local_scalar in node.declarations:
                ctx.root.containers[local_scalar.name] = data.Scalar(
                    dtype=utils.data_type_to_dace_typeclass(local_scalar.dtype),
                    transient=True,
                    storage=dtypes.StorageType.Register,
                    debuginfo=utils.get_dace_debuginfo(local_scalar),
                )

            groups = self._group_statements(node)
            self.visit(groups, ctx=ctx)

    def visit_MaskStmt(self, node: oir.MaskStmt, ctx: tir.Context) -> None:
        condition_name, _ = self._insert_evaluation_tasklet(node, ctx)

        if_else = tir.IfElse(
            if_condition_code=condition_name, children=[], parent=ctx.current_scope
        )

        with if_else.scope(ctx):
            groups = self._group_statements(node)
            self.visit(groups, ctx=ctx)

    def visit_HorizontalRestriction(
        self, node: oir.HorizontalRestriction, ctx: tir.Context
    ) -> None:
        """Translate `region` concept into If control flow in TreeIR."""
        condition_code = self.visit(node.mask, ctx=ctx)
        if_else = tir.IfElse(
            if_condition_code=condition_code, children=[], parent=ctx.current_scope
        )

        with if_else.scope(ctx):
            groups = self._group_statements(node)
            self.visit(groups, ctx=ctx)

    def visit_HorizontalMask(self, node: common.HorizontalMask, ctx: tir.Context) -> str:
        loop_i = tir.Axis.I.iteration_symbol()
        loop_j = tir.Axis.J.iteration_symbol()

        axis_start_i = "0"
        axis_end_i = tir.Axis.I.domain_symbol()
        axis_start_j = "0"
        axis_end_j = tir.Axis.J.domain_symbol()

        conditions: list[str] = []
        if node.i.start is not None:
            conditions.append(
                f"{loop_i} >= {self.visit(node.i.start, axis_start=axis_start_i, axis_end=axis_end_i)}"
            )
        if node.i.end is not None:
            conditions.append(
                f"{loop_i} < {self.visit(node.i.end, axis_start=axis_start_i, axis_end=axis_end_i)}"
            )
        if node.j.start is not None:
            conditions.append(
                f"{loop_j} >= {self.visit(node.j.start, axis_start=axis_start_j, axis_end=axis_end_j)}"
            )
        if node.j.end is not None:
            conditions.append(
                f"{loop_j} < {self.visit(node.j.end, axis_start=axis_start_j, axis_end=axis_end_j)}"
            )

        return " and ".join(conditions)

    def visit_While(self, node: oir.While, ctx: tir.Context) -> None:
        condition_name, assignment = self._insert_evaluation_tasklet(node, ctx)

        # Re-evaluate the condition as last step of the while loop
        node.body.append(assignment)

        # Use the mask created for conditional check
        while_ = tir.While(
            condition_code=condition_name,
            children=[],
            parent=ctx.current_scope,
        )

        with while_.scope(ctx):
            groups = self._group_statements(node)
            self.visit(groups, ctx=ctx)

    def visit_AxisBound(self, node: oir.AxisBound, axis_start: str, axis_end: str) -> str:
        if node.level == common.LevelMarker.START:
            return f"({axis_start}) + ({node.offset})"

        return f"({axis_end}) + ({node.offset})"

    def visit_Interval(
        self, node: oir.Interval, loop_order: common.LoopOrder, axis_start: str, axis_end: str
    ) -> tir.Bounds:
        start = self.visit(node.start, axis_start=axis_start, axis_end=axis_end)
        end = self.visit(node.end, axis_start=axis_start, axis_end=axis_end)

        if loop_order == common.LoopOrder.BACKWARD:
            return tir.Bounds(start=f"{end} - 1", end=start)

        return tir.Bounds(start=start, end=end)

    def _vertical_loop_schedule(self) -> dtypes.ScheduleType:
        """
        Defines the vertical loop schedule.

        Current strategy is to
          - keep the vertical loop on the host for both, CPU and GPU targets
          - and run it in parallel on CPU and sequential on GPU.
        """
        if self._device_type == dtypes.DeviceType.GPU:
            return dtypes.ScheduleType.Sequential

        return DEFAULT_MAP_SCHEDULE[self._device_type]

    def visit_VerticalLoopSection(
        self, node: oir.VerticalLoopSection, ctx: tir.Context, loop_order: common.LoopOrder
    ) -> None:
        bounds = self.visit(
            node.interval,
            loop_order=loop_order,
            axis_start="0",
            axis_end=tir.Axis.K.domain_dace_symbol(),
        )

        loop = tir.VerticalLoop(
            iteration_variable=eve.SymbolRef(
                f"{tir.Axis.K.iteration_symbol()}_{self._vloop_sections}"
            ),
            loop_order=loop_order,
            bounds_k=bounds,
            schedule=self._vertical_loop_schedule(),
            children=[],
            parent=ctx.current_scope,
        )

        with loop.scope(ctx):
            self.visit(node.horizontal_executions, ctx=ctx)

        self._vloop_sections += 1

    def visit_VerticalLoop(self, node: oir.VerticalLoop, ctx: tir.Context) -> None:
        if node.caches:
            raise NotImplementedError("Caches are not supported in this prototype.")

        self.visit(node.sections, ctx=ctx, loop_order=node.loop_order)

    def visit_Stencil(self, node: oir.Stencil) -> tir.TreeRoot:
        # setup the descriptor repository
        containers: dict[str, data.Data] = {}
        dimensions: dict[str, tuple[bool, bool, bool]] = {}
        symbols: tir.SymbolDict = {}
        shift: dict[str, dict[tir.Axis, int]] = {}  # dict of field_name -> (dict of axis -> shift)

        # Make sure we have the domain symbols of all 3 dimensions defined at all times.
        # They will be used e.g. for loop bounds or data sizing
        for axis in tir.Axis.dims_3d():
            symbols[axis.domain_symbol()] = dtypes.int32

        # this is ij blocks = horizontal execution
        field_extents, block_extents = oir_utils.compute_extents(
            node,
            centered_extent=True,
        )
        # When determining the shape of the array, we have to look at the field extents at large.
        # GT4Py tries to give a precise measure by looking at the horizontal restriction and reduce
        # the extent to the only grid points inside the mask. DaCe requires the real size of the
        # data, hence the call with ignore_horizontal_mask=True.
        field_without_mask_extents = oir_utils.compute_fields_extents(
            node,
            centered_extent=True,
            ignore_horizontal_mask=True,
        )

        missing_api_parameters: list[str] = [p.name for p in self._api_signature]
        for param in node.params:
            missing_api_parameters.remove(param.name)
            if isinstance(param, oir.ScalarDecl):
                containers[param.name] = data.Scalar(
                    dtype=utils.data_type_to_dace_typeclass(param.dtype),
                    debuginfo=utils.get_dace_debuginfo(param),
                )
                continue

            if isinstance(param, oir.FieldDecl):
                field_extent = field_extents[param.name]
                k_bound = self._k_bounds[param.name]
                shift[param.name] = {
                    tir.Axis.I: -field_extent[0][0],
                    tir.Axis.J: -field_extent[1][0],
                    tir.Axis.K: max(k_bound[0], 0),
                }
                containers[param.name] = data.Array(
                    dtype=utils.data_type_to_dace_typeclass(param.dtype),
                    shape=get_dace_shape(
                        param,
                        field_without_mask_extents[param.name],
                        k_bound,
                        symbols,
                    ),
                    strides=get_dace_strides(param, symbols),
                    storage=DEFAULT_STORAGE_TYPE[self._device_type],
                    debuginfo=utils.get_dace_debuginfo(param),
                )
                dimensions[param.name] = param.dimensions
                continue

            raise ValueError(f"Unexpected parameter type {type(param)}.")

        for field in node.declarations:
            field_extent = field_extents[field.name]
            k_bound = self._k_bounds[field.name]
            shift[field.name] = {
                tir.Axis.I: -field_extent[0][0],
                tir.Axis.J: -field_extent[1][0],
                tir.Axis.K: max(k_bound[0], 0),
            }
            # TODO / Dev Note: Persistent memory is an overkill here - we should scope
            # the temporary as close to the tasklets as we can, but any lifetime lower
            # than persistent will yield issues with memory leaks.
            containers[field.name] = data.Array(
                dtype=utils.data_type_to_dace_typeclass(field.dtype),
                shape=get_dace_shape(field, field_extent, k_bound, symbols),
                strides=get_dace_strides(field, symbols),
                transient=True,
                lifetime=dtypes.AllocationLifetime.Persistent,
                storage=DEFAULT_STORAGE_TYPE[self._device_type],
                debuginfo=utils.get_dace_debuginfo(field),
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

        ctx = tir.Context(
            root=tree,
            current_scope=tree,
            field_extents=field_extents,
            block_extents=block_extents,
        )

        self.visit(node.vertical_loops, ctx=ctx)

        return ctx.root

    # Visit expressions for condition code in ControlFlow
    def visit_Cast(self, node: oir.Cast, **kwargs: Any) -> str:
        dtype = utils.data_type_to_dace_typeclass(node.dtype)
        expression = self.visit(node.expr, **kwargs)

        return f"{dtype}({expression})"

    def visit_CartesianOffset(
        self, node: common.CartesianOffset, field: oir.FieldAccess, ctx: tir.Context, **_kwargs: Any
    ) -> str:
        shift = ctx.root.shift[field.name]
        indices: list[str] = []

        offset_dict = node.to_dict()
        for index, axis in enumerate(tir.Axis.dims_3d()):
            if ctx.root.dimensions[field.name][index]:
                shift_str = f" + {shift[axis]}" if shift[axis] != 0 else ""
                iteration_symbol = (
                    tir.k_symbol(ctx.current_scope)
                    if axis == tir.Axis.K
                    else axis.iteration_symbol()
                )
                indices.append(f"{iteration_symbol}{shift_str} + {offset_dict[axis.lower()]}")

        return ", ".join(indices)

    def visit_VariableKOffset(
        self, node: oir.VariableKOffset, field: oir.FieldAccess, ctx: tir.Context, **kwargs: Any
    ) -> str:
        shift = ctx.root.shift[field.name]
        i_shift = f" + {shift[tir.Axis.I]}" if shift[tir.Axis.I] != 0 else ""
        j_shift = f" + {shift[tir.Axis.J]}" if shift[tir.Axis.J] != 0 else ""
        k_shift = f" + {shift[tir.Axis.K]}" if shift[tir.Axis.K] != 0 else ""

        return (
            f"{tir.Axis.I.iteration_symbol()}{i_shift}, "
            f"{tir.Axis.J.iteration_symbol()}{j_shift}, "
            f"{tir.k_symbol(ctx.current_scope)}{k_shift} + {self.visit(node.k, ctx=ctx, **kwargs)}"
        )

    def visit_ScalarAccess(self, node: oir.ScalarAccess, **kwargs: Any) -> str:
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

    def visit_BuiltInLiteral(self, node: common.BuiltInLiteral, **kwargs: Any) -> str:
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

    def visit_TernaryOp(self, node: oir.TernaryOp, **kwargs: Any) -> str:
        condition = self.visit(node.cond, **kwargs)
        if_code = self.visit(node.true_expr, **kwargs)
        else_code = self.visit(node.false_expr, **kwargs)

        return f"({if_code} if {condition} else {else_code})"

    # visitors that should _not_ be called

    def visit_Decl(self, node: oir.Decl, **kwargs: Any) -> None:
        raise RuntimeError("visit_Decl should not be called")

    def visit_FieldDecl(self, node: oir.FieldDecl, **kwargs: Any) -> None:
        raise RuntimeError("visit_FieldDecl should not be called")

    def visit_LocalScalar(self, node: oir.LocalScalar, **kwargs: Any) -> None:
        raise RuntimeError("visit_LocalScalar should not be called")


def get_dace_shape(
    field: oir.FieldDecl,
    extent: definitions.Extent,
    k_bound: tuple[int, int],
    symbols: tir.SymbolDict,
) -> list[symbolic.symbol]:
    shape = []
    for index, axis in enumerate(tir.Axis.dims_3d()):
        if field.dimensions[index]:
            symbol = axis.domain_dace_symbol()

            if axis == tir.Axis.I:
                i_padding = extent[0][1] - extent[0][0]
                if i_padding != 0:
                    shape.append(symbol + i_padding)
                    continue

            if axis == tir.Axis.J:
                j_padding = extent[1][1] - extent[1][0]
                if j_padding != 0:
                    shape.append(symbol + j_padding)
                    continue

            if axis == tir.Axis.K:
                k_padding = max(k_bound[0], 0) + max(k_bound[1], 0)
                if k_padding != 0:
                    shape.append(symbol + k_padding)
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
