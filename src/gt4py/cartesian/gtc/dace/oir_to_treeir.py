# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

from dace import data, dtypes, nodes, symbolic

from gt4py import eve
from gt4py.cartesian.gtc import common, definitions, oir
from gt4py.cartesian.gtc.dace import daceir as dcir, oir_to_tasklet, treeir as tir
from gt4py.cartesian.gtc.dace.symbol_utils import data_type_to_dace_typeclass
from gt4py.cartesian.gtc.dace.utils import get_dace_debuginfo
from gt4py.cartesian.gtc.passes.oir_optimizations import utils as oir_utils


@dataclass
class Context:
    root: tir.TreeRoot
    current_scope: tir.TreeScope

    field_extents: dict[str, definitions.Extent]  # field_name -> Extent
    block_extents: dict[int, definitions.Extent]  # id(horizontal execution) -> Extent


# This could (should?) be a NodeTranslator
# (doesn't really matter for now)
class OIRToTreeIR(eve.NodeVisitor):
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

        ctx.current_scope.children.append(loop)
        ctx.current_scope = loop

        # TODO
        # Split horizontal executions into code blocks to group
        # things like if-statements and while-loops.
        # Remember: add support for regions (HorizontalRestrictions)
        # from the start this time.
        code_blocks = [oir.CodeBlock(label=f"he_{id(node)}", body=node.body)]

        self.visit(code_blocks, ctx=ctx)

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

        parent_scope = ctx.current_scope
        ctx.current_scope.children.append(loop)
        ctx.current_scope = loop

        self.visit(node.horizontal_executions, ctx=ctx)

        ctx.current_scope = parent_scope

    def visit_VerticalLoop(self, node: oir.VerticalLoop, ctx: Context) -> None:
        if node.caches:
            raise NotImplementedError("we don't do caches in this prototype")

        self.visit(node.sections, ctx=ctx, loop_order=node.loop_order)

    def visit_Stencil(self, node: oir.Stencil) -> tir.TreeRoot:
        # question
        # define domain as a symbol here and then pass it down to
        # TreeRoot -> stree's symbols

        # setup the descriptor repository
        containers: dict[str, data.Data] = {}
        symbols: tir.SymbolDict = {}

        for param in node.params:
            if isinstance(param, oir.ScalarDecl):
                containers[param.name] = data.Scalar(
                    data_type_to_dace_typeclass(param.dtype),  # dtype
                    debuginfo=get_dace_debuginfo(param),
                )
                continue

            if isinstance(param, oir.FieldDecl):
                containers[param.name] = data.Array(
                    data_type_to_dace_typeclass(param.dtype),  # dtype
                    get_dace_shape(param, symbols),  # shape
                    strides=get_dace_strides(param, symbols),
                    debuginfo=get_dace_debuginfo(param),
                )
                continue

            raise ValueError(f"Unexpected parameter type {type(param)}.")

        for field in node.declarations:
            containers[field.name] = data.Array(
                data_type_to_dace_typeclass(field.dtype),  # dtype
                get_dace_shape(field, symbols),  # shape
                strides=get_dace_strides(field, symbols),
                transient=True,
                lifetime=dtypes.AllocationLifetime.Persistent,
                debuginfo=get_dace_debuginfo(field),
            )

        tree = tir.TreeRoot(
            name=node.name,
            containers=containers,
            symbols=symbols,
            children=[],
            parent=None,
        )

        # this is ij blocks = horizontal execution
        field_extents, block_extents = oir_utils.compute_extents(node)

        ctx = Context(
            root=tree, current_scope=tree, field_extents=field_extents, block_extents=block_extents
        )

        self.visit(node.vertical_loops, ctx=ctx)

        return ctx.root


def get_dace_shape(field: oir.FieldDecl, symbols: tir.SymbolDict) -> list:
    shape = []
    for axis in dcir.Axis.dims_3d():
        if field.dimensions[axis.to_idx()]:
            symbol = axis.domain_dace_symbol()
            symbols[axis.domain_symbol()] = dtypes.int32
            shape.append(symbol)

    shape.extend([d for d in field.data_dims])
    return shape


def get_dace_strides(field: oir.FieldDecl, symbols: tir.SymbolDict) -> list[symbolic.symbol]:
    dimension_strings = [d for i, d in enumerate("IJK") if field.dimensions[i]]
    data_dimenstion_strings = [f"d{ddim}" for ddim in range(len(field.data_dims))]

    strides = []
    for dim in dimension_strings + data_dimenstion_strings:
        stride = f"__{field.name}_{dim}_stride"
        symbol = symbolic.pystr_to_symbolic(stride)
        symbols[stride] = dtypes.int32
        strides.append(symbol)
    return strides
