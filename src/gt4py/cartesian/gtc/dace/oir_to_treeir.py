# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

from dace import nodes

from gt4py import eve
from gt4py.cartesian.gtc import common, definitions, oir
from gt4py.cartesian.gtc.dace import oir_to_tasklet, treeir as tir
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
        code, inputs, outputs = oir_to_tasklet.generate(node)
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
        axis_start_i = "0"
        axis_end_i = "__I"
        axis_start_j = "0"
        axis_end_j = "__J"

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
            return tir.Bounds(start=f"{end} - 1", end=f"{start} - 1")

        return tir.Bounds(start=start, end=end)

    def visit_VerticalLoopSection(
        self, node: oir.VerticalLoopSection, ctx: Context, loop_order: common.LoopOrder
    ) -> None:
        # TODO
        # How do we get the domain in here?!
        bounds = self.visit(node.interval, loop_order=loop_order, axis_start="0", axis_end="__K")

        loop = tir.VerticalLoop(
            loop_order=loop_order, bounds_k=bounds, children=[], parent=ctx.current_scope
        )

        ctx.current_scope.children.append(loop)
        ctx.current_scope = loop

        self.visit(node.horizontal_executions, ctx=ctx)

    def visit_VerticalLoop(self, node: oir.VerticalLoop, ctx: Context) -> None:
        if node.caches:
            raise NotImplementedError("we don't do caches in this prototype")

        self.visit(node.sections, ctx=ctx, loop_order=node.loop_order)

    def visit_Stencil(self, node: oir.Stencil) -> tir.TreeRoot:
        tree = tir.TreeRoot(
            name=node.name,
            transients=node.declarations,
            arrays=[p for p in node.params if isinstance(p, oir.FieldDecl)],
            scalars=[p for p in node.params if isinstance(p, oir.ScalarDecl)],
            children=[],
            parent=None,
        )

        field_extents, block_extents = oir_utils.compute_extents(node)

        ctx = Context(
            root=tree, current_scope=tree, field_extents=field_extents, block_extents=block_extents
        )

        self.visit(node.vertical_loops, ctx=ctx)

        return ctx.root
