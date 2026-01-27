# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType

from dace import dtypes, nodes, subsets
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.state import LoopRegion

from gt4py import eve
from gt4py.cartesian.gtc import common
from gt4py.cartesian.gtc.dace import treeir as tir


@dataclass
class Context:
    tree: tn.ScheduleTreeRoot
    """A reference to the tree root."""
    current_scope: tn.ScheduleTreeScope
    """A reference to the current scope node."""


class ContextPushPop:
    """Append the node to the scope, then push/pop the scope"""

    def __init__(self, ctx: Context, node: tn.ScheduleTreeScope) -> None:
        self._ctx = ctx
        self._parent_scope = ctx.current_scope
        self._node = node

    def __enter__(self) -> None:
        self._node.parent = self._parent_scope
        self._parent_scope.children.append(self._node)
        self._ctx.current_scope = self._node

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._ctx.current_scope = self._parent_scope


class TreeIRToScheduleTree(eve.NodeVisitor):
    """Translate TreeIR temporary IR to DaCe's Schedule Tree.

    TreeIR should have undone most of the DSL specificity when translating
    from OIR. This should be a rather direct translation. No transformation
    should happen here, they should all be done on the resulting Schedule Tree.
    """

    def visit_Tasklet(self, node: tir.Tasklet, ctx: Context) -> None:
        tasklet = tn.TaskletNode(
            node=node.tasklet, in_memlets=node.inputs, out_memlets=node.outputs
        )
        tasklet.parent = ctx.current_scope
        ctx.current_scope.children.append(tasklet)

    def visit_HorizontalLoop(self, node: tir.HorizontalLoop, ctx: Context) -> None:
        # Define axis iteration symbols
        for axis in tir.Axis.dims_horizontal():
            ctx.tree.symbols[axis.iteration_symbol()] = dtypes.int32

        dace_map = nodes.Map(
            label=f"horizontal_loop_{id(node)}",
            params=[tir.Axis.J.iteration_symbol(), tir.Axis.I.iteration_symbol()],
            ndrange=subsets.Range(
                [
                    # -1 because range bounds are inclusive
                    (node.bounds_j.start, f"{node.bounds_j.end} - 1", 1),
                    (node.bounds_i.start, f"{node.bounds_i.end} - 1", 1),
                ]
            ),
            schedule=node.schedule,
        )
        map_scope = tn.MapScope(node=nodes.MapEntry(dace_map), children=[])

        with ContextPushPop(ctx, map_scope):
            self.visit(node.children, ctx=ctx)

    def visit_VerticalLoop(self, node: tir.VerticalLoop, ctx: Context) -> None:
        # In any case, define the iteration symbol
        ctx.tree.symbols[node.iteration_variable] = dtypes.int32

        # For serial loops, create a ForScope and add it to the tree
        if node.loop_order != common.LoopOrder.PARALLEL:
            for_scope = tn.ForScope(loop=_loop_region_for(node), children=[])

            with ContextPushPop(ctx, for_scope):
                self.visit(node.children, ctx=ctx)

            return

        # For parallel loops, create a map and add it to the tree
        dace_map = nodes.Map(
            label=f"vertical_loop_{id(node)}",
            params=[node.iteration_variable],
            ndrange=subsets.Range(
                # -1 because range bounds are inclusive
                [(node.bounds_k.start, f"{node.bounds_k.end} - 1", 1)]
            ),
            schedule=node.schedule,
        )
        map_scope = tn.MapScope(node=nodes.MapEntry(dace_map), children=[])

        with ContextPushPop(ctx, map_scope):
            self.visit(node.children, ctx=ctx)

    def visit_IfElse(self, node: tir.IfElse, ctx: Context) -> None:
        if_scope = tn.IfScope(
            condition=CodeBlock(node.if_condition_code),
            children=[],
        )

        with ContextPushPop(ctx, if_scope):
            self.visit(node.children, ctx=ctx)

    def visit_While(self, node: tir.While, ctx: Context) -> None:
        while_scope = tn.WhileScope(loop=_loop_region_while(node), children=[])

        with ContextPushPop(ctx, while_scope):
            self.visit(node.children, ctx=ctx)

    def visit_TreeRoot(self, node: tir.TreeRoot) -> tn.ScheduleTreeRoot:
        """Construct a schedule tree from TreeIR."""
        tree = tn.ScheduleTreeRoot(
            name=node.name,
            containers=node.containers,
            symbols=node.symbols,
            constants={},
            children=[],
        )
        ctx = Context(tree=tree, current_scope=tree)

        self.visit(node.children, ctx=ctx)

        return ctx.tree


def _loop_region_for(node: tir.VerticalLoop) -> LoopRegion:
    """
    Translates a vertical loop into a Dace LoopRegion to be used in `tn.ForScope`.

    :param node: Vertical loop to translate
    :return: DaCe LoopRegion to use in `tn.ForScope`
    """
    plus_minus = "+" if node.loop_order == common.LoopOrder.FORWARD else "-"
    comparison = "<" if node.loop_order == common.LoopOrder.FORWARD else ">="
    iteration_var = node.iteration_variable

    return LoopRegion(
        label=f"vertical_loop_{id(node)}",
        loop_var=iteration_var,
        initialize_expr=CodeBlock(f"{iteration_var} = {node.bounds_k.start}"),
        condition_expr=CodeBlock(f"{iteration_var} {comparison} {node.bounds_k.end}"),
        update_expr=CodeBlock(f"{iteration_var} = {iteration_var} {plus_minus} 1"),
    )


def _loop_region_while(node: tir.While) -> LoopRegion:
    """
    Translates a while loop into a Dace LoopRegion to be used in `tn.WhileScope`.

    :param node: While loop to translate
    :return: DaCe LoopRegion to use in `tn.WhileScope`
    """
    return LoopRegion(label=f"while_loop_{id(node)}", condition_expr=CodeBlock(node.condition_code))
