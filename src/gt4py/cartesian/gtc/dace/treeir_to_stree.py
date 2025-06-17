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

from dace import __version__ as dace_version, dtypes, nodes, sdfg, subsets
from dace.codegen import control_flow as dcf
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn

from gt4py import eve
from gt4py.cartesian.gtc import common
from gt4py.cartesian.gtc.dace import daceir as dcir, treeir as tir


@dataclass
class Context:
    tree: tn.ScheduleTreeRoot
    """A reference to the tree root."""
    current_scope: tn.ScheduleTreeScope
    """A reference to the current scope node."""


class ContextPushPop:
    """Append the node to the scope, then Push/Pop the scope"""

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
    def visit_Tasklet(self, node: tir.Tasklet, ctx: Context) -> None:
        tasklet = tn.TaskletNode(
            node=node.tasklet, in_memlets=node.inputs, out_memlets=node.outputs
        )
        tasklet.parent = ctx.current_scope
        ctx.current_scope.children.append(tasklet)

    def visit_HorizontalLoop(self, node: tir.HorizontalLoop, ctx: Context) -> None:
        # Define ij/loop
        ctx.tree.symbols[dcir.Axis.I.iteration_symbol()] = dtypes.int32
        ctx.tree.symbols[dcir.Axis.J.iteration_symbol()] = dtypes.int32
        map_entry = nodes.MapEntry(
            map=nodes.Map(
                label=f"horizontal_loop_{id(node)}",
                params=[
                    str(dcir.Axis.I.iteration_symbol()),
                    str(dcir.Axis.J.iteration_symbol()),
                ],
                # TODO (later)
                # Ranges have support support for tiling
                ndrange=subsets.Range(
                    [
                        # -1 because range bounds are inclusive
                        (node.bounds_i.start, f"{node.bounds_i.end} - 1", 1),
                        (node.bounds_j.start, f"{node.bounds_j.end} - 1", 1),
                    ]
                ),
            )
        )

        # Add MapScope and update ctx.current_scope
        map_scope = tn.MapScope(
            node=map_entry,
            children=[],
        )

        with ContextPushPop(ctx, map_scope):
            self.visit(node.children, ctx=ctx)

    def visit_VerticalLoop(self, node: tir.VerticalLoop, ctx: Context) -> None:
        if node.loop_order != common.LoopOrder.PARALLEL:
            # For serial loops, create a ForScope and add it to the tree
            for_scope = tn.ForScope(header=_for_scope_header(node), children=[])

            with ContextPushPop(ctx, for_scope):
                self.visit(node.children, ctx=ctx)

            return

        # For parallel loops, create a map and add it to the tree
        ctx.tree.symbols[dcir.Axis.K.iteration_symbol()] = dtypes.int32
        map_entry = nodes.MapEntry(
            map=nodes.Map(
                label=f"vertical_loop_{id(node)}",
                params=[dcir.Axis.K.iteration_symbol()],
                # TODO (later)
                # Ranges have support support for tiling
                ndrange=subsets.Range(
                    # -1 because range bounds are inclusive
                    [(node.bounds_k.start, f"{node.bounds_k.end} - 1", 1)]
                ),
            )
        )
        map_scope = tn.MapScope(node=map_entry, children=[])

        with ContextPushPop(ctx, map_scope):
            self.visit(node.children, ctx=ctx)

    def visit_IfElse(self, node: tir.IfElse, ctx: Context) -> None:
        if_scope = tn.IfScope(
            condition=tn.CodeBlock(node.if_condition_code),
            children=[],
        )

        with ContextPushPop(ctx, if_scope):
            self.visit(node.children, ctx=ctx)

    def visit_While(self, node: tir.While, ctx: Context) -> None:
        while_scope = tn.WhileScope(children=[], header=_while_scope_header(node))

        with ContextPushPop(ctx, while_scope):
            self.visit(node.children, ctx=ctx)

    def visit_TreeRoot(self, node: tir.TreeRoot) -> tn.ScheduleTreeRoot:
        """
        Construct a schedule tree from TreeIR.
        """
        # TODO
        # Do we have (compile time) constants?
        constants: dict = {}

        # create an empty schedule tree
        tree = tn.ScheduleTreeRoot(
            name=node.name,
            containers=node.containers,
            symbols=node.symbols,
            constants=constants,
            children=[],
        )
        ctx = Context(tree=tree, current_scope=tree)

        self.visit(node.children, ctx=ctx)

        return ctx.tree


def _for_scope_header(node: tir.VerticalLoop) -> dcf.ForScope:
    """Header for the tn.ForScope re-using DaCe codegen ForScope.

    Only setup the required data, default or mock the rest.

    TODO: In DaCe 2.x this will be replaced by an SDFG concept which should
    be closer and required less mockup.
    """
    if not dace_version.startswith("1."):
        raise NotImplementedError("DaCe 2.x detected - please fix below code")
    if node.loop_order == common.LoopOrder.PARALLEL:
        raise ValueError("Parallel vertical loops should be translated to maps instead.")

    plus_minus = "+" if node.loop_order == common.LoopOrder.FORWARD else "-"
    comparison = "<" if node.loop_order == common.LoopOrder.FORWARD else ">="
    iteration_var = dcir.Axis.K.iteration_symbol()

    for_scope = dcf.ForScope(
        condition=CodeBlock(
            code=f"{iteration_var} {comparison} {node.bounds_k.end}",
            language=dtypes.Language.Python,
        ),
        itervar=iteration_var,
        init=node.bounds_k.start,
        update=f"{iteration_var} {plus_minus} 1",
        # Unused
        parent=None,  # not Tree parent, CF parent
        dispatch_state=lambda _state: "",
        last_block=False,
        guard=sdfg.SDFGState(),
        body=dcf.GeneralBlock(
            lambda _state: "",
            None,
            True,
            None,
            [],
            [],
            [],
            [],
            [],
            False,
        ),
        init_edges=[],
    )
    # Kill the loop_range test for memlet propagation check going in
    dcf.ForScope.loop_range = lambda self: None
    return for_scope


def _while_scope_header(node: tir.While) -> dcf.WhileScope:
    """Header for the tn.WhileScope re-using DaCe codegen WhileScope.

    Only setup the required data, default or mock the rest.

    TODO: In DaCe 2.x this will be replaced by an SDFG concept which should
    be closer and required less mockup.
    """
    if not dace_version.startswith("1."):
        raise NotImplementedError("DaCe 2.x detected - please fix below code")

    return dcf.WhileScope(
        test=CodeBlock(node.condition_code),
        # Unused
        guard=sdfg.SDFGState(),
        dispatch_state=lambda _state: "",
        parent=None,
        body=dcf.GeneralBlock(
            lambda _state: "",
            None,
            True,
            None,
            [],
            [],
            [],
            [],
            [],
            False,
        ),
        last_block=False,
    )
