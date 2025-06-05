# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

from dace import nodes, subsets, dtypes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
import dace.codegen.control_flow as dcf
from dace.properties import CodeBlock
from dace import Language as lang
from dace import SDFGState, __version__ as dace_version

from gt4py import eve
from gt4py.cartesian.gtc import common
from gt4py.cartesian.gtc.dace import daceir as dcir, treeir as tir


@dataclass
class Context:
    tree: tn.ScheduleTreeRoot
    """A reference to the tree root."""
    current_scope: tn.ScheduleTreeScope
    """A reference to the current scope node."""


class TreeIRToScheduleTree(eve.NodeVisitor):
    # TODO
    # More visitors to come here

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
                        # -1 because bounds are inclusive
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
        map_scope.parent = ctx.current_scope
        ctx.current_scope.children.append(map_scope)
        ctx.current_scope = map_scope

        self.visit(node.children, ctx=ctx)

    def visit_VerticalLoop(self, node: tir.VerticalLoop, ctx: Context) -> None:
        parent_scope = ctx.current_scope

        if node.loop_order == common.LoopOrder.PARALLEL:
            # create map and add to tree

            ctx.tree.symbols[dcir.Axis.K.iteration_symbol()] = dtypes.int32
            map_entry = nodes.MapEntry(
                map=nodes.Map(
                    label=f"vertical_loop_{id(node)}",
                    params=[dcir.Axis.K.iteration_symbol()],
                    # TODO (later)
                    # Ranges have support support for tiling
                    ndrange=subsets.Range(
                        # -1 because bounds are inclusive
                        [(node.bounds_k.start, f"{node.bounds_k.end} - 1", 1)]
                    ),
                )
            )
            map_scope = tn.MapScope(
                node=map_entry,
                children=[],
            )
            map_scope.parent = ctx.current_scope
            ctx.current_scope.children.append(map_scope)
            ctx.current_scope = map_scope
        else:
            # create loop and add it to tree
            if node.loop_order == common.LoopOrder.FORWARD:
                start = node.bounds_k.start
                end = node.bounds_k.end
                update_ops = "+1"
                cond_ops = f"< {end}"
            elif node.loop_order == common.LoopOrder.BACKWARD:
                end = node.bounds_k.start
                start = node.bounds_k.end
                update_ops = "-1"
                cond_ops = f">= {end}"
                breakpoint()
            cfg_for_scope = _make_for_scope(cond_ops, start, update_ops)

            for_scope = tn.ForScope(header=cfg_for_scope, children=[])
            for_scope.parent = ctx.current_scope
            ctx.current_scope.children.append(for_scope)
            ctx.current_scope = for_scope

        self.visit(node.children, ctx=ctx)
        ctx.current_scope = parent_scope

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


def _make_for_scope(condtional_op: str, bound_start: str, update_op) -> dcf.ForScope:
    if not dace_version.startswith("1."):
        raise NotImplementedError("DaCe 2.x detected - please fix below code")

    for_scope = dcf.ForScope(
        condition=CodeBlock(code=f"__k {condtional_op}", language=lang.Python),
        itervar="__k",
        init=f"{bound_start}",
        update=f"__k{update_op}",
        # Unused
        parent=None,  # not Tree parent, CF parent
        dispatch_state=lambda _state: "",
        last_block=False,
        guard=SDFGState(),
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
    # Kill the loop_range test for memlet propgation check going in
    dcf.ForScope.loop_range = lambda self: None
    return for_scope
