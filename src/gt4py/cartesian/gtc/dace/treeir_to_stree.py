# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

from dace import nodes, subsets
from dace.sdfg.analysis.schedule_tree import treenodes as tn

from gt4py import eve
from gt4py.cartesian.gtc import common
from gt4py.cartesian.gtc.dace import treeir as tir


@dataclass
class Context:
    tree: tn.ScheduleTreeRoot
    current_scope: tn.ScheduleTreeScope


class TreeIRToScheduleTree(eve.NodeVisitor):
    # TODO
    # More visitors to come here

    def visit_HorizontalLoop(self, node: tir.HorizontalLoop, ctx: Context) -> None:
        # Define ij/loop
        map_entry = nodes.MapEntry(
            map=nodes.Map(
                label=f"horizontal_loop_{id(node)}",
                params=["__i", "__j"],
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

        # for each CodeBlock, add a Tasklet
        for _block in node.children:
            # TODO
            # actually do stuff here
            tasklet = nodes.Tasklet(label="my_tasklet")
            inputs: dict = {}  # dict of connector_name -> memlet
            outputs: dict = {}  # dict of connector_name -> memlet
            tree_tasklet = tn.TaskletNode(
                node=tasklet,
                in_memlets=inputs,
                out_memlets=outputs,
            )
            tree_tasklet.parent = ctx.current_scope
            ctx.current_scope.children.append(tree_tasklet)

    def visit_VerticalLoop(self, node: tir.VerticalLoop, ctx: Context) -> None:
        if node.loop_order == common.LoopOrder.PARALLEL:
            # create map and add to tree

            map_entry = nodes.MapEntry(
                map=nodes.Map(
                    label=f"vertical_loop_{id(node)}",
                    params=["__k"],
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
            # create loop and add it to tree.
            raise NotImplementedError("#todo")

        self.visit(node.children, ctx=ctx)

    def visit_TreeRoot(self, node: tir.TreeRoot) -> tn.ScheduleTreeRoot:
        """
        Construct a schedule tree from TreeIR.
        """
        # setup the descriptor repository
        containers: dict = {}
        symbols: dict = {}
        constants: dict = {}

        # map node.transients to transient arrays in containers

        # map node.arrays to non-transient arrays in containers

        # map scalars to 1d arrays in containers

        # TODO
        # Do we have (compile time) constants?

        # create an empty schedule tree
        tree = tn.ScheduleTreeRoot(
            name=node.name, containers=containers, symbols=symbols, constants=constants, children=[]
        )
        ctx = Context(tree=tree, current_scope=tree)

        self.visit(node.children, ctx=ctx)

        return ctx.tree
