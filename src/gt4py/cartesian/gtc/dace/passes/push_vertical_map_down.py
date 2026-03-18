# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy

from dace.sdfg.analysis.schedule_tree import treenodes as tn

from gt4py.cartesian.gtc.dace.passes import utils


class PushVerticalMapDown(tn.ScheduleNodeVisitor):
    """
    Given a schedule tree with K-JI loops, push the K-loop down into the JI-loops to
    achieve a loop structure suitable for a "K-first" memory layout, or a layout
    where K is the highest value in the layout_map.

    Expected input is something like

    // vertical loop outside
    for k ...
        // horizontal loop(s) inside
        for j,i ...
            // computation here (1)

        for j,i ...
            // computation here (2)

    Given such a loop structure, this pass will push down the vertical loop into the
    horizontal loops. In the above example, the expected output is

    // horizontal loop(s) outside
    for j,i ...
        for k ...
            // computation here (1)

    for j,i ...
        for k ...
            // computation here (2)
    """

    def visit_MapScope(self, scope: tn.MapScope) -> None:
        if not scope.node.map.params[0].startswith("__k"):
            return

        # take refs before moving things around
        parent = scope.parent
        parent_children = scope.parent.children
        k_loop_index = utils.list_index(parent_children, scope)

        for child in scope.children:
            if not isinstance(child, tn.MapScope):
                raise NotImplementedError("We don't expect anything else than (IJ)-MapScopes here.")

            child.children = [
                # New loop with MapEntry (`node`) from parent and children from `child`
                tn.MapScope(
                    node=deepcopy(scope.node), children=[c for c in child.children], parent=child
                )
            ]
            child.parent = parent
            parent_children.insert(k_loop_index, child)
            k_loop_index += 1

        # delete old (now unused) node
        parent_children.remove(scope)
