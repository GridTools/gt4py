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

    def _push_K_loop_in_IJ(self, node: tn.MapScope | tn.ForScope):
        # take refs before moving things around
        parent = node
        grandparent = node.parent
        grandparent_children = node.parent.children
        k_loop_index = utils.list_index(grandparent_children, parent)

        for child in node.children:
            if not isinstance(child, tn.MapScope):
                raise NotImplementedError("We don't expect anything else than (IJ)-MapScopes here.")

            # New loop with MapEntry (`node`) from parent and children from `child`
            if isinstance(node, tn.MapScope):
                new_loop = tn.MapScope(
                    node=deepcopy(parent.node), children=[c for c in child.children], parent=child
                )
            else:
                assert isinstance(node, tn.ForScope)
                new_loop = tn.ForScope(
                    loop=deepcopy(parent.loop),
                    children=[c for c in child.children],
                    parent=child,
                )

            child.children = [new_loop]
            child.parent = grandparent
            grandparent_children.insert(k_loop_index, child)
            k_loop_index += 1

        # delete old (now unused) node
        grandparent_children.remove(node)

    def visit_MapScope(self, node: tn.MapScope):
        if node.node.map.params[0].startswith("__k"):
            self._push_K_loop_in_IJ(node)

    def visit_ForScope(self, node: tn.ForScope):
        if node.loop.loop_variable.startswith("__k"):
            self._push_K_loop_in_IJ(node)
