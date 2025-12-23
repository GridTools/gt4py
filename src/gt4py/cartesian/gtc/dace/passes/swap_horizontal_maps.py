# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from dace.sdfg.analysis.schedule_tree import treenodes as tn

from gt4py.cartesian.gtc.dace.treeir import Axis


class SwapHorizontalMaps(tn.ScheduleNodeVisitor):
    """
    Given a schedule tree with K-JI loops, swap the IJ for a K-IJ loop.

    Expected input is something like

    // horizontal loop - potentially nested
    map __j, __i in [0:J, 0:I]
        // computation here


    Expected output swaps the indices & range

    // horizontal loop - potentially nested
    map __i, __j in [0:I, 0:J]
        // computation here
    """

    def visit_MapScope(self, node: tn.MapScope):
        if node.node.params[0].startswith(Axis.J.iteration_symbol()) and node.node.params[
            1
        ].startswith(Axis.I.iteration_symbol()):
            # Swap params
            tmp_index = node.node.params[0]
            node.node.params[0] = node.node.params[1]
            node.node.params[1] = tmp_index
            # Swap ranges
            tmp_bounds = node.node.range[0]
            node.node.range[0] = node.node.range[1]
            node.node.range[1] = tmp_bounds

        self.visit(node.children)
