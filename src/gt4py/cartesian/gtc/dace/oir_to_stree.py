# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dace.sdfg.analysis.schedule_tree import treenodes as tn

from gt4py import eve
from gt4py.cartesian.gtc import oir


class OIRToStree(eve.NodeVisitor):
    stree: tn.ScheduleTreeRoot | None = None

    # TODO
    # More visitors to come here

    def visit_HorizontalExecution(self, node: oir.HorizontalExecution) -> None:
        # self.visit(node.declarations) # noqa
        # self.visit(node.body) # noqa

        raise NotImplementedError("#todo")

    def visit_Interval(self, node: oir.Interval) -> None:
        raise NotImplementedError("#todo")

    def visit_VerticalLoopSection(self, node: oir.VerticalLoopSection) -> None:
        # self.visit(node.interval) # noqa
        # self.visit(node.horizontal_executions) # noqa

        raise NotImplementedError("#todo")

    def visit_VerticalLoop(self, node: oir.VerticalLoop) -> None:
        if node.caches:
            raise NotImplementedError("we don't do caches in this prototype")

        self.visit(node.sections)

        raise NotImplementedError("#todo")

    def visit_Stencil(self, node: oir.Stencil) -> None:
        # setup the descriptor repository
        containers: list = []
        symbols: dict = {}
        constants: dict = {}

        # # assign arrays from node.params
        # -> every field goes into stree.container as an array

        # # assign symbols
        # -> in OirSDFGBuilder, we assign a symbol for every non-field param
        #    (to be evaluated)

        # # assign transient arrays from node.declarations
        # -> these are temporary fields
        # -> they go into stree.container as transient arrays

        # TODO
        # Do we have (compile time) constants?

        # create an empty schedule tree
        self.stree = tn.ScheduleTreeRoot(
            name=node.name,
            containers=containers,
            symbols=symbols,
            constants=constants,
        )

        self.visit(node.vertical_loops)
