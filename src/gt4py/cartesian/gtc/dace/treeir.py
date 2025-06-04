# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dace import Memlet, nodes

from gt4py import eve
from gt4py.cartesian.gtc import common, oir


class Bounds(eve.Node):
    start: str
    end: str


class TreeNode(eve.Node):
    parent: TreeScope | None


class TreeScope(TreeNode):
    children: list


class Tasklet(TreeNode):
    tasklet: nodes.Tasklet

    inputs: dict[str, Memlet]
    """Mapping tasklet.in_connectors to Memlets"""
    outputs: dict[str, Memlet]
    """Mapping tasklet.out_connectors to Memlets"""


class HorizontalLoop(TreeScope):
    # stuff for ij/loop bounds
    bounds_i: Bounds
    bounds_j: Bounds

    children: list[Tasklet]  # others to come (conditions, while loops, ...)


class VerticalLoop(TreeScope):
    # header
    loop_order: common.LoopOrder
    bounds_k: Bounds

    children: list[HorizontalLoop]


class TreeRoot(TreeScope):
    name: str

    # Descriptor repository
    transients: list[oir.Temporary]
    arrays: list[oir.FieldDecl]
    scalars: list[oir.ScalarDecl]

    children: list[VerticalLoop]
