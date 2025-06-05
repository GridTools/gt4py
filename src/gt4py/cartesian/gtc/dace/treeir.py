# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TypeAlias

from dace import Memlet, data, nodes, dtypes

from gt4py import eve
from gt4py.cartesian.gtc import common


SymbolDict: TypeAlias = dict[str, dtypes.typeclass]


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

    children: list[Tasklet]  # others to come (horizontal restriction, conditions, while loops, ...)
    # horizontal restriction:
    # - touches the bounds of the (horizontal) loop
    #  -> this can be important for scheduling
    #     (we could do actual loops on CPU vs. masks in the horizontal loop on GPU)
    # conditionals:
    #  -> have no influence on scheduling


class VerticalLoop(TreeScope):
    # header
    loop_order: common.LoopOrder
    bounds_k: Bounds

    children: list[HorizontalLoop]


class TreeRoot(TreeScope):
    name: str

    # Descriptor repository
    containers: dict[str, data.Data]
    """Mapping field/scalar names to data descriptors."""
    symbols: SymbolDict
    """Mapping between type and symbol name."""

    children: list[VerticalLoop]
