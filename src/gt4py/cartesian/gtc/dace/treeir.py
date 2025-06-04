# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py import eve
from gt4py.cartesian.gtc import common, oir


class Bounds(eve.Node):
    start: str
    end: str


class TreeScope(eve.Node):
    children: list


class HorizontalLoop(TreeScope):
    # stuff for ij/loop bounds
    bounds_i: Bounds
    bounds_j: Bounds

    children: list[oir.CodeBlock]


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
