# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, TypeAlias

from dace import Memlet, data, dtypes, nodes

from gt4py import eve
from gt4py.cartesian.gtc import common, definitions
from gt4py.cartesian.gtc.dace import utils


SymbolDict: TypeAlias = dict[str, dtypes.typeclass]


@dataclass
class Context:
    root: TreeRoot
    current_scope: TreeScope

    field_extents: dict[str, definitions.Extent]  # field_name -> Extent
    block_extents: dict[int, definitions.Extent]  # id(horizontal execution) -> Extent


class ContextPushPop:
    """Append the node to the scope, then push/pop the scope."""

    def __init__(self, ctx: Context, node: TreeScope) -> None:
        self._ctx = ctx
        self._parent_scope = ctx.current_scope
        self._node = node

    def __enter__(self):
        self._node.parent = self._parent_scope
        self._parent_scope.children.append(self._node)
        self._ctx.current_scope = self._node

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self._ctx.current_scope = self._parent_scope


class Axis(eve.StrEnum):
    I = "I"  # noqa: E741 [ambiguous-variable-name]
    J = "J"
    K = "K"

    def domain_symbol(self) -> eve.SymbolRef:
        return eve.SymbolRef("__" + self.upper())

    def iteration_symbol(self) -> eve.SymbolRef:
        return eve.SymbolRef("__" + self.lower())

    def tile_symbol(self) -> eve.SymbolRef:
        return eve.SymbolRef("__tile_" + self.lower())

    @staticmethod
    def dims_3d() -> Generator[Axis, None, None]:
        yield from [Axis.I, Axis.J, Axis.K]

    @staticmethod
    def dims_horizontal() -> Generator[Axis, None, None]:
        yield from [Axis.I, Axis.J]

    def to_idx(self) -> int:
        return [Axis.I, Axis.J, Axis.K].index(self)

    def domain_dace_symbol(self):
        return utils.get_dace_symbol(self.domain_symbol())

    def iteration_dace_symbol(self):
        return utils.get_dace_symbol(self.iteration_symbol())

    def tile_dace_symbol(self):
        return utils.get_dace_symbol(self.tile_symbol())


class Bounds(eve.Node):
    start: str
    end: str


class TreeNode(eve.Node):
    parent: TreeScope | None


class TreeScope(TreeNode):
    children: list[TreeScope | TreeNode]

    def scope(self, ctx: Context) -> ContextPushPop:
        return ContextPushPop(ctx, self)


class Tasklet(TreeNode):
    tasklet: nodes.Tasklet

    inputs: dict[str, Memlet]
    """Mapping tasklet.in_connectors to Memlets"""
    outputs: dict[str, Memlet]
    """Mapping tasklet.out_connectors to Memlets"""


class IfElse(TreeScope):
    # This should become an if/else, someday, so I am naming it if/else in hope
    # to see it before my bodily demise
    if_condition_code: str
    """Condition as ScheduleTree worthy code"""


class While(TreeScope):
    condition_code: str
    """Condition as ScheduleTree worthy code"""


class HorizontalLoop(TreeScope):
    bounds_i: Bounds
    bounds_j: Bounds

    schedule: dtypes.ScheduleType


class VerticalLoop(TreeScope):
    loop_order: common.LoopOrder
    bounds_k: Bounds

    schedule: dtypes.ScheduleType


class TreeRoot(TreeScope):
    name: str

    containers: dict[str, data.Data]
    """Mapping field/scalar names to data descriptors."""

    dimensions: dict[str, tuple[bool, bool, bool]]
    """Mapping field names to shape-axis."""

    shift: dict[str, dict[Axis, int]]
    """Mapping field names to dict[axis] -> shift."""

    symbols: SymbolDict
    """Mapping between type and symbol name."""
