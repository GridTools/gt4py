# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Optimizable Intermediate Representation (working title)

OIR represents a computation at the level of GridTools stages and multistages,
e.g. stage merging, staged computations to compute-on-the-fly, cache annotations, etc.
"""

import enum
from typing import Dict, List, Optional, Tuple, Union
from pydantic import validator

from devtools import debug  # noqa: F401
from eve import IntEnum, Node, Str, SymbolName
from eve.type_definitions import SymbolRef


from gt4py.gtc import common
from gt4py.gtc.common import LocNode


class Expr(common.Expr):
    # TODO Eve could provide support for making a node abstract
    def __init__(self, *args, **kwargs):
        if type(self) is Expr:
            raise TypeError("Trying to instantiate `Expr` abstract class.")
        super().__init__(*args, **kwargs)


class Stmt(common.Stmt):
    # TODO Eve could provide support for making a node abstract
    def __init__(self, *args, **kwargs):
        if type(self) is Stmt:
            raise TypeError("Trying to instantiate `Stmt` abstract class.")
        super().__init__(*args, **kwargs)


class Literal(common.Literal, Expr):
    pass


class ScalarAccess(common.ScalarAccess, Expr):
    pass


class FieldAccess(common.FieldAccess, Expr):
    pass


class AssignStmt(common.AssignStmt[ScalarAccess, FieldAccess, Expr], Stmt):
    @validator("left")
    def no_horizontal_offset_in_assignment(cls, v):
        if isinstance(v, FieldAccess) and (v.offset.i != 0 or v.offset.j != 0):
            raise ValueError("Lhs of assignment must not have a horizontal offset.")
        return v


class IfStmt(common.IfStmt[Stmt, Expr], Stmt):
    pass


class BinaryOp(common.BinaryOp[Expr], Expr):
    pass


class TernaryOp(common.TernaryOp[Expr], Expr):
    pass


class Decl(LocNode):
    name: SymbolName
    dtype: common.DataType

    def __init__(self, *args, **kwargs):
        if type(self) is Decl:
            raise TypeError("Trying to instantiate `Decl` abstract class.")
        super().__init__(*args, **kwargs)


class FieldDecl(Decl):
    # TODO dimensions
    pass


class ScalarDecl(Decl):
    pass


class AxisBound(Node):
    level: common.LevelMarker
    offset: int = 0

    @classmethod
    def from_start(cls, offset: int):
        return cls(level=common.LevelMarker.START, offset=offset)

    @classmethod
    def from_end(cls, offset: int):
        return cls(level=common.LevelMarker.END, offset=offset)

    @classmethod
    def start(cls):
        return cls.from_start(0)

    @classmethod
    def end(cls):
        return cls.from_end(0)


class StageInterval:
    body: List[Stmt]
    start: AxisBound
    end: AxisBound


class Stage(LocNode):
    """Stage according to the `GridTools execution model<https://gridtools.github.io/gridtools/latest/user_manual/user_manual.html#execution-model>`

    If the execution policy is forward (backward), it is guaranteed that if a stage is executed
    at index k, then all stages of the multistage were already applied to the same column
    with smaller (larger) k.
    There is no guarantee that previous stages of the multistage have not already been applied
    to the indices in the same column with larger (smaller) k.

    Example:
    Assume 2 stages, called `a` and `b`, all of the following pseudo-code snippets are valid:

    - stages in same loop, vertical loop outside
        ```
        for k
            for ij
                a
                b
        ```

    - one loop per stage, vertical loop outside
        ```
        for k
            for ij
                a
        for k
            for ij
                b
        ```

    - one loop per stage, vertical loop inside
        ```
        for ij
            for k
                a
        for ij
            for k
                b
        ```
    """

    intervals: List[StageInterval]


class MultiStage(LocNode):
    stages: List[Stage]
    loop_order: common.LoopOrder
    # caches: List[Union[IJCache,KCache]]


class Stencil(LocNode):
    # TODO remove `__main__.`` from name and use default SymbolName constraint
    name: SymbolName.constrained(r"[a-zA-Z_][\w\.]*")
    params: List[Decl]
    multi_stages: List[MultiStage]
