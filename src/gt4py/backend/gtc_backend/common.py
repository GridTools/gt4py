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

import enum

from eve import IntEnum, StrEnum


class AssignmentKind(StrEnum):
    """Kind of assignment: plain or combined with operations."""

    PLAIN = "="
    ADD = "+="
    SUB = "-="
    MUL = "*="
    DIV = "/="


@enum.unique
class UnaryOperator(StrEnum):
    """Unary operator indentifier."""

    POS = "+"
    NEG = "-"


@enum.unique
class BinaryOperator(StrEnum):
    """Binary operator identifier."""

    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"


@enum.unique
class DataType(IntEnum):
    """Data type identifier."""

    # IDs from dawn
    INVALID = 0
    AUTO = 1
    BOOLEAN = 2
    INT32 = 3
    FLOAT32 = 4
    FLOAT64 = 5
    UINT32 = 6


@enum.unique
class LoopOrder(IntEnum):
    """Loop order identifier."""

    FORWARD = 1
    BACKWARD = 2


@enum.unique
class LocationType(IntEnum):
    Vertex = 0
    Edge = 1
    Cell = 2
    NoLocation = 3


@enum.unique
class BuiltInLiteral(IntEnum):
    MAX_VALUE = 0
    MIN_VALUE = 1
    ZERO = 2
    ONE = 3


@enum.unique
class LevelMarker(StrEnum):
    START = "start"
    END = "end"
