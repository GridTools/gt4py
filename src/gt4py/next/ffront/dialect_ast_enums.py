# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

from gt4py.eve import StrEnum


class Namespace(StrEnum):
    LOCAL = "local"
    CLOSURE = "closure"
    EXTERNAL = "external"


class BinaryOperator(StrEnum):
    ADD = "plus"
    SUB = "minus"
    MULT = "multiplies"
    DIV = "divides"
    FLOOR_DIV = "floordiv"
    BIT_AND = "and_"
    BIT_OR = "or_"
    BIT_XOR = "xor_"
    POW = "power"
    MOD = "mod"

    def __str__(self) -> str:
        if self is self.ADD:
            return "+"
        elif self is self.SUB:
            return "-"
        elif self is self.MULT:
            return "*"
        elif self is self.DIV:
            return "/"
        elif self is self.FLOOR_DIV:
            return "//"
        elif self is self.BIT_AND:
            return "&"
        elif self is self.BIT_XOR:
            return "^"
        elif self is self.BIT_OR:
            return "|"
        elif self is self.POW:
            return "**"
        elif self is self.MOD:
            return "%"
        return "Unknown BinaryOperator"


class UnaryOperator(StrEnum):
    UADD = "plus"
    USUB = "minus"
    NOT = "not_"
    INVERT = "invert"

    def __str__(self) -> str:
        if self is self.UADD:
            return "+"
        elif self is self.USUB:
            return "-"
        elif self is self.NOT:
            return "not"
        elif self is self.INVERT:
            return "~"
        return "Unknown UnaryOperator"
