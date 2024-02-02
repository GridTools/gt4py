# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

import dataclasses


class Type: ...


class Trait: ...


@dataclasses.dataclass(frozen=True)
class FunctionParameter:
    """Represents a function parameter within callable types."""

    ty: Type
    """The type of the function parameter."""

    name: str
    """The name of the function parameter."""

    positional: bool
    """
    Whether the corresponding argument can be supplied as a positional in a
    function call.
    """

    keyword: bool
    """
    Whether the corresponding argument can be supplied as a keyword argument
    in a function call.
    """


@dataclasses.dataclass(frozen=True)
class FunctionArgument:
    """Represents an argument to a function call."""

    ty: Type
    """The type of the function call argument."""

    location: int | str
    """
    The position of keyword of the function argument.

    For positional arguments, location is an integer equal to the parameter's
    index. For keyword arguments, location is a string equal to the parameter's
    name.
    """
