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

# flake8: noqa: F401

"""GridTools storages classes."""

from . import interface, layout
from .interface import empty, from_array, full, ones, zeros  # noqa: F401
from .layout import from_name, register


try:
    from .interface import dace_descriptor  # noqa: F401
except ImportError:
    pass


__all__ = [
    "interface",
    "layout",
    "empty",
    "from_array",
    "full",
    "ones",
    "zeros" "from_name",
    "register",
]

if "dace_descriptor" in globals():
    __all__ += ["dace_descriptor"]
