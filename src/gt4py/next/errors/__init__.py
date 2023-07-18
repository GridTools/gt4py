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

"""Contains the exception classes and other utilities for error handling."""

from . import (  # noqa: module needs to be loaded for pretty printing of uncaught exceptions.
    excepthook,
)
from .excepthook import set_verbose_exceptions
from .exceptions import (
    DSLError,
    InvalidParameterAnnotationError,
    MissingAttributeError,
    MissingParameterAnnotationError,
    UndefinedSymbolError,
    UnsupportedPythonFeatureError,
)


__all__ = [
    "DSLError",
    "InvalidParameterAnnotationError",
    "MissingAttributeError",
    "MissingParameterAnnotationError",
    "UndefinedSymbolError",
    "UnsupportedPythonFeatureError",
    "set_verbose_exceptions",
]
