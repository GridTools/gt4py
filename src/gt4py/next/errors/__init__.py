# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Contains the exception classes and other utilities for error handling."""

from . import excepthook  # noqa: F401 [unused-import]
from .exceptions import (
    DSLError,
    InvalidParameterAnnotationError,
    MissingArgumentError,
    MissingAttributeError,
    MissingParameterAnnotationError,
    UndefinedSymbolError,
    UnsupportedPythonFeatureError,
)


__all__ = [
    "DSLError",
    "InvalidParameterAnnotationError",
    "MissingArgumentError",
    "MissingAttributeError",
    "MissingParameterAnnotationError",
    "UndefinedSymbolError",
    "UnsupportedPythonFeatureError",
]
