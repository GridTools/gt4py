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

import textwrap
from typing import Any, Optional, TypeVar

from gt4py.eve import SourceLocation

from . import formatting


LocationTraceT = TypeVar("LocationTraceT", SourceLocation, list[SourceLocation], None)


class CompilerError(Exception):
    location_trace: list[SourceLocation]

    def __init__(self, location: LocationTraceT, message: str):
        self.location_trace = CompilerError._make_location_trace(location)
        super().__init__(message)

    @property
    def message(self) -> str:
        return self.args[0]

    @property
    def location(self) -> Optional[SourceLocation]:
        return self.location_trace[0] if self.location_trace else None

    def with_location(self, location: LocationTraceT) -> "CompilerError":
        self.location_trace = CompilerError._make_location_trace(location)
        return self

    def __str__(self):
        if self.location:
            loc_str = formatting.format_location(self.location, caret=True)
            return f"{self.message}\n{textwrap.indent(loc_str, '  ')}"
        return self.message

    @staticmethod
    def _make_location_trace(location: LocationTraceT) -> list[SourceLocation]:
        if isinstance(location, SourceLocation):
            return [location]
        elif isinstance(location, list):
            return location
        elif location is None:
            return []
        else:
            raise TypeError("expected 'SourceLocation', 'list', or 'None' for 'location'")


class UnsupportedPythonFeatureError(CompilerError):
    def __init__(self, location: LocationTraceT, feature: str):
        super().__init__(location, f"unsupported Python syntax: '{feature}'")


class UndefinedSymbolError(CompilerError):
    def __init__(self, location: LocationTraceT, name: str):
        super().__init__(location, f"name '{name}' is not defined")


class MissingAttributeError(CompilerError):
    def __init__(self, location: LocationTraceT, attr_name: str):
        super().__init__(location, f"object does not have attribute '{attr_name}'")


class CompilerTypeError(CompilerError):
    def __init__(self, location: LocationTraceT, message: str):
        super().__init__(location, message)


class MissingParameterAnnotationError(CompilerTypeError):
    def __init__(self, location: LocationTraceT, param_name: str):
        super().__init__(location, f"parameter '{param_name}' is missing type annotations")


class InvalidParameterAnnotationError(CompilerTypeError):
    def __init__(self, location: LocationTraceT, param_name: str, type_: Any):
        super().__init__(
            location, f"parameter '{param_name}' has invalid type annotation '{type_}'"
        )


class ArgumentCountError(CompilerTypeError):
    def __init__(self, location: LocationTraceT, num_expected: int, num_provided: int):
        super().__init__(
            location, f"expected {num_expected} arguments but {num_provided} were provided"
        )


class KeywordArgumentError(CompilerTypeError):
    def __init__(self, location: LocationTraceT, provided_names: str):
        super().__init__(location, f"unexpected keyword argument(s) '{provided_names}' provided")
