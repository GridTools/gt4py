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
from typing import Any, Optional

from gt4py.eve import SourceLocation

from . import formatting


class CompilerError(Exception):
    location: Optional[SourceLocation]

    def __init__(self, location: Optional[SourceLocation], message: str):
        self.location = location
        super().__init__(message)

    @property
    def message(self) -> str:
        return self.args[0]

    def with_location(self, location: Optional[SourceLocation]) -> "CompilerError":
        self.location = location
        return self

    def __str__(self):
        if self.location:
            loc_str = formatting.format_location(self.location, caret=True)
            return f"{self.message}\n{textwrap.indent(loc_str, '  ')}"
        return self.message


class UnsupportedPythonFeatureError(CompilerError):
    feature: str

    def __init__(self, location: Optional[SourceLocation], feature: str):
        super().__init__(location, f"unsupported Python syntax: '{feature}'")
        self.feature = feature


class UndefinedSymbolError(CompilerError):
    sym_name: str

    def __init__(self, location: Optional[SourceLocation], name: str):
        super().__init__(location, f"name '{name}' is not defined")
        self.sym_name = name


class MissingAttributeError(CompilerError):
    attr_name: str

    def __init__(self, location: Optional[SourceLocation], attr_name: str):
        super().__init__(location, f"object does not have attribute '{attr_name}'")
        self.attr_name = attr_name


class CompilerTypeError(CompilerError):
    def __init__(self, location: Optional[SourceLocation], message: str):
        super().__init__(location, message)


class MissingParameterAnnotationError(CompilerTypeError):
    param_name: str

    def __init__(self, location: Optional[SourceLocation], param_name: str):
        super().__init__(location, f"parameter '{param_name}' is missing type annotations")
        self.param_name = param_name


class InvalidParameterAnnotationError(CompilerTypeError):
    param_name: str
    annotated_type: Any

    def __init__(self, location: Optional[SourceLocation], param_name: str, type_: Any):
        super().__init__(
            location, f"parameter '{param_name}' has invalid type annotation '{type_}'"
        )
        self.param_name = param_name
        self.annotated_type = type_


class ArgumentCountError(CompilerTypeError):
    num_excected: int
    num_provided: int

    def __init__(self, location: Optional[SourceLocation], num_expected: int, num_provided: int):
        super().__init__(
            location, f"expected {num_expected} arguments but {num_provided} were provided"
        )
        self.num_expected = num_expected
        self.num_provided = num_provided


class KeywordArgumentError(CompilerTypeError):
    provided_names: str

    def __init__(self, location: Optional[SourceLocation], provided_names: str):
        super().__init__(location, f"unexpected keyword argument(s) '{provided_names}' provided")
        self.provided_names = provided_names
