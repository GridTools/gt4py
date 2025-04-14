# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
The list of exception classes used in the library.

Exception classes that represent errors within an IR go here as a subclass of
:class:`DSLError`. Exception classes that represent other errors, like
the builtin ValueError, go here as well, although you should use Python's
builtin error classes if you can. Exception classes that are specific to a
certain submodule and have no use for the entire application may be better off
in that submodule as opposed to being in this file.
"""

from __future__ import annotations

import textwrap
from typing import Any, Optional

from gt4py.eve import SourceLocation

from . import formatting


class GT4PyError(Exception):
    @property
    def message(self) -> str:
        return self.args[0]


class DSLError(GT4PyError):
    location: Optional[SourceLocation]

    def __init__(self, location: Optional[SourceLocation], message: str) -> None:
        self.location = location
        super().__init__(message)

    def with_location(self, location: Optional[SourceLocation]) -> DSLError:
        self.location = location
        return self

    def __str__(self) -> str:
        if self.location:
            loc_str = formatting.format_location(self.location, show_caret=True)
            return f"{self.message}\n{textwrap.indent(loc_str, '  ')}"
        return self.message


class UnsupportedPythonFeatureError(DSLError):
    feature: str

    def __init__(self, location: Optional[SourceLocation], feature: str) -> None:
        super().__init__(location, f"Unsupported Python syntax: '{feature}'.")
        self.feature = feature


class UndefinedSymbolError(DSLError):
    sym_name: str

    def __init__(self, location: Optional[SourceLocation], name: str) -> None:
        super().__init__(location, f"Name '{name}' is not defined.")
        self.sym_name = name


class MissingAttributeError(DSLError):
    attr_name: str

    def __init__(self, location: Optional[SourceLocation], attr_name: str) -> None:
        super().__init__(location, f"Object does not have attribute '{attr_name}'.")
        self.attr_name = attr_name


class MissingArgumentError(DSLError):
    arg_name: str
    is_kwarg: bool

    def __init__(self, location: Optional[SourceLocation], arg_name: str, is_kwarg: bool) -> None:
        super().__init__(
            location, f"Expected {'keyword-' if is_kwarg else ''}argument '{arg_name}'."
        )
        self.attr_name = arg_name
        self.is_kwarg = is_kwarg


class TypeError_(DSLError):
    def __init__(self, location: Optional[SourceLocation], message: str) -> None:
        super().__init__(location, message)


class MissingParameterAnnotationError(TypeError_):
    param_name: str

    def __init__(self, location: Optional[SourceLocation], param_name: str) -> None:
        super().__init__(location, f"Parameter '{param_name}' is missing type annotations.")
        self.param_name = param_name


class InvalidParameterAnnotationError(TypeError_):
    param_name: str
    annotated_type: Any

    def __init__(self, location: Optional[SourceLocation], param_name: str, type_: Any) -> None:
        super().__init__(
            location, f"Parameter '{param_name}' has invalid type annotation '{type_}'."
        )
        self.param_name = param_name
        self.annotated_type = type_


class CompilationError(GT4PyError):
    def __init__(self, compilation_error: str) -> None:
        super().__init__(f"See attached compilation log.\n{compilation_error}")
