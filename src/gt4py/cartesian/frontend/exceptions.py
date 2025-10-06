# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from gt4py.cartesian import definitions as gt_definitions
from gt4py.cartesian.frontend import gtscript_frontend, nodes


class GTScriptSyntaxError(gt_definitions.GTSyntaxError):
    def __init__(self, message: str | None, *, loc: nodes.Location | None = None) -> None:
        if not message:
            message = "Syntax error"
            if loc is not None:
                message = f"{message} in '{loc.scope}' (line: {loc.line}, col: {loc.column})"
        super().__init__(message, frontend=gtscript_frontend.GTScriptFrontend.name)
        self.loc = loc


class GTScriptSymbolError(GTScriptSyntaxError):
    def __init__(
        self, name: str, message: str | None = None, *, loc: nodes.Location | None = None
    ) -> None:
        if not message:
            message = f"Unknown symbol '{name}'"
            if loc is not None:
                message = f"{message} in '{loc.scope}' (line: {loc.line}, col: {loc.column})"
        super().__init__(message, loc=loc)
        self.name = name


class GTScriptDefinitionError(GTScriptSyntaxError):
    def __init__(
        self,
        name: str,
        value: Any,
        message: str | None = None,
        *,
        loc: nodes.Location | None = None,
    ) -> None:
        if not message:
            message = f"Invalid definition for '{name}' symbol"
            if loc is not None:
                message = f"{message} in '{loc.scope}' (line: {loc.line}, col: {loc.column})"
        super().__init__(message, loc=loc)
        self.name = name
        self.value = value


class GTScriptValueError(GTScriptDefinitionError):
    def __init__(
        self,
        name: str,
        value: Any,
        message: str | None = None,
        *,
        loc: nodes.Location | None = None,
    ) -> None:
        if not message:
            message = f"Invalid value for '{name}'"
            if loc is not None:
                message = f"{message} in '{loc.scope}' (line: {loc.line}, col: {loc.column})"
        super().__init__(name, value, message, loc=loc)


class GTScriptDataTypeError(GTScriptSyntaxError):
    def __init__(
        self,
        name: str,
        data_type: Any,
        message: str | None = None,
        *,
        loc: nodes.Location | None = None,
    ) -> None:
        if not message:
            message = f"Invalid data type for '{name}' numeric symbol "
            if loc is not None:
                message = f"{message} in '{loc.scope}' (line: {loc.line}, col: {loc.column})"
        super().__init__(message, loc=loc)
        self.name = name
        self.data_type = data_type


class GTScriptAssertionError(gt_definitions.GTSpecificationError):
    def __init__(self, source: list[str], *, loc: nodes.Location | None = None) -> None:
        if loc:
            message = f"Assertion failed at line {loc.line}, col {loc.column}:\n{source}"
        else:
            message = f"Assertion failed.\n{source}"
        super().__init__(message)
        self.loc = loc
