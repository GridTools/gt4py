# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.cartesian import definitions as gt_definitions
from gt4py.cartesian.frontend import gtscript_frontend


class GTScriptSyntaxError(gt_definitions.GTSyntaxError):
    def __init__(self, message, *, loc=None):
        super().__init__(message, frontend=gtscript_frontend.GTScriptFrontend.name)
        self.loc = loc


class GTScriptSymbolError(GTScriptSyntaxError):
    def __init__(self, name, message=None, *, loc=None):
        if message is None:
            if loc is None:
                message = f"Unknown symbol '{name}' symbol"
            else:
                message = f"Unknown symbol '{name}' symbol in '{loc.scope}' (line: {loc.line}, col: {loc.column})"
        super().__init__(message, loc=loc)
        self.name = name


class GTScriptDefinitionError(GTScriptSyntaxError):
    def __init__(self, name, value, message=None, *, loc=None):
        if message is None:
            if loc is None:
                message = f"Invalid definition for '{name}' symbol"
            else:
                message = f"Invalid definition for '{name}' symbol in '{loc.scope}' (line: {loc.line}, col: {loc.column})"
        super().__init__(message, loc=loc)
        self.name = name
        self.value = value


class GTScriptValueError(GTScriptDefinitionError):
    def __init__(self, name, value, message=None, *, loc=None):
        if message is None:
            if loc is None:
                message = f"Invalid value for '{name}' symbol "
            else:
                message = f"Invalid value for '{name}' in '{loc.scope}' (line: {loc.line}, col: {loc.column})"
        super().__init__(name, value, message, loc=loc)


class GTScriptDataTypeError(GTScriptSyntaxError):
    def __init__(self, name, data_type, message=None, *, loc=None):
        if message is None:
            if loc is None:
                message = f"Invalid data type for '{name}' numeric symbol "
            else:
                message = f"Invalid data type for '{name}' numeric symbol in '{loc.scope}' (line: {loc.line}, col: {loc.column})"
        super().__init__(message, loc=loc)
        self.name = name
        self.data_type = data_type


class GTScriptAssertionError(gt_definitions.GTSpecificationError):
    def __init__(self, source, *, loc=None):
        if loc:
            message = f"Assertion failed at line {loc.line}, col {loc.column}:\n{source}"
        else:
            message = f"Assertion failed.\n{source}"
        super().__init__(message)
        self.loc = loc
