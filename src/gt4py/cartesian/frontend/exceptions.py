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

from gt4py.cartesian import definitions as gt_definitions

from . import gtscript_frontend


class GTScriptSyntaxError(gt_definitions.GTSyntaxError):
    def __init__(self, message, *, loc=None):
        super().__init__(message, frontend=gtscript_frontend.GTScriptFrontend.name)
        self.loc = loc


class GTScriptSymbolError(GTScriptSyntaxError):
    def __init__(self, name, message=None, *, loc=None):
        if message is None:
            if loc is None:
                message = "Unknown symbol '{name}' symbol".format(name=name)
            else:
                message = (
                    "Unknown symbol '{name}' symbol in '{scope}' (line: {line}, col: {col})".format(
                        name=name, scope=loc.scope, line=loc.line, col=loc.column
                    )
                )
        super().__init__(message, loc=loc)
        self.name = name


class GTScriptDefinitionError(GTScriptSyntaxError):
    def __init__(self, name, value, message=None, *, loc=None):
        if message is None:
            if loc is None:
                message = "Invalid definition for '{name}' symbol".format(name=name)
            else:
                message = "Invalid definition for '{name}' symbol in '{scope}' (line: {line}, col: {col})".format(
                    name=name, scope=loc.scope, line=loc.line, col=loc.column
                )
        super().__init__(message, loc=loc)
        self.name = name
        self.value = value


class GTScriptValueError(GTScriptDefinitionError):
    def __init__(self, name, value, message=None, *, loc=None):
        if message is None:
            if loc is None:
                message = "Invalid value for '{name}' symbol ".format(name=name)
            else:
                message = (
                    "Invalid value for '{name}' in '{scope}' (line: {line}, col: {col})".format(
                        name=name, scope=loc.scope, line=loc.line, col=loc.column
                    )
                )
        super().__init__(name, value, message, loc=loc)


class GTScriptDataTypeError(GTScriptSyntaxError):
    def __init__(self, name, data_type, message=None, *, loc=None):
        if message is None:
            if loc is None:
                message = "Invalid data type for '{name}' numeric symbol ".format(name=name)
            else:
                message = "Invalid data type for '{name}' numeric symbol in '{scope}' (line: {line}, col: {col})".format(
                    name=name, scope=loc.scope, line=loc.line, col=loc.column
                )
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
