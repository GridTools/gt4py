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

import linecache
import textwrap

import gt4py.next as gtx
from gt4py.eve import codegen
from gt4py.next.ffront import program_ast as past, type_translation
from gt4py.next.otf import stages
from gt4py.next.type_system import type_info


def past_to_fun_def(past_closure: stages.PastClosure):
    node = past_closure.past_node
    arg_types = [type_translation.from_value(arg) for arg in past_closure.args]
    kwarg_types = [
        type_translation.from_value(v)
        for k, v in past_closure.kwargs.items()
        if k not in ("offset_provider", "column_axis")
    ]
    inout_types = list(type_info.flatten(arg_types + kwarg_types))
    dims = set(
        i for j in [type_info.extract_dims(inout_type) for inout_type in inout_types] for i in j
    )
    source_code = ProgamFuncGen.apply(node)

    filename = "<generated>"
    globalns = {dim.value: dim for dim in dims}
    globalns |= gtx.__dict__
    globalns |= past_closure.closure_vars
    localns: dict = {}
    code_obj = compile(source_code, filename, "exec")
    exec(code_obj, globalns, localns)
    lines = [line + "\n" for line in source_code.splitlines()]
    linecache.cache[filename] = (len(source_code), None, lines, filename)
    function_definition = localns[str(node.id)]
    linecache.cache[filename] = (
        len(source_code),
        None,
        [line + "\n" for line in source_code.splitlines()],
        filename,
    )
    return function_definition


class ProgamFuncGen(codegen.TemplatedGenerator):
    def visit_Program(self, node: past.Program, **kwargs) -> str:
        imports = "from __future__ import annotations\nfrom gt4py.next import *"
        params = self.visit(node.params)
        signature = ", ".join(params)
        body = textwrap.indent("\n".join(self.visit(node.body)), prefix=" " * 4)
        return f"{imports}\n\n\ndef {node.id}({signature}) -> None:\n{body}"

    Symbol = codegen.FormatTemplate("{id}: {type}")

    def visit_Call(self, node: past.Call, **kwargs) -> str:
        args_joined = ", ".join(self.visit(node.args))
        kwargs_list = [f"{name}={self.visit(value)}" for name, value in node.kwargs.items()]
        kwargs_joined = ", ".join(kwargs_list)
        params = ", ".join([args_joined, kwargs_joined])
        return f"{self.visit(node.func)}({params})"

    Name = codegen.FormatTemplate("{id}")