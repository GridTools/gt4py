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

import itertools
from collections.abc import Sequence
from typing import Optional

from . import traits, types as ts


def flatten_tuples(ty: ts.Type) -> list[ts.Type]:
    if isinstance(ty, ts.TupleType):
        return list(itertools.chain(*(flatten_tuples(element) for element in ty.elements)))
    return [ty]


def unflatten_tuples(tys: Sequence[ts.Type], structure: ts.Type) -> ts.Type:
    def helper(tys: Sequence[ts.Type], structure: ts.Type) -> tuple[ts.Type, Sequence[ts.Type]]:
        if isinstance(structure, ts.TupleType):
            remaining = tys
            elements: list[ts.Type] = []
            for se in structure.elements:
                e, remaining = helper(remaining, se)
                elements.append(e)
            return ts.TupleType(elements), remaining
        if tys:
            return tys[0], tys[1:]
        raise ValueError("too few types to fill structure")

    new_ty, remaining = helper(tys, structure)
    if remaining:
        raise ValueError("too many types to fill structure")
    return new_ty


def assign_arguments(
    parameters: Sequence[ts.FunctionParameter], arguments: Sequence[traits.FunctionArgument]
) -> list[traits.FunctionArgument]:
    by_index = {arg.location: arg for arg in arguments if isinstance(arg.location, int)}
    by_name = {arg.location: arg for arg in arguments if isinstance(arg.location, str)}

    assignment: dict[int, traits.FunctionArgument] = {}
    for index, param in enumerate(parameters):
        arg: Optional[traits.FunctionArgument] = None
        if param.positional and index in by_index:
            arg = by_index[index]
        elif param.keyword and param.name in by_name:
            arg = by_name[param.name]
        if arg:
            assignment[index] = arg
        else:
            raise ValueError(f"no argument for function parameter '{param.name}' supplied")

    if len(parameters) < len(arguments):
        raise ValueError("too many arguments to function call")

    return [assignment[i] for i in range(len(arguments))]
