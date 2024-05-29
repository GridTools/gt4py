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

from typing import Any, Mapping

import dace

from gt4py.next.common import Connectivity, Dimension, DimensionKind
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace_fieldview import gtir_to_tasklet
from gt4py.next.type_system import type_specifications as ts


def as_dace_type(type_: ts.ScalarType) -> dace.typeclass:
    """Converts GT4Py scalar type to corresponding DaCe type."""
    match type_.kind:
        case ts.ScalarKind.BOOL:
            return dace.bool_
        case ts.ScalarKind.INT32:
            return dace.int32
        case ts.ScalarKind.INT64:
            return dace.int64
        case ts.ScalarKind.FLOAT32:
            return dace.float32
        case ts.ScalarKind.FLOAT64:
            return dace.float64
        case _:
            raise ValueError(f"Scalar type '{type_}' not supported.")


def as_scalar_type(typestr: str) -> ts.ScalarType:
    """Obtain GT4Py scalar type from generic numpy string representation."""
    try:
        kind = getattr(ts.ScalarKind, typestr.upper())
    except AttributeError as ex:
        raise ValueError(f"Data type {typestr} not supported.") from ex
    return ts.ScalarType(kind)


def connectivity_identifier(name: str) -> str:
    return f"connectivity_{name}"


def filter_connectivities(offset_provider: Mapping[str, Any]) -> dict[str, Connectivity]:
    """
    Filter offset providers of type `Connectivity`.

    In other words, filter out the cartesian offset providers.
    Returns a new dictionary containing only `Connectivity` values.
    """
    return {
        offset: table
        for offset, table in offset_provider.items()
        if isinstance(table, Connectivity)
    }


def get_field_domain(
    node: itir.Expr,
) -> list[tuple[Dimension, dace.symbolic.SymbolicType, dace.symbolic.SymbolicType]]:
    """
    Specialized visit method for domain expressions.

    Returns for each domain dimension the corresponding range.
    """
    assert cpm.is_call_to(node, ("cartesian_domain", "unstructured_domain"))

    domain = []
    for named_range in node.args:
        assert cpm.is_call_to(named_range, "named_range")
        assert len(named_range.args) == 3
        axis = named_range.args[0]
        assert isinstance(axis, itir.AxisLiteral)
        bounds = []
        for arg in named_range.args[1:3]:
            sym_str = get_symbolic_expr(arg)
            sym_val = dace.symbolic.SymExpr(sym_str)
            bounds.append(sym_val)
        size_value = str(bounds[1] - bounds[0])
        if size_value.isdigit():
            dim = Dimension(axis.value, DimensionKind.LOCAL)
        else:
            dim = Dimension(axis.value, DimensionKind.HORIZONTAL)
        domain.append((dim, bounds[0], bounds[1]))

    return domain


def get_domain(
    node: itir.Expr,
) -> dict[str, tuple[dace.symbolic.SymbolicType, dace.symbolic.SymbolicType]]:
    """
    Returns domain represented in dictionary form.
    """
    field_domain = get_field_domain(node)

    return {dim.value: (lb, ub) for dim, lb, ub in field_domain}


def get_symbolic_expr(node: itir.Expr) -> str:
    """
    Specialized visit method for symbolic expressions.

    Returns a string containong the corresponding Python code, which as tasklet body
    or symbolic array shape.
    """
    return gtir_to_tasklet.PythonCodegen().visit(node)
