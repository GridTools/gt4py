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

from gt4py.next.common import Connectivity, Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace_fieldview.gtir_python_codegen import (
    GTIRPythonCodegen,
)
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


def connectivity_identifier(name: str) -> str:
    return f"__connectivity_{name}"


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


def get_domain(
    node: itir.Expr,
) -> dict[Dimension, tuple[dace.symbolic.SymbolicType, dace.symbolic.SymbolicType]]:
    """
    Specialized visit method for domain expressions.

    Returns a list of dimensions and the corresponding range.
    """
    assert cpm.is_call_to(node, ("cartesian_domain", "unstructured_domain"))

    domain = {}
    for named_range in node.args:
        assert cpm.is_call_to(named_range, "named_range")
        assert len(named_range.args) == 3
        axis = named_range.args[0]
        assert isinstance(axis, itir.AxisLiteral)
        dim = Dimension(axis.value)
        bounds = []
        for arg in named_range.args[1:3]:
            sym_str = get_symbolic_expr(arg)
            sym_val = dace.symbolic.SymExpr(sym_str)
            bounds.append(sym_val)
        domain[dim] = (bounds[0], bounds[1])

    return domain


def get_symbolic_expr(node: itir.Expr) -> str:
    return GTIRPythonCodegen().visit(node)


def unique_name(prefix: str) -> str:
    """Generate a string containing a unique integer id, which is updated incrementally."""

    unique_id = getattr(unique_name, "_unique_id", 0)  # static variable
    setattr(unique_name, "_unique_id", unique_id + 1)  # noqa: B010 [set-attr-with-constant]

    return f"{prefix}_{unique_id}"
