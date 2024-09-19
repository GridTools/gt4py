# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dace

from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.iterator.type_system import type_specifications as gtir_ts
from gt4py.next.program_processors.runners.dace_fieldview import gtir_python_codegen
from gt4py.next.type_system import type_specifications as ts


def as_dace_type(type_: ts.TypeSpec) -> dace.typeclass:
    """Converts GT4Py scalar type to corresponding DaCe type."""
    if isinstance(type_, ts.ScalarType):
        scalar_type = type_
    elif isinstance(type_, gtir_ts.ListType):
        assert isinstance(type_.element_type, ts.ScalarType)
        scalar_type = type_.element_type
    else:
        raise NotImplementedError

    match scalar_type.kind:
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
            raise ValueError(f"Scalar type '{scalar_type}' not supported.")


def get_domain(
    node: gtir.Expr,
) -> list[tuple[gtx_common.Dimension, dace.symbolic.SymbolicType, dace.symbolic.SymbolicType]]:
    """
    Specialized visit method for domain expressions.

    Returns for each domain dimension the corresponding range.

    TODO: Domain expressions will be recurrent in the GTIR program. An interesting idea
          would be to cache the results of lowering here (e.g. using `functools.lru_cache`)
    """
    assert cpm.is_call_to(node, ("cartesian_domain", "unstructured_domain"))

    domain = []
    for named_range in node.args:
        assert cpm.is_call_to(named_range, "named_range")
        assert len(named_range.args) == 3
        axis = named_range.args[0]
        assert isinstance(axis, gtir.AxisLiteral)
        bounds = [
            dace.symbolic.pystr_to_symbolic(gtir_python_codegen.get_source(arg))
            for arg in named_range.args[1:3]
        ]
        dim = gtx_common.Dimension(axis.value, axis.kind)
        domain.append((dim, bounds[0], bounds[1]))

    return domain


def get_domain_ranges(
    node: gtir.Expr,
) -> dict[gtx_common.Dimension, tuple[dace.symbolic.SymbolicType, dace.symbolic.SymbolicType]]:
    """
    Returns domain represented in dictionary form.
    """
    domain = get_domain(node)

    return {dim: (lb, ub) for dim, lb, ub in domain}


def get_map_variable(dim: gtx_common.Dimension) -> str:
    """
    Format map variable name based on the naming convention for application-specific SDFG transformations.
    """
    suffix = "dim" if dim.kind == gtx_common.DimensionKind.LOCAL else ""
    return f"i_{dim.value}_gtx_{dim.kind}{suffix}"
