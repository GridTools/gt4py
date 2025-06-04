# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Sequence, TypeAlias

import dace
from dace import subsets as dace_subsets

from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, domain_utils
from gt4py.next.program_processors.runners.dace import gtir_python_codegen, gtir_sdfg_utils


FieldopDomain: TypeAlias = list[tuple[gtx_common.Dimension, dace_subsets.Subset]]
"""
Domain of a field operator represented as a list of tuples with 2 elements:
 - dimension definition
 - dimension range represented as a dace subset: start, stop (included), step
"""


def parse_range_boundary(expr: gtir.Expr) -> dace.symbolic.SymbolicType:
    """Lower a domain range to dace symbolic expression."""
    return dace.symbolic.pystr_to_symbolic(gtir_python_codegen.get_source(expr))


def extract_domain(node: gtir.Expr) -> FieldopDomain:
    """
    Visits the domain of a field operator and returns a list of dimensions and
    the corresponding lower and upper bounds. The returned lower bound is inclusive,
    the upper bound is exclusive: [lower_bound, upper_bound[
    """

    domain_dims = []
    domain_range = []

    if cpm.is_call_to(node, ("cartesian_domain", "unstructured_domain")):
        for named_range in node.args:
            assert cpm.is_call_to(named_range, "named_range")
            assert len(named_range.args) == 3
            axis = named_range.args[0]
            assert isinstance(axis, gtir.AxisLiteral)
            lower_bound, upper_bound = (parse_range_boundary(arg) for arg in named_range.args[1:3])
            dim = gtx_common.Dimension(axis.value, axis.kind)
            domain_dims.append(dim)
            domain_range.append((lower_bound, upper_bound - 1, 1))

    elif isinstance(node, domain_utils.SymbolicDomain):
        assert str(node.grid_type) in {"cartesian_domain", "unstructured_domain"}
        for dim, drange in node.ranges.items():
            lower_bound = parse_range_boundary(drange.start)
            upper_bound = parse_range_boundary(drange.stop)
            domain_dims.append(dim)
            domain_range.append((lower_bound, upper_bound - 1, 1))

    else:
        raise ValueError(f"Invalid domain {node}.")

    return list(zip(domain_dims, dace_subsets.Range(domain_range), strict=True))


def get_domain_indices(
    dims: Sequence[gtx_common.Dimension], origin: Sequence[dace.symbolic.SymExpr] | None
) -> dace_subsets.Indices:
    """
    Construct the list of indices for a field domain, applying an optional origin
    in each dimension as start index.

    Args:
        dims: The field dimensions.
        origin: The domain start index in each dimension. If set to `None`, assume all zeros.

    Returns:
        A list of indices for field access in dace arrays. As this list is returned
        as `dace.subsets.Indices`, it should be converted to `dace.subsets.Range` before
        being used in memlet subset because ranges are better supported throughout DaCe.
        See also `get_field_subset()`.
    """
    assert len(dims) != 0
    index_variables = [
        dace.symbolic.pystr_to_symbolic(gtir_sdfg_utils.get_map_variable(dim)) for dim in dims
    ]
    origin = [0] * len(index_variables) if origin is None else origin
    return dace_subsets.Indices(
        [index - start_index for index, start_index in zip(index_variables, origin, strict=True)]
    )


def get_field_layout(
    domain: FieldopDomain,
) -> tuple[list[gtx_common.Dimension], list[dace.symbolic.SymExpr], list[dace.symbolic.SymExpr]]:
    """
    Parse the field domain and generate the layout of the result dace array.

    Args:
        domain: The field domain.

    Returns:
        A tuple representing the field array layout, which contains three lists with same length:
            - the field dimensions
            - the field origin, that is the start indices in each dimension
            - the field size in each dimension
    """
    domain_dims, domain_lbs, domain_sizes = [], [], []
    for dim, dim_range in domain:
        lower_bound = dim_range[0]
        upper_bound = dim_range[1] + dim_range[2]
        domain_dims.append(dim)
        domain_lbs.append(lower_bound)
        domain_sizes.append(upper_bound - lower_bound)
    return domain_dims, domain_lbs, domain_sizes


def get_field_subset(domain: FieldopDomain) -> dace_subsets.Range:
    """
    Construct the memlet subset to access a point in the field global domain.

    The subset is limited to only the global dimensions. The remaining part of the
    subset (being it a local list or the scalar representing a zero-dimensional
    field) must be handled outside by the caller.

    Args:
        domain: The field domain.

    Returns:
        Range to be used as memlet subset.
    """
    if len(domain) == 0:
        return dace_subsets.Range([])
    dims, origin = zip(*[(dim, dim_range[0]) for dim, dim_range in domain])
    field_indices = get_domain_indices(dims, origin)
    return dace_subsets.Range.from_indices(field_indices)
