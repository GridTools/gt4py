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
import sympy
from dace import subsets as dace_subsets

from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, domain_utils
from gt4py.next.program_processors.runners.dace import gtir_sdfg_utils


DomainRange: TypeAlias = list[tuple[gtx_common.Dimension, dace_subsets.Subset]]
"""
Domain of a field operator represented as a list of tuples with 2 elements:
 - dimension definition
 - dimension range represented as a dace subset
"""


class GTIRDomainParser:
    """Utility class to apply domain constraints on a dace symbolic expression.

    Dace uses sympy for symbolic expression in the SDFG. By applying assumptions
    on the sympy expression, we sometimes obtain a simplified expression.
    This is particularly important in the lowering of concat_where domain expressions,
    because it usually results in cleaner memlet subsets and better map fusion.
    """

    domain_constraints: set[
        tuple[dace.symbolic.SymbolicType, dace.symbolic.SymbolicType, sympy.Basic]
    ]

    def __init__(self, domain: DomainRange):
        # We create a set of variables to represent the domain extent. The actual constraint
        # is given by the assumption that this variable should be integer and non-negative.
        self.domain_constraints = {
            (r[0], r[1] + r[2], sympy.var(f"__gtir_{dim.value}_size", integer=True, negative=False))
            for dim, r in domain
        }

    def simplify(self, expr: dace.symbolic.SymbolicType) -> dace.symbolic.SymbolicType:
        """Simplifies a symbolic domain expression by applying some constraints.

        Args:
            expr: The symbolic expression to simplify.

        Returns:
            A new symbolic expression.
        """
        for lb, ub, size in self.domain_constraints:
            expr = expr.subs(lb, ub - size).subs(size, ub - lb)
        return expr


def extract_domain(node: gtir.Node) -> DomainRange:
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
            lower_bound, upper_bound = (
                gtir_sdfg_utils.get_symbolic(arg) for arg in named_range.args[1:3]
            )
            dim = gtx_common.Dimension(axis.value, axis.kind)
            domain_dims.append(dim)
            domain_range.append((lower_bound, upper_bound - 1, 1))

    elif isinstance(node, domain_utils.SymbolicDomain):
        assert str(node.grid_type) in {"cartesian_domain", "unstructured_domain"}
        for dim, drange in node.ranges.items():
            lower_bound = gtir_sdfg_utils.get_symbolic(drange.start)
            upper_bound = gtir_sdfg_utils.get_symbolic(drange.stop)
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
    domain: DomainRange,
    domain_parser: GTIRDomainParser,
) -> tuple[list[gtx_common.Dimension], list[dace.symbolic.SymExpr], list[dace.symbolic.SymExpr]]:
    """
    Parse the field operator domain and generates the shape of the result field.

    It should be enough to allocate an array with shape (upper_bound - lower_bound)
    but this would require to use array offset for compensate for the start index.
    Suppose that a field operator executes on domain [2,N-2], the dace array to store
    the result only needs size (N-4), but this would require to compensate all array
    accesses with offset -2 (which corresponds to -lower_bound). Instead, we choose
    to allocate (N-2), leaving positions [0:2] unused. The reason is that array offset
    is known to cause issues to SDFG inlining. Besides, map fusion will in any case
    eliminate most of transient arrays.

    Args:
        domain: The field domain.

    Returns:
        A tuple of three lists containing:
            - the domain dimensions
            - the domain origin, that is the start indices in all dimensions
            - the domain size in each dimension
    """
    domain_dims, domain_lbs, domain_sizes = [], [], []
    for dim, dim_range in domain:
        lower_bound = dim_range[0]
        # after introduction of concat_where, the strict order of lower and upper bounds is not guaranteed
        upper_bound = domain_parser.simplify(
            dace.symbolic.pystr_to_symbolic(f"max({dim_range[0]}, {dim_range[1] + dim_range[2]})")
        )
        domain_dims.append(dim)
        domain_lbs.append(lower_bound)
        domain_sizes.append(upper_bound - lower_bound)
    return domain_dims, domain_lbs, domain_sizes


def get_field_subset(domain: DomainRange) -> dace_subsets.Range:
    """
    Construct the memlet subset to access a point in the field domain.

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
