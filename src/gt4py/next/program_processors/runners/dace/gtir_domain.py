# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence, TypeAlias

import dace
import sympy
from dace import subsets as dace_subsets

from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, domain_utils
from gt4py.next.program_processors.runners.dace import gtir_to_sdfg_utils


@dataclasses.dataclass(frozen=True)
class FieldopDomainRange:
    """
    Represents the range of a field operator domain in one dimension.

    It contains 3 elements:
        dim: dimension definition
        start: symbolic expression for lower bound (inclusive)
        stop: symbolic expression for upper bound (exclusive)
    """

    dim: gtx_common.Dimension
    start: dace.symbolic.SymbolicType
    stop: dace.symbolic.SymbolicType


FieldopDomain: TypeAlias = list[FieldopDomainRange]
"""Domain of a field operator represented as a list of `FieldopDomainRange` for each dimension."""


def extract_domain(node: gtir.Expr) -> FieldopDomain:
    """
    Visits the domain of a field operator and returns a list of dimensions and
    the corresponding lower and upper bounds. The returned lower bound is inclusive,
    the upper bound is exclusive: [lower_bound, upper_bound[
    """

    domain = []

    if cpm.is_call_to(node, ("cartesian_domain", "unstructured_domain")):
        for named_range in node.args:
            assert cpm.is_call_to(named_range, "named_range")
            assert len(named_range.args) == 3
            axis = named_range.args[0]
            assert isinstance(axis, gtir.AxisLiteral)
            lower_bound, upper_bound = (
                gtir_to_sdfg_utils.get_symbolic(arg) for arg in named_range.args[1:3]
            )
            dim = gtx_common.Dimension(axis.value, axis.kind)
            domain.append(FieldopDomainRange(dim, lower_bound, upper_bound))

    elif isinstance(node, domain_utils.SymbolicDomain):
        for dim, drange in node.ranges.items():
            domain.append(
                FieldopDomainRange(
                    dim,
                    gtir_to_sdfg_utils.get_symbolic(drange.start),
                    gtir_to_sdfg_utils.get_symbolic(drange.stop),
                )
            )

    else:
        raise ValueError(f"Invalid domain {node}.")

    return domain


def simplify_domain_expr(expr: sympy.Basic, domain: FieldopDomain) -> dace.symbolic.SymbolicType:
    """Simplifies a symbolic expression by applying constraints from domain range.

    Dace uses sympy for symbolic expressions in the SDFG. By applying assumptions
    on a sympy expression, we may obtain a simplified expression.
    This is particularly important in the lowering of concat_where domain expressions,
    because it usually results in cleaner memlet subsets and better map fusion.

    Args:
        expr: The symbolic expression to simplify.
    Returns:
        A new symbolic expression.
    """
    for dim_range in domain:
        # We want to enforce the constraint `ub = lb + size`. The actual constraint
        # is given by the assumption that the `size` variable is integer and non-negative.
        size = sympy.var(f"__gtir_{dim_range.dim.value}_size", integer=True, negative=False)
        expr = expr.subs(dim_range.start, dim_range.stop - size).subs(
            size, dim_range.stop - dim_range.start
        )
    return dace.symbolic.simplify_ext(expr)


def get_domain_indices(
    dims: Sequence[gtx_common.Dimension], origin: Optional[Sequence[dace.symbolic.SymExpr]]
) -> dace_subsets.Indices:
    """
    Helper function to construct the list of indices for a field domain, applying
    an optional origin in each dimension as start index.

    Args:
        dims: The field dimensions.
        origin: The domain start index in each dimension. If set to `None`, assume all zeros.

    Returns:
        A list of indices for field access in dace arrays. As this list is returned
        as `dace.subsets.Indices`, it should be converted to `dace.subsets.Range` before
        being used in memlet subset because ranges are better supported throughout DaCe.
    """
    assert len(dims) != 0
    index_variables = [
        dace.symbolic.pystr_to_symbolic(gtir_to_sdfg_utils.get_map_variable(dim)) for dim in dims
    ]
    origin = [0] * len(index_variables) if origin is None else origin
    return dace_subsets.Indices(
        [index - start_index for index, start_index in zip(index_variables, origin, strict=True)]
    )


def get_field_layout(
    field_domain: FieldopDomain,
    target_domain: FieldopDomain,
) -> tuple[list[gtx_common.Dimension], list[dace.symbolic.SymExpr], list[dace.symbolic.SymExpr]]:
    """
    Parse the field operator domain and generate the shape of the result field.

    Note that this function also ensures that the array shape computed from the
    domain range is non-negative. A negative shape can occur in concat_where
    expressions, where it can happen that 'stop' value is smaller than 'start'.
    Also note that this _strange_ domain with 'start' > 'stop' is usually propagated
    to the input arguments of a concat_where expression (the child nodes), thus
    the lowering of regular field operators also needs to apply the sanity check
    in order to avoid allocation of temporary fields with negative size.

    Args:
        field_domain: The field operator domain.
        target_domain: Domain of the target field in the root `SetAt` expression.

    Returns:
        A tuple of three lists containing:
            - the domain dimensions
            - the domain origin, that is the start indices in all dimensions
            - the domain size in each dimension
    """
    if len(field_domain) == 0:
        return [], [], []
    domain_dims = [domain_range.dim for domain_range in field_domain]
    domain_origin = [domain_range.start for domain_range in field_domain]
    domain_shape = [
        simplify_domain_expr(sympy.Max(0, (domain_range.stop - domain_range.start)), target_domain)
        for domain_range in field_domain
    ]
    return domain_dims, domain_origin, domain_shape
