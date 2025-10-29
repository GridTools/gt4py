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
from dace import subsets as dace_subsets

from gt4py import eve
from gt4py.eve.extended_typing import MaybeNestedInTuple
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


def get_field_domain(domain: domain_utils.SymbolicDomain) -> FieldopDomain:
    """
    Visits the domain of a field operator and returns a list of dimensions and
    the corresponding lower and upper bounds. The returned lower bound is inclusive,
    the upper bound is exclusive, i.e. `[lower_bound, upper_bound[`

    Note that the doman dimensions are sorted in gt4py canonical order.
    """
    return [
        FieldopDomainRange(
            dim,
            gtir_to_sdfg_utils.get_symbolic(domain.ranges[dim].start),
            gtir_to_sdfg_utils.get_symbolic(domain.ranges[dim].stop),
        )
        for dim in gtx_common.order_dimensions(domain.ranges.keys())
    ]


TargetDomain: TypeAlias = MaybeNestedInTuple[domain_utils.SymbolicDomain]
"""Symbolic domain which defines the range to write in the target field.

For tuple output, the corresponding domain in fieldview is a tuple of domains.
"""


class TargetDomainParser(eve.visitors.NodeTranslator):
    """Visitor class to build a `TargetDomain` symbolic domain."""

    def visit_FunCall(self, node: gtir.FunCall) -> TargetDomain:
        if cpm.is_call_to(node, "make_tuple"):
            return tuple(self.visit(arg) for arg in node.args)
        else:
            return domain_utils.SymbolicDomain.from_expr(node)

    def apply(cls, node: gtir.Expr) -> TargetDomain:
        return cls.visit(node)


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
        dace.symbolic.pystr_to_symbolic(f"max(0, {domain_range.stop - domain_range.start})")
        for domain_range in field_domain
    ]
    return domain_dims, domain_origin, domain_shape
