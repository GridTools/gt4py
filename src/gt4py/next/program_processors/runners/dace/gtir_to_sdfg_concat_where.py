# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Implements the lowering of concat_where operator.

This builtin translator implements the `PrimitiveTranslator` protocol as other
translators in `gtir_to_sdfg_primitives` module.
"""

from __future__ import annotations

from typing import Sequence

import dace
from dace import subsets as dace_subsets

from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.program_processors.runners.dace import (
    gtir_domain,
    gtir_to_sdfg,
    gtir_to_sdfg_types,
    gtir_to_sdfg_utils,
    utils as gtx_dace_utils,
)
from gt4py.next.type_system import type_specifications as ts


def _translate_concat_where_branch(
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    concat_dim: gtx_common.Dimension,
    concat_dim_bound_expr: gtir.Expr,
    output_domain: domain_utils.SymbolicDomain,
    output_type: ts.FieldType,
    source_expr: gtir.Expr,
    output_desc: dace.data.Array,
    output_node: dace.nodes.AccessNode,
    output_origin: Sequence[dace.symbolic.SymbolicType],
    is_lower: bool,
) -> None:
    assert isinstance(source_expr.type, (ts.FieldType, ts.ScalarType))

    source_domain = source_expr.annex.domain
    if isinstance(source_expr.type, ts.ScalarType) or len(source_expr.type.dims) < len(
        output_type.dims
    ):
        """
        We broadcast the argument field on dimensions on the output field, in case the
        argument is a scalar value, for example:
        ```python
        @gtx.field_operator
        def testee(a: np.int32, b: IJKField) -> IJKField:
            return concat_where(KDim == 0, a, b)
        ```
        
        Similarly, we broadcast a field defined as a slice of the output domain, i.e.
        a field with a smaller number of dimensions than the out field.
        Consider for example the following IR, where the the 'a' IJ-field is used for
        the vertical boundary (`KDim == 0`):
        ```python
        @gtx.field_operator
        def testee(interior: cases.IJKField, boundary: cases.IJField) -> cases.IJKField:
            return concat_where(KDim == 0, boundary, interior)
        ```
        """
        if concat_dim not in source_domain.ranges:
            source_domain.ranges[concat_dim] = (
                domain_utils.SymbolicRange(
                    start=output_domain.ranges[concat_dim].start, stop=concat_dim_bound_expr
                )
                if is_lower
                else domain_utils.SymbolicRange(
                    start=concat_dim_bound_expr,
                    stop=output_domain.ranges[concat_dim].stop,
                )
            )
        source_domain = domain_utils.promote_domain(source_domain, output_type.dims)
        source_domain = domain_utils.domain_intersection(source_domain, output_domain)

        # Use a 'deref' field operator to broadcast the input expression on the target domain.
        bcast_expr = im.as_fieldop("deref", source_domain.as_expr())(source_expr)
        bcast_expr.type = output_type

        source = sdfg_builder.visit(bcast_expr, ctx=ctx)
    else:
        """
        Handle here the _regular_ case, that is concat_where applied to two fields
        with same domain:
        ```python
        @gtx.field_operator
        def testee(a: cases.IJKField, b: cases.IJKField) -> cases.IJKField:
            return concat_where(KDim < 10, a, b)
        ```
        """
        source = sdfg_builder.visit(source_expr, ctx=ctx)

    assert source.gt_type == output_type
    source_domain_range = source_domain.ranges[concat_dim]
    source_range_0 = gtir_to_sdfg_utils.get_symbolic(source_domain_range.start)
    source_range_1 = gtir_to_sdfg_utils.get_symbolic(
        im.maximum(source_domain_range.start, source_domain_range.stop)
    )
    source_range_size = source_range_1 - source_range_0

    if isinstance(output_type.dtype, ts.ScalarType):
        all_dims = gtx_common.order_dimensions(output_type.dims)
    else:
        assert output_type.dtype.offset_type
        all_dims = gtx_common.order_dimensions([*output_type.dims, output_type.dtype.offset_type])

    source_subset = []
    output_subset = []
    for dim, size, src_origin, dst_origin in zip(
        all_dims,
        output_desc.shape,
        source.origin,
        output_origin,
        strict=True,
    ):
        if dim == concat_dim:
            source_subset.append(
                (
                    source_range_0 - src_origin,
                    source_range_1 - src_origin - 1,
                    1,
                )
            )
            output_subset.append(
                (
                    source_range_0 - dst_origin,
                    source_range_0 - dst_origin + source_range_size - 1,
                    1,
                )
            )
        else:
            source_subset.append(
                (
                    dst_origin - src_origin,
                    dst_origin - src_origin + size - 1,
                    1,
                )
            )
            output_subset.append((0, size - 1, 1))

    ctx.state.add_nedge(
        source.dc_node,
        output_node,
        dace.Memlet(
            data=source.dc_node.data,
            subset=dace_subsets.Range(source_subset),
            other_subset=dace_subsets.Range(output_subset),
        ),
    )


def translate_concat_where(
    node: gtir.Node,
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
) -> gtir_to_sdfg_types.FieldopResult:
    """
    Lowers a `concat_where` expression to a dataflow where two memlets write
    disjoint subsets, for the lower and upper domain, on one data access node.

    Implements the `PrimitiveTranslator` protocol.
    """
    assert cpm.is_call_to(node, "concat_where")
    assert len(node.args) == 3
    assert isinstance(node.type, (ts.FieldType, ts.TupleType))

    if isinstance(node.type, ts.TupleType):
        raise NotImplementedError("Unexpected 'concat_where' with tuple output in SDFG lowering.")

    # First argument is a domain expression that defines the mask of the true branch:
    # we extract the dimension along which we need to concatenate the field arguments,
    # and determine whether the true branch argument should be on the lower or upper
    # range with respect to the boundary value.
    mask_domain = domain_utils.SymbolicDomain.from_expr(node.args[0])
    if len(mask_domain.ranges) != 1:
        raise NotImplementedError("Expected `concat_where` along single axis.")

    concat_dim = next(iter(mask_domain.ranges.keys()))

    # Expect unbound range in the concat domain expression on lower or upper range:
    #  - if the domain expression is unbound on lower side (negative infinite),
    #    the expression on the true branch is to be considered the input for the
    #    lower domain.
    #  - viceversa, if the domain expression is unbound on upper side (positive
    #    infinite), the true expression represents the input for the upper domain.
    infinity_literals = (gtir.InfinityLiteral.POSITIVE, gtir.InfinityLiteral.NEGATIVE)
    if mask_domain.ranges[concat_dim].start in infinity_literals:
        bound_expr = mask_domain.ranges[concat_dim].stop
        lower_expr, upper_expr = node.args[1:]
    elif mask_domain.ranges[concat_dim].stop in infinity_literals:
        bound_expr = mask_domain.ranges[concat_dim].start
        upper_expr, lower_expr = node.args[1:]
    else:
        raise ValueError(f"Unexpected concat mask {mask_domain} with finite domain.")

    # We use the concat domain, stored in the annex, as the domain of output field.
    output_domain = gtir_domain.get_field_domain(node.annex.domain)
    output_dims, output_origin, output_shape = gtir_domain.get_field_layout(output_domain)
    assert output_dims == node.type.dims

    if isinstance(node.type.dtype, ts.ScalarType):
        dtype = gtx_dace_utils.as_dace_type(node.type.dtype)
    else:
        # TODO(edopao): Refactor allocation of fields with local dimension and enable this.
        raise NotImplementedError("concat_where with list output is not supported")

    output, output_desc = sdfg_builder.add_temp_array(ctx.sdfg, output_shape, dtype)
    output_node = ctx.state.add_access(output)

    for i, source_expr in enumerate([lower_expr, upper_expr]):
        _translate_concat_where_branch(
            ctx=ctx,
            sdfg_builder=sdfg_builder,
            concat_dim=concat_dim,
            concat_dim_bound_expr=bound_expr,
            output_domain=node.annex.domain,
            output_type=node.type,
            source_expr=source_expr,
            output_desc=output_desc,
            output_node=output_node,
            output_origin=output_origin,
            is_lower=(i == 0),
        )

    return gtir_to_sdfg_types.FieldopData(output_node, node.type, origin=tuple(output_origin))
