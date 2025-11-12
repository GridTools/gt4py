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

import dace
from dace import subsets as dace_subsets

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
    sdfg_library_nodes,
)
from gt4py.next.type_system import type_specifications as ts


def _make_concat_scalar_broadcast(
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    inp: gtir_to_sdfg_types.FieldopData,
    inp_desc: dace.data.Array,
    out_domain: domain_utils.SymbolicDomain,
) -> tuple[gtir_to_sdfg_types.FieldopData, dace.data.Array]:
    """
    Helper function called by `_translate_concat_where_impl` to create a mapped
    tasklet that broadcasts one scalar value on the given domain.

    The scalar value can come from either a scalar node or from a 1D-array (assuming
    the array represents a field in the concat dimension).
    """
    out_dims, out_origin, out_shape = gtir_domain.get_field_layout(
        gtir_domain.get_field_domain(out_domain)
    )

    out_name, out_desc = ctx.sdfg.add_temp_transient(out_shape, inp_desc.dtype)
    out_node = ctx.state.add_access(out_name)

    inp_desc = inp.dc_node.desc(ctx.sdfg)

    if isinstance(inp.gt_type, ts.FieldType):
        assert isinstance(inp.gt_type.dtype, ts.ScalarType)
        inp_axes = [out_dims.index(dim) for dim in inp.gt_type.dims]
        inp_origin = inp.origin
        dtype = inp.gt_type.dtype
    else:
        inp_axes = None
        inp_origin = None
        dtype = inp.gt_type

    # Use a 'Broadcast' library node to write the scalar value to the result field.
    name = sdfg_builder.unique_tasklet_name("broadcast")
    bcast_node = sdfg_library_nodes.Broadcast(name, inp_axes, inp_origin, out_origin)
    ctx.state.add_node(bcast_node)
    ctx.state.add_edge(
        inp.dc_node,
        None,
        bcast_node,
        "_inp",
        dace.Memlet(data=inp.dc_node.data, subset=dace_subsets.Range.from_array(inp_desc)),
    )
    ctx.state.add_edge(
        bcast_node,
        "_outp",
        out_node,
        None,
        dace.Memlet(data=out_name, subset=dace_subsets.Range.from_array(out_desc)),
    )

    out_type = ts.FieldType(dims=out_dims, dtype=dtype)
    out_field = gtir_to_sdfg_types.FieldopData(out_node, out_type, tuple(out_origin))
    return out_field, out_desc


def _translate_concat_where_impl(
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    node_domain: domain_utils.SymbolicDomain,
    mask_domain: domain_utils.SymbolicDomain,
    tb_field: gtir_to_sdfg_types.FieldopData,
    tb_domain: domain_utils.SymbolicDomain,
    fb_field: gtir_to_sdfg_types.FieldopData,
    fb_domain: domain_utils.SymbolicDomain,
) -> gtir_to_sdfg_types.FieldopData:
    """
    Helper function called by `translate_concat_where()` to lower 'concat_where'
    on a single output field.

    In case of tuples, this function is called on all fields by means of `tree_map`.

    It builds the output field by concatanating the two input fields on the lower
    and upper domain. These two domains are computed from the intersection of the
    mask and the input domains.

    Note that 'tb' and 'fb' stand for true/false branch and refer to the two branches
    of the concat_where expression, passed as second and third argument, respectively.

    Args:
        ctx: The SDFG context where the primitive subgraph should be instantiated.
        sdfg_builder: The object responsible for SubgraphContexting child nodes of the primitive node.
        mask_domain: Domain (only for concat dimension) of the true branch, infinite
            on lower or upper boundary.
        node_domain: Domain (all dimensions) of output field.
        tb_field: Input field on the true branch.
        tb_domain: Domain of the field expression on the true branch.
        fb_field: Input field on the false branch.
        fb_domain: Domain of the field expression on the false branch.

    Returns:
        The field resulted from concatanating the input fields on the lower and upper domain.
    """
    tb_data_desc, fb_data_desc = (inp.dc_node.desc(ctx.sdfg) for inp in [tb_field, fb_field])
    assert tb_data_desc.dtype == fb_data_desc.dtype

    assert len(mask_domain.ranges) == 1
    concat_dim = next(iter(mask_domain.ranges.keys()))

    # Expect unbound range in the concat domain expression on lower or upper range:
    #  - if the domain expression is unbound on lower side (negative infinite),
    #    the expression on the true branch is to be considered the input for the
    #    lower domain.
    #  - viceversa, if the domain expression is unbound on upper side (positive
    #    infinite), the true expression represents the input for the upper domain.
    infinity_literals = (gtir.InfinityLiteral.POSITIVE, gtir.InfinityLiteral.NEGATIVE)
    if mask_domain.ranges[concat_dim].start in infinity_literals:
        concat_dim_bound_expr = mask_domain.ranges[concat_dim].stop
        lower, lower_desc, lower_domain = (tb_field, tb_data_desc, tb_domain)
        upper, upper_desc, upper_domain = (fb_field, fb_data_desc, fb_domain)
    elif mask_domain.ranges[concat_dim].stop in infinity_literals:
        concat_dim_bound_expr = mask_domain.ranges[concat_dim].start
        lower, lower_desc, lower_domain = (fb_field, fb_data_desc, fb_domain)
        upper, upper_desc, upper_domain = (tb_field, tb_data_desc, tb_domain)
    else:
        raise ValueError(f"Unexpected concat mask {mask_domain} with finite domain.")

    # We use the concat domain, stored in the annex, as the domain of output field.
    output_domain = gtir_domain.get_field_domain(node_domain)
    output_dims, output_origin, output_shape = gtir_domain.get_field_layout(output_domain)
    concat_dim_index = output_dims.index(concat_dim)

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
    if isinstance(lower.gt_type, ts.ScalarType) or len(lower.gt_type.dims) < len(output_dims):
        if concat_dim not in lower_domain.ranges:
            lower_domain.ranges[concat_dim] = domain_utils.SymbolicRange(
                start=node_domain.ranges[concat_dim].start, stop=concat_dim_bound_expr
            )
        lower_domain = domain_utils.promote_domain(lower_domain, node_domain.ranges.keys())
        lower_domain = domain_utils.domain_intersection(lower_domain, node_domain)
        lower, lower_desc = _make_concat_scalar_broadcast(
            ctx=ctx,
            sdfg_builder=sdfg_builder,
            inp=lower,
            inp_desc=lower_desc,
            out_domain=lower_domain,
        )

    elif isinstance(upper.gt_type, ts.ScalarType) or len(upper.gt_type.dims) < len(output_dims):
        if concat_dim not in upper_domain.ranges:
            upper_domain.ranges[concat_dim] = domain_utils.SymbolicRange(
                start=concat_dim_bound_expr,
                stop=node_domain.ranges[concat_dim].stop,
            )
        upper_domain = domain_utils.promote_domain(upper_domain, node_domain.ranges.keys())
        upper_domain = domain_utils.domain_intersection(upper_domain, node_domain)
        upper, upper_desc = _make_concat_scalar_broadcast(
            ctx=ctx,
            sdfg_builder=sdfg_builder,
            inp=upper,
            inp_desc=upper_desc,
            out_domain=upper_domain,
        )

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
        assert isinstance(lower.gt_type, ts.FieldType)
        assert isinstance(upper.gt_type, ts.FieldType)
        if lower.gt_type.dims != upper.gt_type.dims:
            raise ValueError(
                "Lowering concat_where on fields with different domain is not supported."
            )

    lower_domain_range = lower_domain.ranges[concat_dim]
    lower_range_0 = gtir_to_sdfg_utils.get_symbolic(lower_domain_range.start)
    lower_range_1 = gtir_to_sdfg_utils.get_symbolic(
        im.maximum(lower_domain_range.start, lower_domain_range.stop)
    )
    lower_range_size = lower_range_1 - lower_range_0

    upper_domain_range = upper_domain.ranges[concat_dim]
    upper_range_0 = gtir_to_sdfg_utils.get_symbolic(upper_domain_range.start)
    upper_range_1 = gtir_to_sdfg_utils.get_symbolic(
        im.maximum(upper_domain_range.start, upper_domain_range.stop)
    )
    upper_range_size = upper_range_1 - upper_range_0

    output, output_desc = sdfg_builder.add_temp_array(ctx.sdfg, output_shape, lower_desc.dtype)
    output_node = ctx.state.add_access(output)

    lower_subset = []
    lower_output_subset = []
    upper_subset = []
    upper_output_subset = []
    for dim_index, size in enumerate(output_desc.shape):
        if dim_index == concat_dim_index:
            lower_subset.append(
                (
                    lower_range_0 - lower.origin[dim_index],
                    lower_range_1 - lower.origin[dim_index] - 1,
                    1,
                )
            )
            upper_subset.append(
                (
                    upper_range_0 - upper.origin[dim_index],
                    upper_range_1 - upper.origin[dim_index] - 1,
                    1,
                )
            )
            # we write the data of the lower range into the output array starting
            # from the index zero
            lower_output_subset.append((0, lower_range_size - 1, 1))
            # the upper range should be written next to the lower range, so the
            # destination subset does not start from index zero
            upper_output_subset.append(
                (
                    lower_range_size,
                    lower_range_size + upper_range_size - 1,
                    1,
                )
            )
        else:
            lower_subset.append(
                (
                    output_domain[dim_index].start - lower.origin[dim_index],
                    output_domain[dim_index].start - lower.origin[dim_index] + size - 1,
                    1,
                )
            )
            upper_subset.append(
                (
                    output_domain[dim_index].start - upper.origin[dim_index],
                    output_domain[dim_index].start - upper.origin[dim_index] + size - 1,
                    1,
                )
            )

            lower_output_subset.append((0, size - 1, 1))
            upper_output_subset.append((0, size - 1, 1))

    ctx.state.add_nedge(
        lower.dc_node,
        output_node,
        dace.Memlet(
            data=lower.dc_node.data,
            subset=dace_subsets.Range(lower_subset),
            other_subset=dace_subsets.Range(lower_output_subset),
        ),
    )
    ctx.state.add_nedge(
        upper.dc_node,
        output_node,
        dace.Memlet(
            data=upper.dc_node.data,
            subset=dace_subsets.Range(upper_subset),
            other_subset=dace_subsets.Range(upper_output_subset),
        ),
    )

    return gtir_to_sdfg_types.FieldopData(output_node, lower.gt_type, origin=tuple(output_origin))


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

    # we visit the field arguments for the true and false branch
    (tb, tb_domain), (fb, fb_domain) = (
        (sdfg_builder.visit(node.args[i], ctx=ctx), node.args[i].annex.domain) for i in [1, 2]
    )

    return _translate_concat_where_impl(
        ctx,
        sdfg_builder,
        node.annex.domain,
        mask_domain,
        tb,
        tb_domain,
        fb,
        fb_domain,
    )
