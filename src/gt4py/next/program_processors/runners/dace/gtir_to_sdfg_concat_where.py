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
import sympy
from dace import subsets as dace_subsets

from gt4py.next import common as gtx_common, utils as gtx_utils
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace import (
    gtir_domain,
    gtir_to_sdfg,
    gtir_to_sdfg_types,
    gtir_to_sdfg_utils,
)
from gt4py.next.type_system import type_specifications as ts


def _make_concat_field_slice(
    ctx: gtir_to_sdfg.SubgraphContext,
    field: gtir_to_sdfg_types.FieldopData,
    field_desc: dace.data.Array,
    concat_dim: gtx_common.Dimension,
    concat_dim_index: int,
    concat_dim_origin: dace.symbolic.SymbolicType,
) -> tuple[gtir_to_sdfg_types.FieldopData, dace.data.Array]:
    """
    Helper function called by `_translate_concat_where_impl` to create a slice along
    the concat dimension, that is a new array with an extra dimension and a single
    level. This allows to concatanate the input fields along the concat dimension.
    """
    assert isinstance(field.gt_type, ts.FieldType)
    assert concat_dim not in field.gt_type.dims
    dims = [
        *field.gt_type.dims[:concat_dim_index],
        concat_dim,
        *field.gt_type.dims[concat_dim_index:],
    ]
    origin = tuple(
        [*field.origin[:concat_dim_index], concat_dim_origin, *field.origin[concat_dim_index:]]
    )
    shape = tuple([*field_desc.shape[:concat_dim_index], 1, *field_desc.shape[concat_dim_index:]])
    extended_field_data, extended_field_desc = ctx.sdfg.add_temp_transient(shape, field_desc.dtype)
    extended_field_node = ctx.state.add_access(extended_field_data)
    ctx.state.add_nedge(
        field.dc_node,
        extended_field_node,
        dace.Memlet(
            data=field.dc_node.data,
            subset=dace_subsets.Range.from_array(field_desc),
            other_subset=dace_subsets.Range.from_array(extended_field_desc),
        ),
    )
    extended_field = gtir_to_sdfg_types.FieldopData(
        extended_field_node, ts.FieldType(dims=dims, dtype=field.gt_type.dtype), origin
    )
    return extended_field, extended_field_desc


def _make_concat_scalar_broadcast(
    ctx: gtir_to_sdfg.SubgraphContext,
    inp: gtir_to_sdfg_types.FieldopData,
    inp_desc: dace.data.Array,
    out_domain: gtir_domain.FieldopDomain,
    concat_dim_index: int,
) -> tuple[gtir_to_sdfg_types.FieldopData, dace.data.Array]:
    """
    Helper function called by `_translate_concat_where_impl` to create a mapped
    tasklet that broadcasts one scalar value on the given domain.

    The scalar value can come from either a scalar node or from a 1D-array (assuming
    the array represents a field in the concat dimension).
    """
    assert isinstance(inp.gt_type, ts.FieldType)
    assert len(inp.gt_type.dims) == 1
    out_dims, out_origin, out_shape = gtir_domain.get_field_layout(out_domain, ctx.target_domain)
    out_type = ts.FieldType(dims=out_dims, dtype=inp.gt_type.dtype)

    out_name, out_desc = ctx.sdfg.add_temp_transient(out_shape, inp_desc.dtype)
    out_node = ctx.state.add_access(out_name)

    map_variables = [gtir_to_sdfg_utils.get_map_variable(dim) for dim in out_dims]
    inp_index = (
        "0"
        if isinstance(inp.dc_node.desc(ctx.sdfg), dace.data.Scalar)
        else (
            f"({map_variables[concat_dim_index]} + {out_origin[concat_dim_index] - inp.origin[0]})"
        )
    )
    ctx.state.add_mapped_tasklet(
        "broadcast",
        map_ranges=dict(zip(map_variables, dace_subsets.Range.from_array(out_desc), strict=True)),
        code="__out = __inp",
        inputs={"__inp": dace.Memlet(data=inp.dc_node.data, subset=inp_index)},
        outputs={"__out": dace.Memlet(data=out_name, subset=",".join(map_variables))},
        input_nodes={inp.dc_node},
        output_nodes={out_node},
        external_edges=True,
    )

    out_field = gtir_to_sdfg_types.FieldopData(out_node, out_type, tuple(out_origin))
    return out_field, out_desc


def _translate_concat_where_impl(
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    mask_domain: gtir_domain.FieldopDomain,
    node_domain: gtir.Expr,
    tb_node_domain: gtir.Expr,
    fb_node_domain: gtir.Expr,
    tb_field: gtir_to_sdfg_types.FieldopData,
    fb_field: gtir_to_sdfg_types.FieldopData,
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
        tb_node_domain: Domain of the field passed on the true branch.
        fb_node_domain: Domain of the field passed on the false branch.
        tb_field: Input field on the true branch.
        fb_field: Input field on the false branch.

    Returns:
        The field resulted from concatanating the input fields on the lower and upper domain.
    """
    tb_data_desc, fb_data_desc = (inp.dc_node.desc(ctx.sdfg) for inp in [tb_field, fb_field])
    assert tb_data_desc.dtype == fb_data_desc.dtype

    tb_domain, fb_domain = (
        gtir_domain.extract_domain(domain) for domain in [tb_node_domain, fb_node_domain]
    )
    assert len(mask_domain) == 1
    concat_domain = mask_domain[0]

    # Expect unbound range in the concat domain expression on lower or upper range:
    #  - if the domain expression is unbound on lower side (negative infinite),
    #    the expression on the true branch is to be considered the input for the
    #    lower domain.
    #  - viceversa, if the domain expression is unbound on upper side (positive
    #    infinite), the true expression represents the input for the upper domain.
    if concat_domain.start == gtir_to_sdfg_utils.get_symbolic(gtir.InfinityLiteral.NEGATIVE):
        concat_dim_bound = concat_domain.stop
        lower, lower_desc, lower_domain = (tb_field, tb_data_desc, tb_domain)
        upper, upper_desc, upper_domain = (fb_field, fb_data_desc, fb_domain)
    elif concat_domain.stop == gtir_to_sdfg_utils.get_symbolic(gtir.InfinityLiteral.POSITIVE):
        concat_dim_bound = concat_domain.start
        lower, lower_desc, lower_domain = (fb_field, fb_data_desc, fb_domain)
        upper, upper_desc, upper_domain = (tb_field, tb_data_desc, tb_domain)
    else:
        raise ValueError(f"Unexpected concat mask {mask_domain[0]}.")

    # we use the concat domain, stored in the annex, as the domain of output field
    output_domain = gtir_domain.extract_domain(node_domain)
    output_dims, output_origin, output_shape = gtir_domain.get_field_layout(
        output_domain, ctx.target_domain
    )
    concat_dim_index = output_dims.index(concat_domain.dim)

    """
    In case one of the arguments is a scalar value, for example:
    ```python
    @gtx.field_operator
    def testee(a: np.int32, b: cases.IJKField) -> cases.IJKField:
        return concat_where(KDim < 1, a, b)
    ```
    we convert it to a single-element 1D field with the dimension of the concat expression.
    """
    if isinstance(lower.gt_type, ts.ScalarType):
        assert len(lower_domain) == 0
        assert isinstance(upper.gt_type, ts.FieldType)
        lower = gtir_to_sdfg_types.FieldopData(
            lower.dc_node,
            ts.FieldType(dims=[concat_domain.dim], dtype=lower.gt_type),
            origin=(concat_dim_bound - 1,),
        )
        lower_domain = [
            gtir_domain.FieldopDomainRange(
                dim=concat_domain.dim,
                start=output_domain[concat_dim_index].start,
                stop=concat_dim_bound,
            )
        ]
    elif isinstance(upper.gt_type, ts.ScalarType):
        assert len(upper_domain) == 0
        assert isinstance(lower.gt_type, ts.FieldType)
        upper = gtir_to_sdfg_types.FieldopData(
            upper.dc_node,
            ts.FieldType(dims=[concat_domain.dim], dtype=upper.gt_type),
            origin=(concat_dim_bound,),
        )
        upper_domain = [
            gtir_domain.FieldopDomainRange(
                dim=concat_domain.dim,
                start=concat_dim_bound,
                stop=output_domain[concat_dim_index].stop,
            )
        ]

    if concat_domain.dim not in lower.gt_type.dims:  # type: ignore[union-attr]
        """
        The field on the lower domain is to be treated as a slice to add as one
        level in the concat dimension, on the lower bound.
        Consider for example the following IR, where a horizontal field is added
        as level zero in K-dimension:
        ```python
        @gtx.field_operator
        def testee(interior: cases.IJKField, boundary: cases.IJField) -> cases.IJKField:
            return concat_where(KDim == 0, boundary, interior)
        ```
        """
        assert (
            lower.gt_type.dims  # type: ignore[union-attr]
            == [
                *upper.gt_type.dims[0:concat_dim_index],  # type: ignore[union-attr]
                *upper.gt_type.dims[concat_dim_index + 1 :],  # type: ignore[union-attr]
            ]
        )
        lower, lower_desc = _make_concat_field_slice(
            ctx=ctx,
            field=lower,
            field_desc=lower_desc,
            concat_dim=concat_domain.dim,
            concat_dim_index=concat_dim_index,
            concat_dim_origin=concat_dim_bound - 1,
        )
        lower_domain.insert(
            concat_dim_index,
            gtir_domain.FieldopDomainRange(
                dim=concat_domain.dim,
                start=gtir_domain.simplify_domain_expr(
                    sympy.Max(concat_dim_bound - 1, output_domain[concat_dim_index].start),
                    ctx.target_domain,
                ),
                stop=concat_dim_bound,
            ),
        )
    elif concat_domain.dim not in upper.gt_type.dims:  # type: ignore[union-attr]
        # Same as previous case, but the field slice is added on the upper bound.
        assert (
            upper.gt_type.dims  # type: ignore[union-attr]
            == [
                *lower.gt_type.dims[0:concat_dim_index],  # type: ignore[union-attr]
                *lower.gt_type.dims[concat_dim_index + 1 :],  # type: ignore[union-attr]
            ]
        )
        upper, upper_desc = _make_concat_field_slice(
            ctx=ctx,
            field=upper,
            field_desc=upper_desc,
            concat_dim=concat_domain.dim,
            concat_dim_index=concat_dim_index,
            concat_dim_origin=concat_dim_bound,
        )
        upper_domain.insert(
            concat_dim_index,
            gtir_domain.FieldopDomainRange(
                dim=concat_domain.dim,
                start=concat_dim_bound,
                stop=gtir_domain.simplify_domain_expr(
                    sympy.Min(concat_dim_bound + 1, output_domain[concat_dim_index].stop),
                    ctx.target_domain,
                ),
            ),
        )
    elif isinstance(lower_desc, dace.data.Scalar) or (
        len(lower.gt_type.dims) == 1 and len(output_domain) > 1  # type: ignore[union-attr]
    ):
        """
        The input on the lower domain is either a scalar or a 1d field, representing
        the value(s) to be added as one level in the concat dimension below the upper domain.
        Consider for example the following IR, where the scalar value is one level
        (`KDim == 0`) taken from lower input 'a':
        ```python
        @gtx.field_operator
        def testee(a: cases.KField, b: cases.IJKField) -> cases.IJKField:
            return concat_where(KDim == 0, a, b)
        ```
        """
        assert len(lower_domain) == 1 and lower_domain[0].dim == concat_domain.dim
        lower_domain = [
            *output_domain[:concat_dim_index],
            lower_domain[0],
            *output_domain[concat_dim_index + 1 :],
        ]
        lower, lower_desc = _make_concat_scalar_broadcast(
            ctx=ctx,
            inp=lower,
            inp_desc=lower_desc,
            out_domain=lower_domain,
            concat_dim_index=concat_dim_index,
        )
    elif isinstance(upper_desc, dace.data.Scalar) or (
        len(upper.gt_type.dims) == 1 and len(output_domain) > 1  # type: ignore[union-attr]
    ):
        # Same as previous case, but the scalar value is taken from `upper` input.
        assert len(upper_domain) == 1 and upper_domain[0].dim == concat_domain.dim
        upper_domain = [
            *output_domain[:concat_dim_index],
            upper_domain[0],
            *output_domain[concat_dim_index + 1 :],
        ]
        upper, upper_desc = _make_concat_scalar_broadcast(
            ctx=ctx,
            inp=upper,
            inp_desc=upper_desc,
            out_domain=upper_domain,
            concat_dim_index=concat_dim_index,
        )
    else:
        """
        Handle here the _regular_ case, that is concat_where applied to two fields
        with same domain:
        ```python
        @gtx.field_operator
        def testee(a: cases.IJKField, b: cases.IJKField) -> cases.IJKField:
            return concat_where(KDim <=10 , a, b)
        ```
        """
        assert isinstance(lower.gt_type, ts.FieldType)
        assert isinstance(lower_desc, dace.data.Array)
        assert isinstance(upper.gt_type, ts.FieldType)
        assert isinstance(upper_desc, dace.data.Array)
        if lower.gt_type.dims != upper.gt_type.dims:
            raise NotImplementedError(
                "Lowering concat_where on fields with different domain is not supported."
            )

    # ensure that the arguments have the same domain as the concat result
    assert all(ftype.dims == output_dims for ftype in (lower.gt_type, upper.gt_type))  # type: ignore[union-attr]

    lower_range_0 = output_domain[concat_dim_index].start
    lower_range_1 = gtir_domain.simplify_domain_expr(
        sympy.Max(lower_range_0, lower_domain[concat_dim_index].stop),
        ctx.target_domain,
    )
    lower_range_size = lower_range_1 - lower_range_0

    upper_range_1 = output_domain[concat_dim_index].stop
    upper_range_0 = gtir_domain.simplify_domain_expr(
        sympy.Min(upper_range_1, upper_domain[concat_dim_index].start),
        ctx.target_domain,
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

    # use dynamic memlets because the subset range could be empty, but this is known only at runtime
    ctx.state.add_nedge(
        lower.dc_node,
        output_node,
        dace.Memlet(
            data=lower.dc_node.data,
            subset=dace_subsets.Range(lower_subset),
            other_subset=dace_subsets.Range(lower_output_subset),
            dynamic=True,
        ),
    )
    ctx.state.add_nedge(
        upper.dc_node,
        output_node,
        dace.Memlet(
            data=upper.dc_node.data,
            subset=dace_subsets.Range(upper_subset),
            other_subset=dace_subsets.Range(upper_output_subset),
            dynamic=True,
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

    # First argument is a domain expression that defines the mask of the true branch:
    # we extract the dimension along which we need to concatenate the field arguments,
    # and determine whether the true branch argument should be on the lower or upper
    # range with respect to the boundary value.
    mask_domain = gtir_domain.extract_domain(node.args[0])
    if len(mask_domain) != 1:
        raise NotImplementedError("Expected `concat_where` along single axis.")

    # we visit the field arguments for the true and false branch
    tb, fb = (sdfg_builder.visit(node.args[i], ctx=ctx) for i in [1, 2])

    return (
        _translate_concat_where_impl(
            ctx,
            sdfg_builder,
            mask_domain,
            node.annex.domain,
            node.args[1].annex.domain,
            node.args[2].annex.domain,
            tb,
            fb,
        )
        if isinstance(node.type, ts.FieldType)
        else gtx_utils.tree_map(
            lambda _node_domain,
            _tb_node_domain,
            _fb_node_domain,
            _tb_field,
            _fb_field,
            _sdfg_builder=sdfg_builder,
            _ctx=ctx,
            _mask_domain=mask_domain: _translate_concat_where_impl(
                _ctx,
                _sdfg_builder,
                _mask_domain,
                _node_domain,
                _tb_node_domain,
                _fb_node_domain,
                _tb_field,
                _fb_field,
            )
        )(node.annex.domain, node.args[1].annex.domain, node.args[2].annex.domain, tb, fb)
    )
