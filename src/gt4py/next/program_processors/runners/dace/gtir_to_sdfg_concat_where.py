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
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    f: gtir_to_sdfg_types.FieldopData,
    f_desc: dace.data.Array,
    concat_dim: gtx_common.Dimension,
    concat_dim_index: int,
    concat_dim_origin: dace.symbolic.SymbolicType,
) -> tuple[gtir_to_sdfg_types.FieldopData, dace.data.Array]:
    """
    Helper function called by `_translate_concat_where_impl` to create a slice along
    the concat dimension, that is a new array with an extra dimension and a single
    level. This allows to treat 'f' as a slice and concatanate it to the other field.
    """
    assert isinstance(f.gt_type, ts.FieldType)
    dims = [*f.gt_type.dims[:concat_dim_index], concat_dim, *f.gt_type.dims[concat_dim_index:]]
    origin = tuple([*f.origin[:concat_dim_index], concat_dim_origin, *f.origin[concat_dim_index:]])
    shape = tuple([*f_desc.shape[:concat_dim_index], 1, *f_desc.shape[concat_dim_index:]])
    slice_data, slice_data_desc = sdfg.add_temp_transient(shape, f_desc.dtype)
    slice_node = state.add_access(slice_data)
    state.add_nedge(
        f.dc_node,
        slice_node,
        dace.Memlet(
            data=f.dc_node.data,
            subset=dace_subsets.Range.from_array(f_desc),
            other_subset=dace_subsets.Range.from_array(slice_data_desc),
        ),
    )
    fslice = gtir_to_sdfg_types.FieldopData(
        slice_node, ts.FieldType(dims=dims, dtype=f.gt_type.dtype), origin
    )
    return fslice, slice_data_desc


def _make_concat_scalar_broadcast(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    inp: gtir_to_sdfg_types.FieldopData,
    inp_desc: dace.data.Array,
    domain: gtir_domain.FieldopDomain,
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
    out_dims, out_origin, out_shape = gtir_domain.get_field_layout(domain)
    out_type = ts.FieldType(dims=out_dims, dtype=inp.gt_type.dtype)

    out_name, out_desc = sdfg.add_temp_transient(out_shape, inp_desc.dtype)
    out_node = state.add_access(out_name)

    map_variables = [gtir_to_sdfg_utils.get_map_variable(dim) for dim in out_dims]
    inp_index = (
        "0"
        if isinstance(inp.dc_node.desc(sdfg), dace.data.Scalar)
        else (
            f"({map_variables[concat_dim_index]} + {out_origin[concat_dim_index] - inp.origin[0]})"
        )
    )
    state.add_mapped_tasklet(
        "broadcast",
        map_ranges={
            index: r
            for index, r in zip(map_variables, dace_subsets.Range.from_array(out_desc), strict=True)
        },
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
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    mask_domain: gtir_domain.FieldopDomain,
    node_domain: gtir.Expr,
    tb_node_domain: gtir.Expr,
    fb_node_domain: gtir.Expr,
    tb_field: gtir_to_sdfg_types.FieldopData,
    fb_field: gtir_to_sdfg_types.FieldopData,
) -> gtir_to_sdfg_types.FieldopData:
    """
    Helper function to lower 'concat_where' on a single output field.

    It builds the output field by concatanting the two input fields on the lower
    and upper domain, respectively. These two domain are computed from the intersection
    of the mask and the input domains.

    Args:
        sdfg: The SDFG where the primitive subgraph should be instantiated
        state: The SDFG state where the result of the primitive function should be made available
        sdfg_builder: The object responsible for visiting child nodes of the primitive node.
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
    tb_data_desc, fb_data_desc = (inp.dc_node.desc(sdfg) for inp in [tb_field, fb_field])
    assert tb_data_desc.dtype == fb_data_desc.dtype

    tb_domain, fb_domain = (
        gtir_domain.extract_domain(domain) for domain in [tb_node_domain, fb_node_domain]
    )
    concat_dim, mask_lower_bound, mask_upper_bound = mask_domain[0]

    # expect unbound range in the concat domain expression on lower or upper range
    if mask_lower_bound == gtir_to_sdfg_utils.get_symbolic(gtir.InfinityLiteral.NEGATIVE):
        concat_dim_bound = mask_upper_bound
        lower, lower_desc, lower_domain = (tb_field, tb_data_desc, tb_domain)
        upper, upper_desc, upper_domain = (fb_field, fb_data_desc, fb_domain)
    elif mask_upper_bound == gtir_to_sdfg_utils.get_symbolic(gtir.InfinityLiteral.POSITIVE):
        concat_dim_bound = mask_lower_bound
        lower, lower_desc, lower_domain = (fb_field, fb_data_desc, fb_domain)
        upper, upper_desc, upper_domain = (tb_field, tb_data_desc, tb_domain)
    else:
        raise ValueError(f"Unexpected concat mask {mask_domain[0]}.")

    # we use the concat domain, stored in the annex, as the domain of output field
    output_domain = gtir_domain.extract_domain(node_domain)
    # The strict order of lower and upper bounds of output domain is not guaranteed,
    #   for concat_where expressions, so we apply a runtime check.
    output_dims, output_origin, output_shape = gtir_domain.get_field_layout(
        output_domain, check_strict_order=True
    )
    concat_dim_index = output_dims.index(concat_dim)

    # in case one of the arguments is a scalar value, we convert it to a single-element
    # 1D field with the dimension of the concat expression
    if isinstance(lower.gt_type, ts.ScalarType):
        assert len(lower_domain) == 0
        assert isinstance(upper.gt_type, ts.FieldType)
        lower = gtir_to_sdfg_types.FieldopData(
            lower.dc_node,
            ts.FieldType(dims=[concat_dim], dtype=lower.gt_type),
            origin=(concat_dim_bound - 1,),
        )
        lower_bound = output_domain[concat_dim_index][1]
        lower_domain = [(concat_dim, lower_bound, concat_dim_bound)]
    elif isinstance(upper.gt_type, ts.ScalarType):
        assert len(upper_domain) == 0
        assert isinstance(lower.gt_type, ts.FieldType)
        upper = gtir_to_sdfg_types.FieldopData(
            upper.dc_node,
            ts.FieldType(dims=[concat_dim], dtype=upper.gt_type),
            origin=(concat_dim_bound,),
        )
        upper_bound = output_domain[concat_dim_index][2]
        upper_domain = [(concat_dim, concat_dim_bound, upper_bound)]

    if concat_dim not in lower.gt_type.dims:  # type: ignore[union-attr]
        assert (
            lower.gt_type.dims  # type: ignore[union-attr]
            == [
                *upper.gt_type.dims[0:concat_dim_index],  # type: ignore[union-attr]
                *upper.gt_type.dims[concat_dim_index + 1 :],  # type: ignore[union-attr]
            ]
        )
        lower, lower_desc = _make_concat_field_slice(
            sdfg, state, lower, lower_desc, concat_dim, concat_dim_index, concat_dim_bound - 1
        )
        lower_bound = dace.symbolic.pystr_to_symbolic(
            f"max({concat_dim_bound - 1}, {output_domain[concat_dim_index][1]})"
        )
        lower_domain.insert(concat_dim_index, (concat_dim, lower_bound, concat_dim_bound))
    elif concat_dim not in upper.gt_type.dims:  # type: ignore[union-attr]
        assert (
            upper.gt_type.dims  # type: ignore[union-attr]
            == [
                *lower.gt_type.dims[0:concat_dim_index],  # type: ignore[union-attr]
                *lower.gt_type.dims[concat_dim_index + 1 :],  # type: ignore[union-attr]
            ]
        )
        upper, upper_desc = _make_concat_field_slice(
            sdfg, state, upper, upper_desc, concat_dim, concat_dim_index, concat_dim_bound
        )
        upper_bound = dace.symbolic.pystr_to_symbolic(
            f"min({concat_dim_bound + 1}, {output_domain[concat_dim_index][2]})"
        )
        upper_domain.insert(concat_dim_index, (concat_dim, concat_dim_bound, upper_bound))
    elif len(lower.gt_type.dims) == 1 and len(output_domain) > 1:  # type: ignore[union-attr]
        assert len(lower_domain) == 1 and lower_domain[0][0] == concat_dim
        lower_domain = [
            *output_domain[:concat_dim_index],
            lower_domain[0],
            *output_domain[concat_dim_index + 1 :],
        ]
        lower, lower_desc = _make_concat_scalar_broadcast(
            sdfg, state, lower, lower_desc, lower_domain, concat_dim_index
        )
    elif len(upper.gt_type.dims) == 1 and len(output_domain) > 1:  # type: ignore[union-attr]
        assert len(upper_domain) == 1 and upper_domain[0][0] == concat_dim
        upper_domain = [
            *output_domain[:concat_dim_index],
            upper_domain[0],
            *output_domain[concat_dim_index + 1 :],
        ]
        upper, upper_desc = _make_concat_scalar_broadcast(
            sdfg, state, upper, upper_desc, upper_domain, concat_dim_index
        )
    elif lower.gt_type.dims != upper.gt_type.dims:  # type: ignore[union-attr]
        raise NotImplementedError("concat_where on fields with different domain is not supported.")

    # ensure that the arguments have the same domain as the concat result
    assert all(ftype.dims == output_dims for ftype in (lower.gt_type, upper.gt_type))  # type: ignore[union-attr]

    lower_range_0 = output_domain[concat_dim_index][1]
    lower_range_1 = dace.symbolic.pystr_to_symbolic(
        f"max({lower_range_0}, {lower_domain[concat_dim_index][2]})"
    )
    lower_range_size = lower_range_1 - lower_range_0

    upper_range_1 = output_domain[concat_dim_index][2]
    upper_range_0 = dace.symbolic.pystr_to_symbolic(
        f"min({upper_range_1}, {upper_domain[concat_dim_index][1]})"
    )
    upper_range_size = upper_range_1 - upper_range_0

    output, output_desc = sdfg_builder.add_temp_array(sdfg, output_shape, lower_desc.dtype)
    output_node = state.add_access(output)

    lower_subset = dace_subsets.Range(
        [
            (
                lower_range_0 - lower.origin[dim_index],
                lower_range_1 - lower.origin[dim_index] - 1,
                1,
            )
            if dim_index == concat_dim_index
            else (
                output_domain[dim_index][1] - lower.origin[dim_index],
                output_domain[dim_index][1] - lower.origin[dim_index] + size - 1,
                1,
            )
            for dim_index, size in enumerate(output_desc.shape)
        ]
    )
    # we write the data of the lower range into the output array starting from the index zero
    lower_output_subset = dace_subsets.Range(
        [
            (0, lower_range_size - 1, 1) if dim_index == concat_dim_index else (0, size - 1, 1)
            for dim_index, size in enumerate(output_desc.shape)
        ]
    )
    state.add_nedge(
        lower.dc_node,
        output_node,
        dace.Memlet(
            data=lower.dc_node.data,
            subset=lower_subset,
            other_subset=lower_output_subset,
            dynamic=True,  # this memlet could be empty, but this is known only at runtime
        ),
    )

    upper_subset = dace_subsets.Range(
        [
            (
                upper_range_0 - upper.origin[dim_index],
                upper_range_1 - upper.origin[dim_index] - 1,
                1,
            )
            if dim_index == concat_dim_index
            else (
                output_domain[dim_index][1] - upper.origin[dim_index],
                output_domain[dim_index][1] - upper.origin[dim_index] + size - 1,
                1,
            )
            for dim_index, size in enumerate(output_desc.shape)
        ]
    )
    # the upper range should be written next to the lower range, so the destination
    # subset does not start from index zero
    upper_output_subset = dace_subsets.Range(
        [
            (
                lower_range_size,
                lower_range_size + upper_range_size - 1,
                1,
            )
            if dim_index == concat_dim_index
            else (0, size - 1, 1)
            for dim_index, size in enumerate(output_desc.shape)
        ]
    )
    state.add_nedge(
        upper.dc_node,
        output_node,
        dace.Memlet(
            data=upper.dc_node.data,
            subset=upper_subset,
            other_subset=upper_output_subset,
            dynamic=True,  # this memlet could be empty, but this is known only at runtime
        ),
    )

    return gtir_to_sdfg_types.FieldopData(output_node, lower.gt_type, origin=tuple(output_origin))


def translate_concat_where(
    node: gtir.Node,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
) -> gtir_to_sdfg_types.FieldopResult:
    """
    Lowers a `concat_where` expression to a dataflow where two memlets write
    disjoint subsets, for the lower and upper domain, on one data access node.

    Implements the `PrimitiveTranslator` protocol.

    In case of tuples, this function calls `_translate_concat_where_impl()` on all
    fields by means of `tree_map`.
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
    tb, fb = (sdfg_builder.visit(node.args[i], sdfg=sdfg, head_state=state) for i in [1, 2])

    return (
        _translate_concat_where_impl(
            sdfg,
            state,
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
            _sdfg=sdfg,
            _state=state,
            _mask_domain=mask_domain: _translate_concat_where_impl(
                _sdfg,
                _state,
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
