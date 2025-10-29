# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Contains type definitions and data utilities used during lowering to SDFG."""

from __future__ import annotations

import dataclasses
from typing import Final, TypeAlias

import dace
from dace import subsets as dace_subsets

from gt4py.eve.extended_typing import MaybeNestedInTuple
from gt4py.next import common as gtx_common, utils as gtx_utils
from gt4py.next.ffront import fbuiltins as gtx_fbuiltins
from gt4py.next.iterator import ir as gtir
from gt4py.next.program_processors.runners.dace import (
    gtir_dataflow,
    gtir_domain,
    gtir_to_sdfg,
    gtir_to_sdfg_utils,
    utils as gtx_dace_utils,
)
from gt4py.next.type_system import type_specifications as ts


@dataclasses.dataclass(frozen=True)
class FieldopData:
    """
    Abstraction to represent data (scalars, arrays) during the lowering to SDFG.

    Attribute 'local_offset' must always be set for `FieldType` data with a local
    dimension generated from neighbors access in unstructured domain, and indicates
    the name of the offset provider used to generate the list of neighbor values.

    Args:
        dc_node: DaCe access node to the data storage.
        gt_type: GT4Py type definition, which includes the field domain information.
        origin: Tuple of start indices, in each dimension, for `FieldType` data.
            Pass an empty tuple for `ScalarType` data or zero-dimensional fields.
    """

    dc_node: dace.nodes.AccessNode
    gt_type: ts.FieldType | ts.ScalarType
    origin: tuple[dace.symbolic.SymbolicType, ...]

    def __post_init__(self) -> None:
        """Implements a sanity check on the constructed data type."""
        assert (
            len(self.origin) == 0
            if isinstance(self.gt_type, ts.ScalarType)
            else len(self.origin) == len(self.gt_type.dims)
        )

    def map_to_parent_sdfg(
        self,
        sdfg_builder: gtir_to_sdfg.SDFGBuilder,
        inner_sdfg: dace.SDFG,
        outer_sdfg: dace.SDFG,
        outer_sdfg_state: dace.SDFGState,
        symbol_mapping: dict[str, dace.symbolic.SymbolicType],
    ) -> FieldopData:
        """
        Make the data descriptor which 'self' refers to, and which is located inside
        a NestedSDFG, available in its parent SDFG.

        Thus, it turns 'self' into a non-transient array and creates a new data
        descriptor inside the parent SDFG, with same shape and strides.
        """
        inner_desc = self.dc_node.desc(inner_sdfg)
        assert inner_desc.transient
        inner_desc.transient = False

        if isinstance(self.gt_type, ts.ScalarType):
            outer, outer_desc = sdfg_builder.add_temp_scalar(outer_sdfg, inner_desc.dtype)
            outer_origin = []
        else:
            outer, outer_desc = sdfg_builder.add_temp_array_like(outer_sdfg, inner_desc)
            # We cannot use a copy of the inner data descriptor directly, we have to apply the symbol mapping.
            dace.symbolic.safe_replace(
                symbol_mapping,
                lambda m: dace.sdfg.replace_properties_dict(outer_desc, m),
            )
            # Same applies to the symbols used as field origin (the domain range start)
            outer_origin = [
                gtx_dace_utils.safe_replace_symbolic(val, symbol_mapping) for val in self.origin
            ]

        outer_node = outer_sdfg_state.add_access(outer)
        return FieldopData(outer_node, self.gt_type, tuple(outer_origin))

    def get_local_view(
        self, domain: gtir_domain.FieldopDomain, sdfg: dace.SDFG
    ) -> gtir_dataflow.IteratorExpr | gtir_dataflow.MemletExpr:
        """Helper method to access a field in local view, given the compute domain of a field operator."""
        if isinstance(self.gt_type, ts.ScalarType):
            assert isinstance(self.dc_node.desc(sdfg), dace.data.Scalar)
            return gtir_dataflow.MemletExpr(
                dc_node=self.dc_node,
                gt_field=ts.FieldType(dims=[], dtype=self.gt_type),
                subset=dace_subsets.Range.from_string("0"),
            )

        if isinstance(self.gt_type, ts.FieldType):
            it_indices: dict[gtx_common.Dimension, gtir_dataflow.DataExpr]
            if isinstance(self.dc_node.desc(sdfg), dace.data.Scalar):
                assert len(self.gt_type.dims) == 0  # zero-dimensional field
                it_indices = {}
            else:
                # The invariant below is ensured by calling `make_field()` to construct `FieldopData`.
                # The `make_field` constructor converts any local dimension, if present, to `ListType`
                # element type, while leaving the field domain with all global dimensions.
                assert all(dim != gtx_common.DimensionKind.LOCAL for dim in self.gt_type.dims)
                domain_dims = [domain_range.dim for domain_range in domain]
                domain_indices = gtir_domain.get_domain_indices(domain_dims, origin=None)
                it_indices = {
                    dim: gtir_dataflow.SymbolExpr(index, INDEX_DTYPE)
                    for dim, index in zip(domain_dims, domain_indices)
                }
            field_origin = [
                (dim, dace.symbolic.SymExpr(0) if self.origin is None else self.origin[i])
                for i, dim in enumerate(self.gt_type.dims)
            ]
            return gtir_dataflow.IteratorExpr(
                self.dc_node, self.gt_type.dtype, field_origin, it_indices
            )

        raise NotImplementedError(f"Node type {type(self.gt_type)} not supported.")

    def get_symbol_mapping(
        self, dataname: str, sdfg: dace.SDFG
    ) -> dict[str, dace.symbolic.SymExpr]:
        """
        Helper method to create the symbol mapping for array storage in a nested SDFG.

        Args:
            dataname: Name of the data container insiode the nested SDFG.
            sdfg: The parent SDFG where the `FieldopData` object lives.

        Returns:
            Mapping from symbols in nested SDFG to the corresponding symbolic values
            in the parent SDFG. This includes the range start and stop symbols (used
            to calculate the array shape as range 'stop - start') and the strides.
        """
        if isinstance(self.gt_type, ts.ScalarType):
            return {}
        outer_desc = self.dc_node.desc(sdfg)
        assert isinstance(outer_desc, dace.data.Array)
        # origin and size of the local dimension, in case of a field with `ListType` data,
        # are assumed to be compiled-time values (not symbolic), therefore the start and
        # stop range symbols of the inner field only extend over the global dimensions.
        if isinstance(self.gt_type.dtype, ts.ListType):
            local_dim = self.gt_type.dtype.offset_type
            assert local_dim is not None
            full_dims = gtx_common.order_dimensions([*self.gt_type.dims, local_dim])
            globals_size = [
                size
                for dim, size in zip(full_dims, outer_desc.shape, strict=True)
                if dim != local_dim
            ]
        else:
            globals_size = outer_desc.shape

        symbol_mapping: dict[str, dace.symbolic.SymbolicType] = {}
        for dim, origin, size in zip(self.gt_type.dims, self.origin, globals_size, strict=True):
            symbol_mapping |= {
                gtx_dace_utils.range_start_symbol(dataname, dim): origin,
                gtx_dace_utils.range_stop_symbol(dataname, dim): (origin + size),
            }
        for i, stride in enumerate(outer_desc.strides):
            symbol_mapping |= {
                gtx_dace_utils.field_stride_symbol_name(dataname, i): stride,
            }
        return symbol_mapping


FieldopResult: TypeAlias = MaybeNestedInTuple[FieldopData | None]
"""Result of a field operator, can be either a field or a tuple fields."""


@dataclasses.dataclass(frozen=True)
class SymbolicData:
    gt_type: ts.ScalarType
    value: dace.symbolic.SymbolicType


INDEX_DTYPE: Final[dace.typeclass] = dace.dtype_to_typeclass(gtx_fbuiltins.IndexType)
"""Data type used for field indexing."""


def flatten_tuple(sym: gtir.Sym, arg: FieldopResult) -> list[tuple[str, FieldopData | None]]:
    """
    Visit a `FieldopResult`, potentially containing nested tuples, and construct a list
    of pairs `(str, FieldopData)` containing the symbol name of each tuple field and
    the corresponding `FieldopData`.
    """
    if isinstance(arg, tuple):
        assert isinstance(sym.type, ts.TupleType)
        tuple_symbols = gtir_to_sdfg_utils.flatten_tuple_fields(sym.id, sym.type)
        tuple_data_fields = gtx_utils.flatten_nested_tuple(arg)
        return [
            (str(tsym.id), tfield)
            for tsym, tfield in zip(tuple_symbols, tuple_data_fields, strict=True)
        ]
    else:
        return [(sym.id, arg)]
