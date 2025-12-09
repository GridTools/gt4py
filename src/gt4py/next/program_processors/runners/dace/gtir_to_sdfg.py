# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Contains visitors to lower GTIR to DaCe SDFG.

Note: this module covers the fieldview flavour of GTIR.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

import dace
from dace.frontend.python import astutils as dace_astutils

from gt4py import eve
from gt4py.eve import concepts
from gt4py.next import common as gtx_common, config as gtx_config, utils as gtx_utils
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, domain_utils
from gt4py.next.iterator.transforms import prune_casts as ir_prune_casts, symbol_ref_utils
from gt4py.next.iterator.type_system import inference as gtir_type_inference
from gt4py.next.program_processors.runners.dace import (
    gtir_domain,
    gtir_to_sdfg_concat_where,
    gtir_to_sdfg_primitives,
    gtir_to_sdfg_types,
    gtir_to_sdfg_utils,
    utils as gtx_dace_utils,
)
from gt4py.next.type_system import type_specifications as ts, type_translation as tt


def _replace_connectors_in_code_string(
    code: str, language: dace.dtypes.Language, connector_mapping: Mapping[str, str]
) -> str:
    """Helper function to replace connector names in the code of a Python tasklet."""
    code_block = dace.properties.CodeBlock(code, language)
    transformed_code_stmts = [
        dace_astutils.ASTFindReplace(connector_mapping).visit(stmt) for stmt in code_block.code
    ]
    return dace.properties.CodeBlock(transformed_code_stmts, language).as_string


class DataflowBuilder(Protocol):
    """Visitor interface to build a dataflow subgraph."""

    @abc.abstractmethod
    def get_offset_provider_type(self, offset: str) -> gtx_common.OffsetProviderTypeElem: ...

    @abc.abstractmethod
    def unique_nsdfg_name(self, sdfg: dace.SDFG, prefix: str) -> str: ...

    @abc.abstractmethod
    def unique_map_name(self, name: str) -> str: ...

    @abc.abstractmethod
    def unique_tasklet_name(self, name: str) -> str: ...

    def add_temp_array(
        self, sdfg: dace.SDFG, shape: Sequence[Any], dtype: dace.dtypes.typeclass
    ) -> tuple[str, dace.data.Scalar]:
        """Add a temporary array to the SDFG."""
        return sdfg.add_temp_transient(shape, dtype)

    def add_temp_array_like(
        self, sdfg: dace.SDFG, datadesc: dace.data.Array
    ) -> tuple[str, dace.data.Scalar]:
        """Add a temporary array to the SDFG."""
        return sdfg.add_temp_transient_like(datadesc)

    def add_temp_scalar(
        self, sdfg: dace.SDFG, dtype: dace.dtypes.typeclass
    ) -> tuple[str, dace.data.Scalar]:
        """Add a temporary scalar to the SDFG."""
        temp_name = sdfg.temp_data_name()
        return sdfg.add_scalar(temp_name, dtype, transient=True)

    def add_map(
        self,
        name: str,
        state: dace.SDFGState,
        ndrange: Union[
            Dict[str, Union[str, dace.subsets.Subset]],
            List[Tuple[str, Union[str, dace.subsets.Subset]]],
        ],
        **kwargs: Any,
    ) -> Tuple[dace.nodes.MapEntry, dace.nodes.MapExit]:
        """Wrapper of `dace.SDFGState.add_map` that assigns unique name."""
        unique_name = self.unique_map_name(name)
        return state.add_map(unique_name, ndrange, **kwargs)

    def add_tasklet(
        self,
        name: str,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        inputs: set[str] | Mapping[str, dace.dtypes.typeclass | None],
        outputs: set[str] | Mapping[str, dace.dtypes.typeclass | None],
        code: str,
        language: dace.dtypes.Language = dace.dtypes.Language.Python,
        **kwargs: Any,
    ) -> dace.nodes.Tasklet:
        """Wrapper of `dace.SDFGState.add_tasklet` that assigns a unique name.

        It also modifies the tasklet connectors by adding a prefix string (see
        `gtir_to_sdfg_utils.get_tasklet_connector()`), in order to avoid name conflicts
        with SDFG data. Otherwise, SDFG validation would detect such conflicts and fail.
        """
        if isinstance(inputs, set):
            inputs = {k: None for k in sorted(inputs)}
        if isinstance(outputs, set):
            outputs = {k: None for k in sorted(outputs)}
        assert inputs.keys().isdisjoint(outputs.keys())

        connector_mapping = {
            conn: gtir_to_sdfg_utils.make_tasklet_connector_for(conn)
            for conn in (inputs.keys() | outputs.keys())
        }
        new_code = _replace_connectors_in_code_string(code, language, connector_mapping)

        inputs = {connector_mapping[k]: v for k, v in inputs.items()}
        outputs = {connector_mapping[k]: v for k, v in outputs.items()}

        unique_name = self.unique_tasklet_name(name)
        tasklet_node = state.add_tasklet(
            unique_name, inputs, outputs, new_code, language=language, **kwargs
        )
        return tasklet_node, connector_mapping

    def add_mapped_tasklet(
        self,
        name: str,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        map_ranges: Mapping[str, str | dace.subsets.Subset],
        inputs: Mapping[str, dace.Memlet],
        code: str,
        outputs: Mapping[str, dace.Memlet],
        language: dace.dtypes.Language = dace.dtypes.Language.Python,
        **kwargs: Any,
    ) -> tuple[dace.nodes.Tasklet, dace.nodes.MapEntry, dace.nodes.MapExit, dict[str, str]]:
        """Wrapper of `dace.SDFGState.add_mapped_tasklet` that assigns a unique name.

        It also modifies the tasklet connectors, in the same way as `add_tasklet()`.
        """
        assert inputs.keys().isdisjoint(outputs.keys())

        connector_mapping = {
            conn: gtir_to_sdfg_utils.make_tasklet_connector_for(conn)
            for conn in (inputs.keys() | outputs.keys())
        }
        new_code = _replace_connectors_in_code_string(code, language, connector_mapping)

        inputs = {connector_mapping[k]: v for k, v in inputs.items()}
        outputs = {connector_mapping[k]: v for k, v in outputs.items()}
        unique_name = self.unique_tasklet_name(name)
        tasklet_node, map_entry, map_exit = state.add_mapped_tasklet(
            unique_name, map_ranges, inputs, new_code, outputs, language=language, **kwargs
        )
        return tasklet_node, map_entry, map_exit, connector_mapping


@dataclasses.dataclass(frozen=True)
class SubgraphContext:
    """Represents the subgraph context in which to lower a GTIR expression to dataflow."""

    sdfg: dace.SDFG
    state: dace.SDFGState


class SDFGBuilder(DataflowBuilder, Protocol):
    """Visitor interface available to GTIR-primitive translators."""

    @abc.abstractmethod
    def make_field(
        self,
        data_node: dace.nodes.AccessNode,
        data_type: ts.FieldType,
    ) -> gtir_to_sdfg_types.FieldopData:
        """Retrieve the field descriptor of a data node, including the origin information.

        In case of `ScalarType` data, the `FieldopData` is constructed with `origin=None`.
        In case of `FieldType` data, the field origin is added to the data descriptor.
        Besides, if the `FieldType` contains a local dimension, the descriptor is converted
        to a canonical form where the field domain consists of all global dimensions
        (the grid axes) and the field data type is `ListType`, with `offset_type` equal
        to the field local dimension.

        TODO(edoapo): consider refactoring this method and moving it to a type module
            close to the `FieldopData` type declaration.

        Args:
            data_node: The access node to the SDFG data storage.
            data_type: The GT4Py data descriptor, which can either come from a program
                symbol, or from an intermediate field for an argument expression.

        Returns:
            The descriptor associated with the SDFG data storage, filled with field origin.
        """
        ...

    @abc.abstractmethod
    def get_symbol_type(self, symbol_name: str) -> ts.DataType:
        """Retrieve the GT4Py type of a symbol used in the SDFG."""
        ...

    @abc.abstractmethod
    def is_column_axis(self, dim: gtx_common.Dimension) -> bool:
        """Check if the given dimension is the column axis."""
        ...

    @abc.abstractmethod
    def setup_nested_context(
        self,
        expr: gtir.Lambda,
        sdfg_name: str,
        parent_ctx: SubgraphContext,
        params: Iterable[gtir.Sym],
        symbolic_inputs: set[str],
        capture_scope_symbols: bool,
    ) -> tuple[SDFGBuilder, SubgraphContext]:
        """
        Create an nested SDFG context to lower a lambda expression, indipendent
        from the current context where the parent expression is being translated.

        This method will setup the global symbols, that correspond to the parameters
        of the expression to be lowered, as well as the set of symbolic arguments,
        that is symbols used in domain expressions or scalar values represented
        as dace symbols in the parent SDFG.

        Args:
            expr: The lambda expression to be lowered as a nested SDFG.
            sdfg_name: The name of the nested SDFG where to lower the given expression.
            parent_ctx: The parent SDFG context.
            params: List of GTIR symbols passed as parameters to the lambda expression.
            symbolic_inputs: Arguments that have to be passed to the nested SDFG
                as dace symbols.
            capture_scope_symbols: When True, the lambda expression will capture
                GTIR symbols defined in the parent scope.

        Returns:
            A visitor object implementing the `SDFGBuilder` protocol.
        """
        ...

    def add_nested_sdfg(
        self,
        node: gtir.Lambda,
        inner_ctx: SubgraphContext,
        outer_ctx: SubgraphContext,
        symbolic_args: Mapping[str, gtir_to_sdfg_types.SymbolicData],
        data_args: Mapping[str, gtir_to_sdfg_types.FieldopData],
        inner_result: gtir_to_sdfg_types.FieldopResult,
    ) -> tuple[dace.nodes.NestedSDFG, Mapping[str, dace.Memlet]]:
        """
        Helper function that prepares the input connections and symbol mapping before
        calling `SDFG.add_nestd_sdfg()` to add the given SDFG as a nested SDFG node
        inside the parent SDFG.

        Args:
            node: The lambda GTIR node containing the expression which was lowered in `inner_ctx`.
            inner_ctx: The nested SDFG context, containing the state where `inner_result` is written.
            outer_ctx: The parent SDFG context, containing the state where the nested SDFG should be added.
            symbolic_args: Scalar argumemts to be passed to the nested SDFG through symbol mapping.
            data_args: Data arguments to be passed through edge memlets.
            inner_result: The data produced by the nested SDFG, inside the state specified by `inner_ctx`.

        Returns:
            A tuple of two elements:
            - The nested SDFG graph node.
            - The mapping from input connectors to data memlets.
        """
        ...

    @abc.abstractmethod
    def visit(self, node: concepts.RootNode, **kwargs: Any) -> Any:
        """Visit a node of the GT4Py IR."""
        ...


def _flatten_tuple_symbols(symbols: Iterable[gtir.Sym]) -> list[gtir.Sym]:
    """
    Helper function to flatten tuple symbols, recursively in case of nested tuples,
    and extract all scalar symbols.
    """
    flat_symbols: list[gtir.Sym] = []
    for sym in symbols:
        if isinstance(sym.type, ts.TupleType):
            flat_symbols.extend(
                f for f in gtir_to_sdfg_utils.flatten_tuple_fields(sym.id, sym.type)
            )
        else:
            flat_symbols.append(sym)
    return flat_symbols


def _make_access_index_for_field(
    domain: gtir_domain.FieldopDomain, data: gtir_to_sdfg_types.FieldopData
) -> dace.subsets.Range:
    """Helper method to build a memlet subset of a field over the given domain."""
    # convert domain expression to dictionary to ease access to the dimensions,
    # since the access indices have to follow the order of dimensions in field domain
    if isinstance(data.gt_type, ts.FieldType) and len(data.gt_type.dims) != 0:
        assert data.origin is not None
        domain_ranges = {
            domain_range.dim: (domain_range.start, domain_range.stop) for domain_range in domain
        }
        return dace.subsets.Range(
            (domain_ranges[dim][0] - origin, domain_ranges[dim][1] - origin - 1, 1)
            for dim, origin in zip(data.gt_type.dims, data.origin, strict=True)
        )
    else:
        assert len(domain) == 0
        return dace.subsets.Range.from_string("0")


@dataclasses.dataclass(frozen=True)
class GTIRToSDFG(eve.NodeVisitor, SDFGBuilder):
    """Provides translation capability from a GTIR program to a DaCe SDFG.

    This class is responsible for translation of `ir.Program`, that is the top level representation
    of a GT4Py program as a sequence of `ir.Stmt` (aka statement) expressions.
    Each statement is translated to a taskgraph inside a separate state. Statement states are chained
    one after the other: concurrency between states should be extracted by means of SDFG analysis.
    The translator will extend the SDFG while preserving the property of single exit state:
    branching is allowed within the context of one statement, but in that case the statement should
    terminate with a join state; the join state will represent the head state for next statement,
    from where to continue building the SDFG.
    """

    offset_provider_type: gtx_common.OffsetProviderType
    column_axis: Optional[gtx_common.Dimension]
    scope_symbols: dict[str, ts.DataType]
    map_uids: eve.utils.UIDGenerator = dataclasses.field(
        init=False, repr=False, default_factory=lambda: eve.utils.UIDGenerator(prefix="map")
    )
    tasklet_uids: eve.utils.UIDGenerator = dataclasses.field(
        init=False, repr=False, default_factory=lambda: eve.utils.UIDGenerator(prefix="tlet")
    )

    def get_offset_provider_type(self, offset: str) -> gtx_common.OffsetProviderTypeElem:
        return gtx_common.get_offset_type(self.offset_provider_type, offset)

    def make_field(
        self,
        data_node: dace.nodes.AccessNode,
        data_type: ts.FieldType,
    ) -> gtir_to_sdfg_types.FieldopData:
        local_dims = [dim for dim in data_type.dims if dim.kind == gtx_common.DimensionKind.LOCAL]
        if len(local_dims) == 0:
            # do nothing: the field domain consists of all global dimensions
            field_type = data_type
        elif len(local_dims) == 1:
            local_dim = local_dims[0]
            # the local dimension is converted into `ListType` data element
            if not isinstance(data_type.dtype, ts.ScalarType):
                raise ValueError(f"Invalid field type {data_type}.")
            if not gtx_common.has_offset(self.offset_provider_type, local_dim.value):
                raise ValueError(
                    f"The provided local dimension {local_dim} does not match any offset provider type."
                )
            local_type = ts.ListType(element_type=data_type.dtype, offset_type=local_dim)
            field_type = ts.FieldType(
                dims=[dim for dim in data_type.dims if dim != local_dim], dtype=local_type
            )
        else:
            raise NotImplementedError(
                "Fields with more than one local dimension are not supported."
            )
        field_origin = tuple(
            gtx_dace_utils.range_start_symbol(data_node.data, dim) for dim in field_type.dims
        )
        return gtir_to_sdfg_types.FieldopData(data_node, field_type, field_origin)

    def get_symbol_type(self, symbol_name: str) -> ts.DataType:
        return self.scope_symbols[symbol_name]

    def is_column_axis(self, dim: gtx_common.Dimension) -> bool:
        assert self.column_axis
        return dim == self.column_axis

    def setup_nested_context(
        self,
        expr: gtir.Lambda,
        sdfg_name: str,
        parent_ctx: SubgraphContext,
        params: Iterable[gtir.Sym],
        symbolic_inputs: set[str],
        capture_scope_symbols: bool,
    ) -> tuple[SDFGBuilder, SubgraphContext]:
        lambda_pnames = {str(p.id) for p in params}
        assert symbolic_inputs.issubset(lambda_pnames) and all(
            isinstance(p.type, ts.ScalarType) for p in params if str(p.id) in symbolic_inputs
        )

        # If `capture_scope_symbols` is True, besides the values mapped to the parameters
        # of the lambda expression (`lambda_params`), we also pass to the nested SDFG
        # all GTIR-symbols in scope, which means the symbols defined in current context.
        if capture_scope_symbols:
            lambda_symbols = {
                sym: self.scope_symbols[sym]
                for sym in symbol_ref_utils.collect_symbol_refs(expr, self.scope_symbols.keys())
            }
        else:
            lambda_symbols = {}

        assert all(isinstance(p.type, ts.DataType) for p in params)
        lambda_symbols |= {
            str(p.id): p.type  # type: ignore[misc]
            for p in params
        }
        all_params = [gtir.Sym(id=p_name, type=p_type) for p_name, p_type in lambda_symbols.items()]

        sdfg = dace.SDFG(name=self.unique_nsdfg_name(parent_ctx.sdfg, sdfg_name))
        sdfg.debuginfo = gtir_to_sdfg_utils.debug_info(expr, default=parent_ctx.sdfg.debuginfo)
        state = sdfg.add_state(f"{sdfg_name}_entry")
        nested_ctx = SubgraphContext(sdfg, state)
        nsdfg_builder = GTIRToSDFG(
            offset_provider_type=self.offset_provider_type,
            column_axis=self.column_axis,
            scope_symbols=lambda_symbols,
        )

        # Scalar GTIR-symbols represented as dace symbols in parent SDFG are mapped
        # to dace symbols in the nested SDFG.
        parent_symbols = set(
            s for p in _flatten_tuple_symbols(all_params) if (s := str(p.id)) not in lambda_pnames
        ).intersection(parent_ctx.sdfg.symbols.keys())

        # All GTIR-symbols accessed in domain expressions by the lambda need to be
        # represented as dace symbols.
        domain_symrefs = (
            eve.walk_values(expr)
            .filter(lambda node: cpm.is_call_to(node, ("cartesian_domain", "unstructured_domain")))
            .map(
                lambda domain: eve.walk_values(domain)
                .if_isinstance(gtir.SymRef)
                .filter(lambda sym: str(sym.id) in lambda_symbols)
                .to_set()
            )
            .reduce(lambda x, y: x | y, init=set())
        )
        domain_symbols = {str(p.id) for p in _flatten_tuple_symbols(domain_symrefs)}

        nsdfg_builder._add_sdfg_params(
            sdfg,
            node_params=all_params,
            symbolic_params=(domain_symbols | parent_symbols | symbolic_inputs),
        )
        return nsdfg_builder, nested_ctx

    def add_nested_sdfg(
        self,
        node: gtir.Lambda,
        inner_ctx: SubgraphContext,
        outer_ctx: SubgraphContext,
        symbolic_args: Mapping[str, gtir_to_sdfg_types.SymbolicData],
        data_args: Mapping[str, gtir_to_sdfg_types.FieldopData],
        inner_result: gtir_to_sdfg_types.FieldopResult,
    ) -> tuple[dace.nodes.NestedSDFG, Mapping[str, dace.Memlet]]:
        assert data_args.keys().isdisjoint(symbolic_args.keys())

        # Collect the names of all output data, by flattening any tuple structure.
        lambda_output_data = (
            gtx_utils.flatten_nested_tuple(inner_result)
            if isinstance(inner_result, tuple)
            else [inner_result]
        )
        # The output connectors only need to be setup for the actual result of the
        # internal dataflow that writes to some sink data nodes of the nested SDFG.
        # All sink nodes inside the nested SDFG which contain the lambda result
        # are expected to be transient data. The caller is responsible for turning
        # them into global data and map it to arrays outside the nested SDFG.
        lambda_outputs = {
            output.dc_node.data
            for output in lambda_output_data
            if output is not None and output.dc_node.desc(inner_ctx.sdfg).transient
        }

        # Map free symbols to parent SDFG
        nsdfg_symbols_mapping = {}
        for dc_symbol in inner_ctx.sdfg.free_symbols:
            if dc_symbol in data_args:
                assert isinstance(data_args[dc_symbol].gt_type, ts.ScalarType)
                raise NotImplementedError(
                    "Unexpected mapping of scalar node to symbol on nested SDFG."
                )
            elif dc_symbol in symbolic_args:
                nsdfg_symbols_mapping[dc_symbol] = symbolic_args[dc_symbol].value
            else:
                nsdfg_symbols_mapping[dc_symbol] = dc_symbol
        for gt_symbol, arg in data_args.items():
            nsdfg_symbols_mapping |= arg.get_symbol_mapping(gt_symbol, outer_ctx.sdfg)

        connectivity_arrays = {
            gtx_dace_utils.connectivity_identifier(offset)
            for offset in gtx_dace_utils.filter_connectivity_types(self.offset_provider_type)
        }

        input_memlets = {}
        unused_data = set()
        for nsdfg_dataname, nsdfg_datadesc in inner_ctx.sdfg.arrays.items():
            if nsdfg_datadesc.transient:
                pass  # nothing to do here
            elif nsdfg_dataname in data_args:
                arg_node = data_args[nsdfg_dataname]
                source_data = arg_node.dc_node.data
                input_memlets[nsdfg_dataname] = outer_ctx.sdfg.make_array_memlet(source_data)
            elif nsdfg_dataname in outer_ctx.sdfg.arrays:
                # include fields from the parent scope
                source_data = nsdfg_dataname
                # ensure that connectivity tables are non-transient arrays in parent SDFG
                if source_data in connectivity_arrays:
                    outer_ctx.sdfg.arrays[source_data].transient = False
                input_memlets[nsdfg_dataname] = outer_ctx.sdfg.make_array_memlet(source_data)
            else:
                # Arguments with empty domain do not need to be connected on the nested SDFG.
                unused_data.add(nsdfg_dataname)

        for data in sorted(unused_data):  # NOTE: remove the data in deterministic order
            inner_ctx.sdfg.remove_data(data, validate=gtx_config.DEBUG)

        nsdfg_node = outer_ctx.state.add_nested_sdfg(
            inner_ctx.sdfg,
            inputs=input_memlets.keys(),
            outputs=lambda_outputs,
            symbol_mapping=nsdfg_symbols_mapping,
            debuginfo=gtir_to_sdfg_utils.debug_info(node, default=outer_ctx.sdfg.debuginfo),
        )

        return nsdfg_node, input_memlets

    def unique_nsdfg_name(self, sdfg: dace.SDFG, prefix: str) -> str:
        nsdfg_list = [
            nsdfg.label for nsdfg in sdfg.all_sdfgs_recursive() if nsdfg.label.startswith(prefix)
        ]
        return f"{prefix}_{len(nsdfg_list)}"

    def unique_map_name(self, name: str) -> str:
        return f"{self.map_uids.sequential_id()}_{name}"

    def unique_tasklet_name(self, name: str) -> str:
        return f"{self.tasklet_uids.sequential_id()}_{name}"

    def _make_array_shape_and_strides(
        self, name: str, dims: Sequence[gtx_common.Dimension]
    ) -> tuple[list[dace.symbolic.SymbolicType], list[dace.symbolic.SymbolicType]]:
        """
        Parse field dimensions and allocate symbols for array shape and strides.

        For local dimensions, the size is known at compile-time and therefore
        the corresponding array shape dimension is set to an integer literal value.

        This method is only called for non-transient arrays, which require symbolic
        memory layout. The memory layout of transient arrays, used for temporary
        fields, is left to the DaCe default (row major, not necessarily the optimal
        one) and might be changed during optimization.

        Returns:
            Two lists of symbols, one for the shape and the other for the strides of the array.
        """
        neighbor_table_types = gtx_dace_utils.filter_connectivity_types(self.offset_provider_type)
        shape = []
        for dim in dims:
            if dim.kind == gtx_common.DimensionKind.LOCAL:
                # for local dimension, the size is taken from the associated connectivity type
                shape.append(neighbor_table_types[dim.value].max_neighbors)
            elif gtx_dace_utils.is_connectivity_identifier(name, self.offset_provider_type):
                # we use symbolic size for the global dimension of a connectivity
                shape.append(gtx_dace_utils.field_size_symbol(name, dim, neighbor_table_types))
            else:
                # the size of global dimensions for a regular field is the symbolic
                # expression of domain range 'stop - start'
                shape.append(
                    dace.symbolic.pystr_to_symbolic(
                        "{} - {}".format(
                            gtx_dace_utils.range_stop_symbol(name, dim),
                            gtx_dace_utils.range_start_symbol(name, dim),
                        )
                    )
                )
        strides = [
            gtx_dace_utils.field_stride_symbol(name, dim, neighbor_table_types) for dim in dims
        ]
        return shape, strides

    def _add_storage(
        self,
        sdfg: dace.SDFG,
        symbolic_params: set[str] | None,
        name: str,
        gt_type: ts.DataType,
        transient: bool = True,
    ) -> list[tuple[str, ts.DataType]]:
        """
        Add storage in the SDFG for a given GT4Py data symbol.

        GT4Py fields are allocated as DaCe arrays. GT4Py scalars are represented
        as scalar data containers, unless they are listed in `symbolic_arguments`,
        in which case they are represented as SDFG symbols. This is the case of
        start/stop symbols for field range or other scalar values accessed in
        domain symbolic expressions.

        The fields used as temporary arrays, when `transient = True`, are allocated
        and exist only within the SDFG; when `transient = False`, the fields have
        to be allocated outside and have to be passed as arguments to the SDFG call.

        Args:
            sdfg: The SDFG where storage needs to be allocated.
            symbolic_params: GT4Py scalars that must be represented as SDFG symbols.
                If `None`, all scalar parameters should be represented as dace symbols.
            name: Symbol Name to be allocated.
            gt_type: GT4Py symbol type.
            transient: True when the data symbol has to be allocated as internal storage.

        Returns:
            List of tuples '(data_name, gt_type)' where 'data_name' is the name of
            the data container used as storage in the SDFG and 'gt_type' is the
            corresponding GT4Py type. In case the storage has to be allocated for
            a tuple symbol the list contains a flattened version of the tuple,
            otherwise the list will contain a single entry.
        """
        if isinstance(gt_type, ts.TupleType):
            tuple_fields = []
            for sym in gtir_to_sdfg_utils.flatten_tuple_fields(name, gt_type):
                assert isinstance(sym.type, ts.DataType)
                tuple_fields.extend(
                    self._add_storage(
                        sdfg=sdfg,
                        symbolic_params=symbolic_params,
                        name=str(sym.id),
                        gt_type=sym.type,
                        transient=transient,
                    )
                )
            return tuple_fields

        elif isinstance(gt_type, ts.FieldType):
            if len(gt_type.dims) == 0:
                # represent zero-dimensional fields as scalar arguments
                return self._add_storage(
                    sdfg=sdfg,
                    symbolic_params=set(),  # force use of scalar data container
                    name=name,
                    gt_type=gt_type.dtype,
                    transient=transient,
                )
            if isinstance(gt_type.dtype, ts.ScalarType):
                dc_dtype = gtx_dace_utils.as_dace_type(gt_type.dtype)
                all_dims = gt_type.dims
            elif not transient:  # 'ts.ListType': use 'offset_type' as local dimension
                assert gt_type.dtype.offset_type is not None
                assert gt_type.dtype.offset_type.kind == gtx_common.DimensionKind.LOCAL
                assert isinstance(gt_type.dtype.element_type, ts.ScalarType)
                dc_dtype = gtx_dace_utils.as_dace_type(gt_type.dtype.element_type)
                all_dims = gtx_common.order_dimensions([*gt_type.dims, gt_type.dtype.offset_type])
            else:
                # By design, the domain of temporary fields used by SDFG lowering
                # contains only the global dimensions. The local dimension is extracted,
                # when needed, from the GTIR data type (`ts.ListType`).
                raise ValueError("Unexpected local dimension in temporary field domain.")
            # Use symbolic shape, which allows to invoke the program with fields of different size;
            # and symbolic strides, which enables decoupling the memory layout from generated code.
            sym_shape, sym_strides = self._make_array_shape_and_strides(name, all_dims)
            sdfg.add_array(name, sym_shape, dc_dtype, strides=sym_strides, transient=transient)
            return [(name, gt_type)]

        elif isinstance(gt_type, ts.ScalarType):
            dc_dtype = gtx_dace_utils.as_dace_type(gt_type)
            if symbolic_params is None or name in symbolic_params:
                sdfg.add_symbol(name, dc_dtype)
            else:
                sdfg.add_scalar(name, dc_dtype, transient=transient)

            return [(name, gt_type)]

        raise RuntimeError(f"Data type '{type(gt_type)}' not supported.")

    def _add_storage_for_temporary(self, temp_decl: gtir.Temporary) -> dict[str, str]:
        """
        Add temporary storage (aka transient) for data containers used as GTIR temporaries.

        Assume all temporaries to be fields, therefore represented as dace arrays.
        """
        raise NotImplementedError("Temporaries not supported yet by GTIR DaCe backend.")

    def _visit_expression(
        self,
        node: gtir.Expr,
        sdfg: dace.SDFG,
        head_state: dace.SDFGState,
        use_temp: bool = True,
    ) -> gtir_to_sdfg_types.FieldopResult:
        """
        Specialized visit method for fieldview expressions.

        This method represents the entry point to visit `ir.Stmt` expressions.
        As such, it must preserve the property of single exit state in the SDFG.

        Returns:
            The SDFG array nodes containing the result of the fieldview expression.
            The nodes are organized in tree form, in case of tuples.
        """

        ctx = SubgraphContext(sdfg, head_state)
        result = self.visit(node, ctx=ctx)

        # sanity check: each statement should preserve the property of single exit state (aka head state),
        # i.e. eventually only introduce internal branches, and keep the same head state
        sink_states = sdfg.sink_nodes()
        assert len(sink_states) == 1
        assert sink_states[0] == head_state

        def _visit_result(
            src: gtir_to_sdfg_types.FieldopData,
        ) -> gtir_to_sdfg_types.FieldopData:
            src_desc = sdfg.arrays[src.dc_node.data]
            if src_desc.transient or not use_temp:
                return src
            else:
                dst, dst_desc = self.add_temp_array_like(sdfg, src_desc)
                dst_node = head_state.add_access(dst)
                head_state.add_nedge(
                    src.dc_node,
                    dst_node,
                    dace.Memlet(
                        data=dst,
                        subset=dace.subsets.Range.from_array(dst_desc),
                        other_subset=dace.subsets.Range.from_array(src_desc),
                    ),
                )
                return gtir_to_sdfg_types.FieldopData(dst_node, src.gt_type, src.origin)

        return gtx_utils.tree_map(_visit_result)(result)

    def _add_sdfg_params(
        self,
        sdfg: dace.SDFG,
        node_params: Sequence[gtir.Sym],
        symbolic_params: set[str] | None,
    ) -> list[str]:
        """
        Helper function to add storage for node parameters and connectivity tables.

        For details about storage allocation of each data type, see `_add_storage()`.
        """

        # add non-transient arrays and/or SDFG symbols for the program arguments
        sdfg_args = []
        for param in node_params:
            gt_symbol_name = str(param.id)
            assert isinstance(param.type, (ts.DataType))
            sdfg_args += self._add_storage(
                sdfg=sdfg,
                symbolic_params=symbolic_params,
                name=gt_symbol_name,
                gt_type=param.type,
                transient=False,
            )

        # add SDFG storage for connectivity tables
        for offset, connectivity_type in gtx_dace_utils.filter_connectivity_types(
            self.offset_provider_type
        ).items():
            gt_type = ts.FieldType(
                dims=[connectivity_type.source_dim, connectivity_type.neighbor_dim],
                dtype=tt.from_dtype(connectivity_type.dtype),
            )
            # We store all connectivity tables as transient arrays here; later, while building
            # the field operator expressions, we change to non-transient (i.e. allocated externally)
            # the tables that are actually used. This way, we avoid adding SDFG arguments for
            # the connectivity tables that are not used. The remaining unused transient arrays
            # are removed by the dace simplify pass.
            self._add_storage(
                sdfg=sdfg,
                symbolic_params=symbolic_params,
                name=gtx_dace_utils.connectivity_identifier(offset),
                gt_type=gt_type,
            )

        # the list of all sdfg arguments (aka non-transient arrays) which include tuple-element fields
        return [arg_name for arg_name, _ in sdfg_args]

    def visit_Program(self, node: gtir.Program) -> dace.SDFG:
        """Translates `ir.Program` to `dace.SDFG`.

        First, it will allocate field and scalar storage for global data. The storage
        represents global data, available everywhere in the SDFG, either containing
        external data (aka non-transient data) or temporary data (aka transient data).
        The temporary data is global, therefore available everywhere in the SDFG
        but not outside. Then, all statements are translated, one after the other.
        """
        sdfg = dace.SDFG(node.id)
        sdfg.debuginfo = gtir_to_sdfg_utils.debug_info(node)

        # start block of the stateful graph
        entry_state = sdfg.add_state("program_entry", is_start_block=True)

        # declarations of temporaries result in transient array definitions in the SDFG
        if node.declarations:
            temp_symbols: dict[str, str] = {}
            for decl in node.declarations:
                temp_symbols |= self._add_storage_for_temporary(decl)

            # define symbols for shape and offsets of temporary arrays as interstate edge symbols
            head_state = sdfg.add_state_after(entry_state, "init_temps", assignments=temp_symbols)
        else:
            head_state = entry_state

        # By passing `symbolic_arguments=None` all scalars are represented as dace symbols.
        #   We do this to allow lowering of scalar expressions in let-statements,
        #   that only depend on scalar parameters, as dace symbolic expressions
        #   mapped to symbols on a nested SDFG.
        sdfg_arg_names = self._add_sdfg_params(sdfg, node.params, symbolic_params=None)

        # visit one statement at a time and expand the SDFG from the current head state
        for i, stmt in enumerate(node.body):
            # include `debuginfo` only for `ir.Program` and `ir.Stmt` nodes: finer granularity would be too messy
            head_state = sdfg.add_state_after(head_state, f"stmt_{i}")
            head_state._debuginfo = gtir_to_sdfg_utils.debug_info(stmt, default=sdfg.debuginfo)
            head_state = self.visit(stmt, sdfg=sdfg, state=head_state)

        # remove unused connectivity tables (by design, arrays are marked as non-transient when they are used)
        for nsdfg in sdfg.all_sdfgs_recursive():
            unused_connectivities = [
                data
                for data, datadesc in nsdfg.arrays.items()
                if gtx_dace_utils.is_connectivity_identifier(data, self.offset_provider_type)
                and datadesc.transient
            ]
            for data in unused_connectivities:
                assert isinstance(nsdfg.arrays[data], dace.data.Array)
                nsdfg.arrays.pop(data)

        # Create the call signature for the SDFG.
        #  Only the arguments required by the GT4Py program, i.e. `node.params`, are added
        #  as positional arguments. The implicit arguments, such as the offset providers or
        #  the arguments created by the translation process, must be passed as keyword arguments.
        sdfg.arg_names = sdfg_arg_names

        sdfg.validate()
        return sdfg

    def visit_SetAt(
        self, stmt: gtir.SetAt, sdfg: dace.SDFG, state: dace.SDFGState
    ) -> dace.SDFGState:
        """Visits a `SetAt` statement expression and writes the local result to some external storage.

        Each statement expression results in some sort of dataflow gragh writing to temporary storage.
        The translation of `SetAt` ensures that the result is written back to the target external storage.

        Returns:
          The SDFG head state, eventually updated if the target write requires a new state.
        """

        # Visit the domain expression.
        domain = gtir_domain.extract_target_domain(stmt.domain)

        # Visit the field operator expression.
        source_tree = self._visit_expression(stmt.expr, sdfg, state)

        # The target expression could be a `SymRef` to an output field or a `make_tuple`
        # expression in case the statement returns more than one field.
        target_tree = self._visit_expression(stmt.target, sdfg, state, use_temp=False)

        expr_input_args = {
            sym_id
            for sym in eve.walk_values(stmt.expr).if_isinstance(gtir.SymRef)
            if (sym_id := str(sym.id)) in sdfg.arrays
        }
        state_input_data = {
            node.data
            for node in state.data_nodes()
            if node.data in expr_input_args and state.degree(node) != 0
        }

        # For inout argument, write the result in separate next state
        # this is needed to avoid undefined behavior for expressions like: X, Y = X + 1, X
        # If this state is not used, we remove it before returning from the function.
        target_state = sdfg.add_state_after(state, f"post_{state.label}")

        def _visit_target(
            source: gtir_to_sdfg_types.FieldopData,
            target: gtir_to_sdfg_types.FieldopData,
            target_domain: domain_utils.SymbolicDomain,
            target_state: dace.SDFGState,
        ) -> None:
            target_desc = sdfg.arrays[target.dc_node.data]
            assert not target_desc.transient

            assert source.gt_type == target.gt_type
            field_domain = gtir_domain.get_field_domain(target_domain)
            source_subset = _make_access_index_for_field(field_domain, source)
            target_subset = _make_access_index_for_field(field_domain, target)

            if target.dc_node.data in state_input_data:
                # create new access nodes in the target state
                target_state.add_nedge(
                    target_state.add_access(source.dc_node.data),
                    target_state.add_access(target.dc_node.data),
                    dace.Memlet(
                        data=target.dc_node.data, subset=target_subset, other_subset=source_subset
                    ),
                )
                # remove isolated access node
                state.remove_node(target.dc_node)
            else:
                state.add_nedge(
                    source.dc_node,
                    target.dc_node,
                    dace.Memlet(
                        data=target.dc_node.data, subset=target_subset, other_subset=source_subset
                    ),
                )

        gtx_utils.tree_map(
            lambda source, target, target_domain: _visit_target(
                source, target, target_domain, target_state
            )
        )(source_tree, target_tree, domain)

        if target_state.is_empty():
            sdfg.remove_node(target_state)
            return state
        else:
            return target_state

    def visit_FunCall(
        self,
        node: gtir.FunCall,
        ctx: SubgraphContext,
    ) -> gtir_to_sdfg_types.FieldopResult:
        # use specialized dataflow builder classes for each builtin function
        if cpm.is_call_to(node, "concat_where"):
            return gtir_to_sdfg_concat_where.translate_concat_where(node, ctx, self)
        elif cpm.is_call_to(node, "if_"):
            return gtir_to_sdfg_primitives.translate_if(node, ctx, self)
        elif cpm.is_call_to(node, "index"):
            return gtir_to_sdfg_primitives.translate_index(node, ctx, self)
        elif cpm.is_call_to(node, "make_tuple"):
            return gtir_to_sdfg_primitives.translate_make_tuple(node, ctx, self)
        elif cpm.is_call_to(node, "tuple_get"):
            return gtir_to_sdfg_primitives.translate_tuple_get(node, ctx, self)
        elif cpm.is_applied_as_fieldop(node):
            return gtir_to_sdfg_primitives.translate_as_fieldop(node, ctx, self)
        elif isinstance(node.fun, gtir.Lambda):
            # Special handling of scalar arguments of a let-lambda that can be lowered
            # as symbolic expressions: when all the GTIR-symbols the scalar expression
            # depends on are dace symbols, the argument can be passed to the nested
            # SDFG by means of symbol mapping.
            symbolic_args = {}
            for p, lambda_arg in zip(node.fun.params, node.args, strict=True):
                if not isinstance(lambda_arg.type, ts.ScalarType):
                    continue
                # Convert the scalar argument to a dace symbolic expression if all
                # of its dependencies are symbols to.
                try:
                    symbolic_expr = gtir_to_sdfg_utils.get_symbolic(lambda_arg)
                except TypeError:
                    # sympy parsing failed, it can happen with 'cast_' expressions
                    if not any(
                        eve.walk_values(lambda_arg).map(lambda node: cpm.is_call_to(node, "cast_"))
                    ):
                        raise
                    continue
                if all(str(s) in ctx.sdfg.symbols for s in symbolic_expr.free_symbols):
                    symbolic_args[str(p.id)] = symbolic_expr
            # All other lambda arguments are lowered to some dataflow that produces a data node.
            args = {
                param: (
                    gtir_to_sdfg_types.SymbolicData(param.type, symbolic_args[gt_symbol_name])  # type: ignore[arg-type]
                    if (gt_symbol_name := str(param.id)) in symbolic_args
                    else self.visit(arg, ctx=ctx)
                )
                for param, arg in zip(node.fun.params, node.args, strict=True)
            }
            return self.visit(node.fun, ctx=ctx, args=args)
        elif isinstance(node.type, ts.ScalarType):
            return gtir_to_sdfg_primitives.translate_scalar_expr(node, ctx, self)
        else:
            raise NotImplementedError(f"Unexpected 'FunCall' expression ({node}).")

    def visit_Lambda(
        self,
        node: gtir.Lambda,
        ctx: SubgraphContext,
        args: Mapping[gtir.Sym, gtir_to_sdfg_types.FieldopResult | gtir_to_sdfg_types.SymbolicData],
    ) -> gtir_to_sdfg_types.FieldopResult:
        """
        Translates a `Lambda` node to a nested SDFG in the current state.

        The reason for creating a nested SDFG is to define local symbols (the lambda
        paremeters) that map to parent fields, either program arguments, temporaries
        or symbolic expressions.

        The arguments passed to the lambda expression are divided in two groups:
        * arguments passed as access nodes, i.e. `as_fieldop` result, field and scalar
          `gtir.SymRef`, or nested let-lambdas;
        * arguments (scalars only) that can be evaluated as symbolic expressions,
          which can be mapped to symbols on the nested SDFG.


        If the lambda has a parameter whose name is already present in `GTIRToSDFG.global_symbols`,
        i.e. a lambda parameter with the same name as a symbol in scope, the parameter will shadow
        the previous symbol during traversal of the lambda expression.
        """

        symbolic_args: dict[str, gtir_to_sdfg_types.SymbolicData] = {}
        lambda_arg_nodes: dict[str, gtir_to_sdfg_types.FieldopData] = {}
        for gtsym, arg in args.items():
            gtsym_id = str(gtsym.id)
            if arg is None:
                pass  # domain inference has detetcted that this argument is not used
            elif isinstance(arg, gtir_to_sdfg_types.SymbolicData):
                symbolic_args[gtsym_id] = arg
            else:
                lambda_arg_nodes |= {
                    str(nested_gtsym.id): nested_arg
                    for nested_gtsym, nested_arg in gtir_to_sdfg_types.flatten_tuple(gtsym, arg)
                    if nested_arg is not None  # we filter out arguments with empty domain
                }

        # lower let-statement lambda node as a nested SDFG
        lambda_translator, lambda_ctx = self.setup_nested_context(
            expr=node,
            sdfg_name="lambda",
            parent_ctx=ctx,
            params=args.keys(),
            symbolic_inputs=set(symbolic_args.keys()),
            capture_scope_symbols=True,
        )

        lambda_result = lambda_translator.visit(node.expr, ctx=lambda_ctx)

        nsdfg_node, input_memlets = self.add_nested_sdfg(
            node=node,
            inner_ctx=lambda_ctx,
            outer_ctx=ctx,
            symbolic_args=symbolic_args,
            data_args=lambda_arg_nodes,
            inner_result=lambda_result,
        )

        for input_connector, memlet in input_memlets.items():
            if input_connector in lambda_arg_nodes:
                arg_node = lambda_arg_nodes[input_connector]
                src_node = arg_node.dc_node
            else:
                src_node = ctx.state.add_access(memlet.data)

            ctx.state.add_edge(src_node, None, nsdfg_node, input_connector, memlet)

        def construct_output_for_nested_sdfg(
            inner_data: gtir_to_sdfg_types.FieldopData,
        ) -> gtir_to_sdfg_types.FieldopData:
            """
            This function makes a data container that lives inside a nested SDFG, denoted by `inner_data`,
            available in the parent SDFG.
            In order to achieve this, the data container inside the nested SDFG is marked as non-transient
            (in other words, externally allocated - a requirement of the SDFG IR) and a new data container
            is created within the parent SDFG, with the same properties (shape, stride, etc.) of `inner_data`
            but appropriatly remapped using the symbol mapping table.
            For lambda arguments that are simply returned by the lambda, the `inner_data` was already mapped
            to a parent SDFG data container, therefore it can be directly accessed in the parent SDFG.
            The same happens to symbols available in the lambda context but not explicitly passed as lambda
            arguments, that are simply returned by the lambda: it can be directly accessed in the parent SDFG.
            """
            inner_desc = inner_data.dc_node.desc(lambda_ctx.sdfg)
            inner_dataname = inner_data.dc_node.data
            if inner_desc.transient:
                # Transient data nodes only exist within the nested SDFG. In order to return some result data,
                # the corresponding data container inside the nested SDFG has to be changed to non-transient,
                # that is externally allocated, as required by the SDFG IR. An output edge will write the result
                # from the nested-SDFG to a new intermediate data container allocated in the parent SDFG.
                outer_data = inner_data.map_to_parent_sdfg(
                    self, lambda_ctx.sdfg, ctx.sdfg, ctx.state, nsdfg_node.symbol_mapping
                )
                ctx.state.add_edge(
                    nsdfg_node,
                    inner_dataname,
                    outer_data.dc_node,
                    None,
                    ctx.sdfg.make_array_memlet(outer_data.dc_node.data),
                )
            elif inner_dataname in lambda_arg_nodes:
                # This if branch and the next one handle the non-transient result nodes.
                # Non-transient nodes are just input nodes that are immediately returned
                # by the lambda expression. Therefore, these nodes are already available
                # in the parent context and can be directly accessed there.
                if (outer_arg := lambda_arg_nodes[inner_dataname]) is None:
                    raise ValueError(f"Unexpected argument with empty domain {inner_data}.")
                outer_data = outer_arg
            else:
                # This must be a symbol captured from the lambda parent scope.
                outer_node = ctx.state.add_access(inner_dataname)
                outer_data = gtir_to_sdfg_types.FieldopData(
                    outer_node, inner_data.gt_type, inner_data.origin
                )
            # Isolated access node will make validation fail.
            # Isolated access nodes can be found in the join-state of an if-expression
            # or in lambda expressions that just construct tuples from input arguments.
            if lambda_ctx.state.degree(inner_data.dc_node) == 0:
                lambda_ctx.state.remove_node(inner_data.dc_node)
            return outer_data

        return gtx_utils.tree_map(construct_output_for_nested_sdfg)(lambda_result)

    def visit_Literal(
        self,
        node: gtir.Literal,
        ctx: SubgraphContext,
    ) -> gtir_to_sdfg_types.FieldopResult:
        return gtir_to_sdfg_primitives.translate_literal(node, ctx, self)

    def visit_SymRef(
        self,
        node: gtir.SymRef,
        ctx: SubgraphContext,
    ) -> gtir_to_sdfg_types.FieldopResult:
        return gtir_to_sdfg_primitives.translate_symbol_ref(node, ctx, self)


def build_sdfg_from_gtir(
    ir: gtir.Program,
    offset_provider_type: gtx_common.OffsetProviderType,
    column_axis: Optional[gtx_common.Dimension] = None,
) -> dace.SDFG:
    """
    Receives a GTIR program and lowers it to a DaCe SDFG.

    The lowering to SDFG requires that the program node is type-annotated, therefore
    this function runs type ineference as first step.

    Args:
        ir: The GTIR program node to be lowered to SDFG
        offset_provider_type: The definitions of offset providers used by the program node
        column_axis: Vertical dimension used for column scan expressions.

    Returns:
        An SDFG in the DaCe canonical form (simplified)
    """

    if ir.function_definitions:
        raise NotImplementedError("Functions expected to be inlined as lambda calls.")

    ir = gtir_type_inference.infer(ir, offset_provider_type=offset_provider_type)
    ir = ir_prune_casts.PruneCasts().visit(ir)

    # DaCe requires C-compatible strings for the names of data containers,
    # such as arrays and scalars. GT4Py uses a unicode symbols ('ᐞ') as name
    # separator in the SSA pass, which generates invalid symbols for DaCe.
    # Here we find new names for invalid symbols present in the IR.
    ir = gtir_to_sdfg_utils.replace_invalid_symbols(ir)

    global_symbols = {str(p.id): p.type for p in ir.params if isinstance(p.type, ts.DataType)}
    sdfg_genenerator = GTIRToSDFG(offset_provider_type, column_axis, global_symbols)
    sdfg = sdfg_genenerator.visit(ir)
    assert isinstance(sdfg, dace.SDFG)

    return sdfg
