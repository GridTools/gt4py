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
import functools
import itertools
import operator
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple, Union

import dace
from dace.sdfg import utils as dace_sdfg_utils

from gt4py import eve
from gt4py.eve import concepts
from gt4py.next import common as gtx_common, utils as gtx_utils
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.iterator.transforms import prune_casts as ir_prune_casts, symbol_ref_utils
from gt4py.next.iterator.type_system import inference as gtir_type_inference
from gt4py.next.program_processors.runners.dace import (
    gtir_builtin_translators,
    gtir_sdfg_utils,
    utils as gtx_dace_utils,
)
from gt4py.next.type_system import type_specifications as ts, type_translation as tt


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
        state: dace.SDFGState,
        inputs: Union[Set[str], Dict[str, dace.dtypes.typeclass]],
        outputs: Union[Set[str], Dict[str, dace.dtypes.typeclass]],
        code: str,
        **kwargs: Any,
    ) -> dace.nodes.Tasklet:
        """Wrapper of `dace.SDFGState.add_tasklet` that assigns unique name."""
        unique_name = self.unique_tasklet_name(name)
        return state.add_tasklet(unique_name, inputs, outputs, code, **kwargs)

    def add_mapped_tasklet(
        self,
        name: str,
        state: dace.SDFGState,
        map_ranges: Dict[str, str | dace.subsets.Subset]
        | List[Tuple[str, str | dace.subsets.Subset]],
        inputs: Dict[str, dace.Memlet],
        code: str,
        outputs: Dict[str, dace.Memlet],
        **kwargs: Any,
    ) -> tuple[dace.nodes.Tasklet, dace.nodes.MapEntry, dace.nodes.MapExit]:
        """Wrapper of `dace.SDFGState.add_mapped_tasklet` that assigns unique name."""
        unique_name = self.unique_tasklet_name(name)
        return state.add_mapped_tasklet(unique_name, map_ranges, inputs, code, outputs, **kwargs)


class SDFGBuilder(DataflowBuilder, Protocol):
    """Visitor interface available to GTIR-primitive translators."""

    @abc.abstractmethod
    def make_field(
        self, data_node: dace.nodes.AccessNode, data_type: ts.FieldType | ts.ScalarType
    ) -> gtir_builtin_translators.FieldopData:
        """Retrieve the field data descriptor including the domain offset information."""
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
        expr: gtir.Expr,
        sdfg: dace.SDFG,
        global_symbols: dict[str, ts.DataType],
        field_offsets: dict[str, Optional[list[dace.symbolic.SymExpr]]],
    ) -> SDFGBuilder:
        """
        Create an SDFG context to translate a nested expression, indipendent
        from the current context where the parent expression is being translated.

        This method will setup the global symbols, that correspond to the parameters
        of the expression to be lowered, as well as the set of symbolic arguments,
        that is scalar values used in internal domain expressions.

        Args:
            expr: The nested expresson to be lowered.
            sdfg: The SDFG where to lower the nested expression.
            global_symbols: Mapping from symbol name to GTIR data type.
            field_offsets: Mapping from symbol name to field origin, `None` if field origin is 0 in all dimensions.

        Returns:
            A visitor object implementing the `SDFGBuilder` protocol.
        """
        ...

    @abc.abstractmethod
    def visit(self, node: concepts.RootNode, **kwargs: Any) -> Any:
        """Visit a node of the GT4Py IR."""
        ...


def _collect_symbols_in_domain_expressions(
    ir: gtir.Node, ir_params: Sequence[gtir.Sym]
) -> set[str]:
    """
    Collect symbols accessed in domain expressions that also appear in the paremeter list.

    This function is used to identify all parameters that are accessed in domain
    expressions. They have to be passed to the SDFG call as DaCe symbols (instead
    of scalars) such that they can be used as bounds in map ranges.

    Args:
        ir: GTIR node to be traversed and where to search for domain expressions.
        ir_params: List of parameters to search for in domain expressions.

    Returns:
        A set of names corresponding to the parameters found in domain expressions.
    """
    params = {str(sym.id) for sym in ir_params}
    return set(
        eve.walk_values(ir)
        .filter(lambda node: cpm.is_call_to(node, ("cartesian_domain", "unstructured_domain")))
        .map(
            lambda domain: eve.walk_values(domain)
            .if_isinstance(gtir.SymRef)
            .map(lambda symref: str(symref.id))
            .filter(lambda sym: sym in params)
            .to_list()
        )
        .reduce(operator.add, init=[])
    )


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
    global_symbols: dict[str, ts.DataType] = dataclasses.field(default_factory=dict)
    field_offsets: dict[str, Optional[list[dace.symbolic.SymExpr]]] = dataclasses.field(
        default_factory=dict
    )
    map_uids: eve.utils.UIDGenerator = dataclasses.field(
        init=False, repr=False, default_factory=lambda: eve.utils.UIDGenerator(prefix="map")
    )
    tasklet_uids: eve.utils.UIDGenerator = dataclasses.field(
        init=False, repr=False, default_factory=lambda: eve.utils.UIDGenerator(prefix="tlet")
    )

    def get_offset_provider_type(self, offset: str) -> gtx_common.OffsetProviderTypeElem:
        return self.offset_provider_type[offset]

    def make_field(
        self, data_node: dace.nodes.AccessNode, data_type: ts.FieldType | ts.ScalarType
    ) -> gtir_builtin_translators.FieldopData:
        """
        Helper method to build the field data type associated with an access node in the SDFG.

        In case of `ScalarType` data, the descriptor is constructed with `offset=None`.
        In case of `FieldType` data, the field origin is added to the data descriptor.
        Besides, if the `FieldType` contains a local dimension, the descriptor is converted
        to a canonical form where the field domain consists of all global dimensions
        (the grid axes) and the field data type is `ListType`, with `offset_type` equal
        to the field local dimension.

        Args:
            data_node: The access node to the SDFG data storage.
            data_type: The GT4Py data descriptor, which can either come from a field parameter
                of an expression node, or from an intermediate field in a previous expression.

        Returns:
            The descriptor associated with the SDFG data storage, filled with field origin.
        """
        if isinstance(data_type, ts.ScalarType):
            return gtir_builtin_translators.FieldopData(data_node, data_type, offset=None)
        domain_offset = self.field_offsets.get(data_node.data, None)
        local_dims = [dim for dim in data_type.dims if dim.kind == gtx_common.DimensionKind.LOCAL]
        if len(local_dims) == 0:
            # do nothing: the field domain consists of all global dimensions
            field_type = data_type
        elif len(local_dims) == 1:
            local_dim = local_dims[0]
            local_dim_index = data_type.dims.index(local_dim)
            # the local dimension is converted into `ListType` data element
            if not isinstance(data_type.dtype, ts.ScalarType):
                raise ValueError(f"Invalid field type {data_type}.")
            if local_dim_index != (len(data_type.dims) - 1):
                raise ValueError(
                    f"Invalid field domain: expected the local dimension to be at the end, found at position {local_dim_index}."
                )
            if local_dim.value not in self.offset_provider_type:
                raise ValueError(
                    f"The provided local dimension {local_dim} does not match any offset provider type."
                )
            local_type = ts.ListType(element_type=data_type.dtype, offset_type=local_dim)
            field_type = ts.FieldType(dims=data_type.dims[:local_dim_index], dtype=local_type)
        else:
            raise NotImplementedError(
                "Fields with more than one local dimension are not supported."
            )
        return gtir_builtin_translators.FieldopData(data_node, field_type, domain_offset)

    def get_symbol_type(self, symbol_name: str) -> ts.DataType:
        return self.global_symbols[symbol_name]

    def is_column_axis(self, dim: gtx_common.Dimension) -> bool:
        assert self.column_axis
        return dim == self.column_axis

    def setup_nested_context(
        self,
        expr: gtir.Expr,
        sdfg: dace.SDFG,
        global_symbols: dict[str, ts.DataType],
        field_offsets: dict[str, Optional[list[dace.symbolic.SymExpr]]],
    ) -> SDFGBuilder:
        nsdfg_builder = GTIRToSDFG(
            self.offset_provider_type, self.column_axis, global_symbols, field_offsets
        )
        nsdfg_params = [
            gtir.Sym(id=p_name, type=p_type) for p_name, p_type in global_symbols.items()
        ]
        domain_symbols = _collect_symbols_in_domain_expressions(expr, nsdfg_params)
        nsdfg_builder._add_sdfg_params(
            sdfg, node_params=nsdfg_params, symbolic_arguments=domain_symbols
        )
        return nsdfg_builder

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
    ) -> tuple[list[dace.symbol], list[dace.symbol]]:
        """
        Parse field dimensions and allocate symbols for array shape and strides.

        For local dimensions, the size is known at compile-time and therefore
        the corresponding array shape dimension is set to an integer literal value.

        Returns:
            Two lists of symbols, one for the shape and the other for the strides of the array.
        """
        dc_dtype = gtir_builtin_translators.INDEX_DTYPE
        neighbor_table_types = gtx_dace_utils.filter_connectivity_types(self.offset_provider_type)
        shape = [
            (
                neighbor_table_types[dim.value].max_neighbors
                if dim.kind == gtx_common.DimensionKind.LOCAL
                else dace.symbol(gtx_dace_utils.field_size_symbol_name(name, i), dc_dtype)
            )
            for i, dim in enumerate(dims)
        ]
        strides = [
            dace.symbol(gtx_dace_utils.field_stride_symbol_name(name, i), dc_dtype)
            for i in range(len(dims))
        ]
        return shape, strides

    def _add_storage(
        self,
        sdfg: dace.SDFG,
        symbolic_arguments: set[str],
        name: str,
        gt_type: ts.DataType,
        transient: bool = True,
    ) -> list[tuple[str, ts.DataType]]:
        """
        Add storage in the SDFG for a given GT4Py data symbol.

        GT4Py fields are allocated as DaCe arrays. GT4Py scalars are represented
        as DaCe scalar objects in the SDFG; the exception are the symbols passed as
        `symbolic_arguments`, e.g. symbols used in domain expressions, and those used
        for symbolic array shape and strides.

        The fields used as temporary arrays, when `transient = True`, are allocated
        and exist only within the SDFG; when `transient = False`, the fields have
        to be allocated outside and have to be passed as arguments to the SDFG call.

        Args:
            sdfg: The SDFG where storage needs to be allocated.
            symbolic_arguments: Set of GT4Py scalars that must be represented as SDFG symbols.
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
            for sym in gtir_sdfg_utils.flatten_tuple_fields(name, gt_type):
                assert isinstance(sym.type, ts.DataType)
                tuple_fields.extend(
                    self._add_storage(sdfg, symbolic_arguments, sym.id, sym.type, transient)
                )
            return tuple_fields

        elif isinstance(gt_type, ts.FieldType):
            if len(gt_type.dims) == 0:
                # represent zero-dimensional fields as scalar arguments
                return self._add_storage(sdfg, symbolic_arguments, name, gt_type.dtype, transient)
            if not isinstance(gt_type.dtype, ts.ScalarType):
                raise ValueError(f"Field type '{gt_type.dtype}' not supported.")
            # handle default case: field with one or more dimensions
            dc_dtype = gtx_dace_utils.as_dace_type(gt_type.dtype)
            # Use symbolic shape, which allows to invoke the program with fields of different size;
            # and symbolic strides, which enables decoupling the memory layout from generated code.
            sym_shape, sym_strides = self._make_array_shape_and_strides(name, gt_type.dims)
            sdfg.add_array(name, sym_shape, dc_dtype, strides=sym_strides, transient=transient)
            return [(name, gt_type)]

        elif isinstance(gt_type, ts.ScalarType):
            dc_dtype = gtx_dace_utils.as_dace_type(gt_type)
            if gtx_dace_utils.is_field_symbol(name) or name in symbolic_arguments:
                if name in sdfg.symbols:
                    # Sometimes, when the field domain is implicitly derived from the
                    # field domain, the gt4py lowering adds the field size as a scalar
                    # argument to the program IR. Suppose a field '__sym', then gt4py
                    # will add '__sym_size_0'.
                    # Therefore, here we check whether the shape symbol was already
                    # created by `_make_array_shape_and_strides()`, when allocating
                    # storage for field arguments. We assume that the scalar argument
                    # for field size, if present, always follows the field argument.
                    assert gtx_dace_utils.is_field_symbol(name)
                    if sdfg.symbols[name].dtype != dc_dtype:
                        raise ValueError(
                            f"Type mismatch on argument {name}: got {dc_dtype}, expected {sdfg.symbols[name].dtype}."
                        )
                else:
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
        self, node: gtir.Expr, sdfg: dace.SDFG, head_state: dace.SDFGState, use_temp: bool = True
    ) -> list[gtir_builtin_translators.FieldopData]:
        """
        Specialized visit method for fieldview expressions.

        This method represents the entry point to visit `ir.Stmt` expressions.
        As such, it must preserve the property of single exit state in the SDFG.

        Returns:
            A list of array nodes containing the result fields.
        """
        result = self.visit(node, sdfg=sdfg, head_state=head_state)

        # sanity check: each statement should preserve the property of single exit state (aka head state),
        # i.e. eventually only introduce internal branches, and keep the same head state
        sink_states = sdfg.sink_nodes()
        assert len(sink_states) == 1
        assert sink_states[0] == head_state

        def make_temps(
            field: gtir_builtin_translators.FieldopData,
        ) -> gtir_builtin_translators.FieldopData:
            desc = sdfg.arrays[field.dc_node.data]
            if desc.transient or not use_temp:
                return field
            else:
                temp, _ = self.add_temp_array_like(sdfg, desc)
                temp_node = head_state.add_access(temp)
                head_state.add_nedge(
                    field.dc_node, temp_node, sdfg.make_array_memlet(field.dc_node.data)
                )
                return field.make_copy(temp_node)

        temp_result = gtx_utils.tree_map(make_temps)(result)
        return list(gtx_utils.flatten_nested_tuple((temp_result,)))

    def _add_sdfg_params(
        self,
        sdfg: dace.SDFG,
        node_params: Sequence[gtir.Sym],
        symbolic_arguments: set[str],
    ) -> list[str]:
        """
        Helper function to add storage for node parameters and connectivity tables.

        GT4Py field arguments will be translated to `dace.data.Array` objects.
        GT4Py scalar arguments will be translated to `dace.data.Scalar` objects,
        except when they are listed in 'symbolic_arguments', in which case they
        will be represented in the SDFG as DaCe symbols.
        """

        # add non-transient arrays and/or SDFG symbols for the program arguments
        sdfg_args = []
        for param in node_params:
            pname = str(param.id)
            assert isinstance(param.type, (ts.DataType))
            sdfg_args += self._add_storage(
                sdfg, symbolic_arguments, pname, param.type, transient=False
            )
            self.global_symbols[pname] = param.type

        # add SDFG storage for connectivity tables
        for offset, connectivity_type in gtx_dace_utils.filter_connectivity_types(
            self.offset_provider_type
        ).items():
            scalar_type = tt.from_dtype(connectivity_type.dtype)
            gt_type = ts.FieldType(
                [connectivity_type.source_dim, connectivity_type.neighbor_dim], scalar_type
            )
            # We store all connectivity tables as transient arrays here; later, while building
            # the field operator expressions, we change to non-transient (i.e. allocated externally)
            # the tables that are actually used. This way, we avoid adding SDFG arguments for
            # the connectivity tables that are not used. The remaining unused transient arrays
            # are removed by the dace simplify pass.
            self._add_storage(
                sdfg,
                symbolic_arguments,
                gtx_dace_utils.connectivity_identifier(offset),
                gt_type,
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
        if node.function_definitions:
            raise NotImplementedError("Functions expected to be inlined as lambda calls.")

        # Since program field arguments are passed to the SDFG as full-shape arrays,
        # there is no offset that needs to be compensated.
        assert len(self.field_offsets) == 0

        sdfg = dace.SDFG(node.id)
        sdfg.debuginfo = gtir_sdfg_utils.debug_info(node)

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

        domain_symbols = _collect_symbols_in_domain_expressions(node, node.params)
        sdfg_arg_names = self._add_sdfg_params(sdfg, node.params, symbolic_arguments=domain_symbols)

        # visit one statement at a time and expand the SDFG from the current head state
        for i, stmt in enumerate(node.body):
            # include `debuginfo` only for `ir.Program` and `ir.Stmt` nodes: finer granularity would be too messy
            head_state = sdfg.add_state_after(head_state, f"stmt_{i}")
            head_state._debuginfo = gtir_sdfg_utils.debug_info(stmt, default=sdfg.debuginfo)
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

        source_fields = self._visit_expression(stmt.expr, sdfg, state)

        # the target expression could be a `SymRef` to an output node or a `make_tuple` expression
        # in case the statement returns more than one field
        target_fields = self._visit_expression(stmt.target, sdfg, state, use_temp=False)

        # convert domain expression to dictionary to ease access to dimension boundaries
        domain = {
            dim: (lb, ub) for dim, lb, ub in gtir_builtin_translators.extract_domain(stmt.domain)
        }

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

        target_state: Optional[dace.SDFGState] = None
        for source, target in zip(source_fields, target_fields, strict=True):
            target_desc = sdfg.arrays[target.dc_node.data]
            assert not target_desc.transient

            if isinstance(target.gt_type, ts.FieldType):
                target_subset = ",".join(
                    f"{domain[dim][0]}:{domain[dim][1]}" for dim in target.gt_type.dims
                )
                source_subset = (
                    target_subset
                    if source.offset is None
                    else ",".join(
                        f"{domain[dim][0] - offset}:{domain[dim][1] - offset}"
                        for dim, offset in zip(target.gt_type.dims, source.offset, strict=True)
                    )
                )
            else:
                assert len(domain) == 0
                target_subset = "0"
                source_subset = "0"

            if target.dc_node.data in state_input_data:
                # if inout argument, write the result in separate next state
                # this is needed to avoid undefined behavior for expressions like: X, Y = X + 1, X
                if not target_state:
                    target_state = sdfg.add_state_after(state, f"post_{state.label}")
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

        return target_state or state

    def visit_FunCall(
        self,
        node: gtir.FunCall,
        sdfg: dace.SDFG,
        head_state: dace.SDFGState,
    ) -> gtir_builtin_translators.FieldopResult:
        # use specialized dataflow builder classes for each builtin function
        if cpm.is_call_to(node, "if_"):
            return gtir_builtin_translators.translate_if(node, sdfg, head_state, self)
        elif cpm.is_call_to(node, "index"):
            return gtir_builtin_translators.translate_index(node, sdfg, head_state, self)
        elif cpm.is_call_to(node, "make_tuple"):
            return gtir_builtin_translators.translate_make_tuple(node, sdfg, head_state, self)
        elif cpm.is_call_to(node, "tuple_get"):
            return gtir_builtin_translators.translate_tuple_get(node, sdfg, head_state, self)
        elif cpm.is_applied_as_fieldop(node):
            return gtir_builtin_translators.translate_as_fieldop(node, sdfg, head_state, self)
        elif isinstance(node.fun, gtir.Lambda):
            lambda_args = [
                self.visit(
                    arg,
                    sdfg=sdfg,
                    head_state=head_state,
                )
                for arg in node.args
            ]

            return self.visit(
                node.fun,
                sdfg=sdfg,
                head_state=head_state,
                args=lambda_args,
            )
        elif isinstance(node.type, ts.ScalarType):
            return gtir_builtin_translators.translate_scalar_expr(node, sdfg, head_state, self)
        else:
            raise NotImplementedError(f"Unexpected 'FunCall' expression ({node}).")

    def visit_Lambda(
        self,
        node: gtir.Lambda,
        sdfg: dace.SDFG,
        head_state: dace.SDFGState,
        args: Sequence[gtir_builtin_translators.FieldopResult],
    ) -> gtir_builtin_translators.FieldopResult:
        """
        Translates a `Lambda` node to a nested SDFG in the current state.

        All arguments to lambda functions are fields (i.e. `as_fieldop`, field or scalar `gtir.SymRef`,
        nested let-lambdas thereof). The reason for creating a nested SDFG is to define local symbols
        (the lambda paremeters) that map to parent fields, either program arguments or temporary fields.

        If the lambda has a parameter whose name is already present in `GTIRToSDFG.global_symbols`,
        i.e. a lambda parameter with the same name as a symbol in scope, the parameter will shadow
        the previous symbol during traversal of the lambda expression.
        """
        lambda_args_mapping = [
            (str(param.id), arg) for param, arg in zip(node.params, args, strict=True)
        ]

        lambda_arg_nodes = dict(
            itertools.chain(
                *[
                    gtir_builtin_translators.flatten_tuples(pname, arg)
                    for pname, arg in lambda_args_mapping
                ]
            )
        )

        # inherit symbols from parent scope but eventually override with local symbols
        lambda_symbols = {
            sym: self.global_symbols[sym]
            for sym in symbol_ref_utils.collect_symbol_refs(node.expr, self.global_symbols.keys())
        } | {
            pname: gtir_builtin_translators.get_tuple_type(arg)
            if isinstance(arg, tuple)
            else arg.gt_type
            for pname, arg in lambda_args_mapping
        }

        def get_field_domain_offset(
            p_name: str, p_type: ts.DataType
        ) -> dict[str, Optional[list[dace.symbolic.SymExpr]]]:
            if isinstance(p_type, ts.FieldType):
                if p_name in lambda_arg_nodes:
                    arg = lambda_arg_nodes[p_name]
                    assert isinstance(arg, gtir_builtin_translators.FieldopData)
                    return {p_name: arg.offset}
                elif field_domain_offset := self.field_offsets.get(p_name, None):
                    return {p_name: field_domain_offset}
            elif isinstance(p_type, ts.TupleType):
                tsyms = gtir_sdfg_utils.flatten_tuple_fields(p_name, p_type)
                return functools.reduce(
                    lambda field_offsets, sym: (
                        field_offsets | get_field_domain_offset(sym.id, sym.type)  # type: ignore[arg-type]
                    ),
                    tsyms,
                    {},
                )
            return {}

        # populate mapping from field name to domain offset
        lambda_field_offsets: dict[str, Optional[list[dace.symbolic.SymExpr]]] = {}
        for p_name, p_type in lambda_symbols.items():
            lambda_field_offsets |= get_field_domain_offset(p_name, p_type)

        # lower let-statement lambda node as a nested SDFG
        nsdfg = dace.SDFG(name=self.unique_nsdfg_name(sdfg, "lambda"))
        nsdfg.debuginfo = gtir_sdfg_utils.debug_info(node, default=sdfg.debuginfo)
        lambda_translator = self.setup_nested_context(
            node.expr, nsdfg, lambda_symbols, lambda_field_offsets
        )

        nstate = nsdfg.add_state("lambda")
        lambda_result = lambda_translator.visit(
            node.expr,
            sdfg=nsdfg,
            head_state=nstate,
        )

        # Process lambda inputs
        #
        # All input arguments are passed as parameters to the nested SDFG, therefore
        # we they are stored as non-transient array and scalar objects.
        #
        connectivity_arrays = {
            gtx_dace_utils.connectivity_identifier(offset)
            for offset in gtx_dace_utils.filter_connectivity_types(self.offset_provider_type)
        }

        input_memlets = {}
        nsdfg_symbols_mapping = {str(sym): sym for sym in nsdfg.free_symbols}
        for nsdfg_dataname, nsdfg_datadesc in nsdfg.arrays.items():
            if nsdfg_datadesc.transient:
                continue

            if nsdfg_dataname in lambda_arg_nodes:
                src_node = lambda_arg_nodes[nsdfg_dataname].dc_node
                dataname = src_node.data
                datadesc = src_node.desc(sdfg)
                nsdfg_symbols_mapping |= {
                    str(nested_symbol): parent_symbol
                    for nested_symbol, parent_symbol in zip(
                        [*nsdfg_datadesc.shape, *nsdfg_datadesc.strides],
                        [*datadesc.shape, *datadesc.strides],
                        strict=True,
                    )
                    if dace.symbolic.issymbolic(nested_symbol)
                }
            else:
                dataname = nsdfg_dataname
                datadesc = sdfg.arrays[nsdfg_dataname]

            # ensure that connectivity tables are non-transient arrays in parent SDFG
            if dataname in connectivity_arrays:
                datadesc.transient = False

            input_memlets[nsdfg_dataname] = sdfg.make_array_memlet(dataname)

        # Process lambda outputs
        #
        # The output arguments do not really exist, so they are not allocated before
        # visiting the lambda expression. Therefore, the result appears inside the
        # nested SDFG as transient array/scalar storage. The exception is given by
        # input arguments that are just passed through and returned by the lambda,
        # e.g. when the lambda is constructing a tuple: in this case, the result
        # data is non-transient, because it corresponds to an input node.
        # The transient storage of the lambda result in nested-SDFG is corrected
        # below by the call to `make_temps()`: this function ensures that the result
        # transient nodes are changed to non-transient and the corresponding output
        # connecters on the nested SDFG are connected to new data nodes in parent SDFG.
        #
        lambda_output_data: Iterable[gtir_builtin_translators.FieldopData] = (
            gtx_utils.flatten_nested_tuple(lambda_result)
        )
        # The output connectors only need to be setup for the actual result of the
        # internal dataflow that writes to transient nodes.
        # We filter out the non-transient nodes because they are already available
        # in the current context. Later these nodes will eventually be removed
        # from the nested SDFG because they are isolated (see `make_temps()`).
        lambda_outputs = {
            output_data.dc_node.data
            for output_data in lambda_output_data
            if output_data.dc_node.desc(nsdfg).transient
        }

        nsdfg_node = head_state.add_nested_sdfg(
            nsdfg,
            parent=sdfg,
            inputs=set(input_memlets.keys()),
            outputs=lambda_outputs,
            symbol_mapping=nsdfg_symbols_mapping,
            debuginfo=gtir_sdfg_utils.debug_info(node, default=sdfg.debuginfo),
        )

        for connector, memlet in input_memlets.items():
            if connector in lambda_arg_nodes:
                src_node = lambda_arg_nodes[connector].dc_node
            else:
                src_node = head_state.add_access(memlet.data)

            head_state.add_edge(src_node, None, nsdfg_node, connector, memlet)

        def construct_output_for_nested_sdfg(
            inner_data: gtir_builtin_translators.FieldopData,
        ) -> gtir_builtin_translators.FieldopData:
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
            inner_desc = inner_data.dc_node.desc(nsdfg)
            if inner_desc.transient:
                # Transient data nodes only exist within the nested SDFG. In order to return some result data,
                # the corresponding data container inside the nested SDFG has to be changed to non-transient,
                # that is externally allocated, as required by the SDFG IR. An output edge will write the result
                # from the nested-SDFG to a new intermediate data container allocated in the parent SDFG.
                inner_desc.transient = False
                outer, outer_desc = self.add_temp_array_like(sdfg, inner_desc)
                # We cannot use a copy of the inner data descriptor directly, we have to apply the symbol mapping.
                dace.symbolic.safe_replace(
                    nsdfg_symbols_mapping,
                    lambda m: dace.sdfg.replace_properties_dict(outer_desc, m),
                )
                connector = inner_data.dc_node.data
                outer_node = head_state.add_access(outer)
                head_state.add_edge(
                    nsdfg_node, connector, outer_node, None, sdfg.make_array_memlet(outer)
                )
                outer_data = inner_data.make_copy(outer_node)
            elif inner_data.dc_node.data in lambda_arg_nodes:
                # This if branch and the next one handle the non-transient result nodes.
                # Non-transient nodes are just input nodes that are immediately returned
                # by the lambda expression. Therefore, these nodes are already available
                # in the parent context and can be directly accessed there.
                outer_data = lambda_arg_nodes[inner_data.dc_node.data]
            else:
                outer_node = head_state.add_access(inner_data.dc_node.data)
                outer_data = inner_data.make_copy(outer_node)
            # Isolated access node will make validation fail.
            # Isolated access nodes can be found in the join-state of an if-expression
            # or in lambda expressions that just construct tuples from input arguments.
            if nstate.degree(inner_data.dc_node) == 0:
                nstate.remove_node(inner_data.dc_node)
            return outer_data

        return gtx_utils.tree_map(construct_output_for_nested_sdfg)(lambda_result)

    def visit_Literal(
        self,
        node: gtir.Literal,
        sdfg: dace.SDFG,
        head_state: dace.SDFGState,
    ) -> gtir_builtin_translators.FieldopResult:
        return gtir_builtin_translators.translate_literal(node, sdfg, head_state, self)

    def visit_SymRef(
        self,
        node: gtir.SymRef,
        sdfg: dace.SDFG,
        head_state: dace.SDFGState,
    ) -> gtir_builtin_translators.FieldopResult:
        return gtir_builtin_translators.translate_symbol_ref(node, sdfg, head_state, self)


def build_sdfg_from_gtir(
    ir: gtir.Program,
    offset_provider_type: gtx_common.OffsetProviderType,
    column_axis: Optional[gtx_common.Dimension] = None,
) -> dace.SDFG:
    """
    Receives a GTIR program and lowers it to a DaCe SDFG.

    The lowering to SDFG requires that the program node is type-annotated, therefore this function
    runs type ineference as first step.

    Args:
        ir: The GTIR program node to be lowered to SDFG
        offset_provider_type: The definitions of offset providers used by the program node
        column_axis: Vertical dimension used for column scan expressions.

    Returns:
        An SDFG in the DaCe canonical form (simplified)
    """

    ir = gtir_type_inference.infer(ir, offset_provider_type=offset_provider_type)
    ir = ir_prune_casts.PruneCasts().visit(ir)

    # DaCe requires C-compatible strings for the names of data containers,
    # such as arrays and scalars. GT4Py uses a unicode symbols ('ᐞ') as name
    # separator in the SSA pass, which generates invalid symbols for DaCe.
    # Here we find new names for invalid symbols present in the IR.
    ir = gtir_sdfg_utils.replace_invalid_symbols(ir)

    sdfg_genenerator = GTIRToSDFG(offset_provider_type, column_axis)
    sdfg = sdfg_genenerator.visit(ir)
    assert isinstance(sdfg, dace.SDFG)

    # TODO(edopao): remove inlining when DaCe transformations support LoopRegion construct
    dace_sdfg_utils.inline_loop_blocks(sdfg)

    return sdfg
