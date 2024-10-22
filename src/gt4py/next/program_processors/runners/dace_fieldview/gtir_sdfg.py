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
import itertools
import operator
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple, Union

import dace

from gt4py import eve
from gt4py.eve import concepts
from gt4py.next import common as gtx_common, utils as gtx_utils
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.iterator.transforms import prune_casts as ir_prune_casts, symbol_ref_utils
from gt4py.next.iterator.type_system import inference as gtir_type_inference
from gt4py.next.program_processors.runners.dace_common import utility as dace_utils
from gt4py.next.program_processors.runners.dace_fieldview import (
    gtir_builtin_translators,
    gtir_dataflow,
    utility as dace_gtir_utils,
)
from gt4py.next.type_system import type_specifications as ts, type_translation as tt


class DataflowBuilder(Protocol):
    """Visitor interface to build a dataflow subgraph."""

    @abc.abstractmethod
    def get_offset_provider(self, offset: str) -> gtx_common.OffsetProviderElem: ...

    @abc.abstractmethod
    def unique_map_name(self, name: str) -> str: ...

    @abc.abstractmethod
    def unique_tasklet_name(self, name: str) -> str: ...

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
        inputs: Union[Set[str], Dict[str, dace.dtypes.typeclass]],
        code: str,
        outputs: Union[Set[str], Dict[str, dace.dtypes.typeclass]],
        **kwargs: Any,
    ) -> tuple[dace.nodes.Tasklet, dace.nodes.MapEntry, dace.nodes.MapExit]:
        """Wrapper of `dace.SDFGState.add_mapped_tasklet` that assigns unique name."""
        unique_name = self.unique_tasklet_name(name)
        return state.add_mapped_tasklet(unique_name, map_ranges, inputs, code, outputs, **kwargs)


class SDFGBuilder(DataflowBuilder, Protocol):
    """Visitor interface available to GTIR-primitive translators."""

    @abc.abstractmethod
    def get_symbol_type(self, symbol_name: str) -> ts.DataType:
        """Retrieve the GT4Py type of a symbol used in the program."""
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

    offset_provider: gtx_common.OffsetProvider
    global_symbols: dict[str, ts.DataType] = dataclasses.field(default_factory=lambda: {})
    map_uids: eve.utils.UIDGenerator = dataclasses.field(
        init=False, repr=False, default_factory=lambda: eve.utils.UIDGenerator(prefix="map")
    )
    tasklet_uids: eve.utils.UIDGenerator = dataclasses.field(
        init=False, repr=False, default_factory=lambda: eve.utils.UIDGenerator(prefix="tlet")
    )

    def get_offset_provider(self, offset: str) -> gtx_common.OffsetProviderElem:
        return self.offset_provider[offset]

    def get_symbol_type(self, symbol_name: str) -> ts.DataType:
        return self.global_symbols[symbol_name]

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
        dtype = dace.int32
        neighbor_tables = dace_utils.filter_connectivities(self.offset_provider)
        shape = [
            (
                neighbor_tables[dim.value].max_neighbors
                if dim.kind == gtx_common.DimensionKind.LOCAL
                else dace.symbol(dace_utils.field_size_symbol_name(name, i), dtype)
            )
            for i, dim in enumerate(dims)
        ]
        strides = [
            dace.symbol(dace_utils.field_stride_symbol_name(name, i), dtype)
            for i in range(len(dims))
        ]
        return shape, strides

    def _add_storage(
        self,
        sdfg: dace.SDFG,
        known_symbols: set[str],
        name: str,
        gt_type: ts.DataType,
        transient: bool = True,
    ) -> list[tuple[str, ts.DataType]]:
        """
        Add storage in the SDFG for a given GT4Py data symbol.

        GT4Py fields are allocated as dace arrays. GT4Py scalars are represented
        as scalar objects in the SDFG; the exception are the symbols passed as
        `known_symbols`, e.g. symbols used in domain expressions, and those used
        for symbolic array shape and strides.

        The fields used as temporary arrays, when `transient = True`, are allocated
        and exist only within the SDFG; when `transient = False`, the fields have
        to be allocated outside and have to be passed as arguments to the SDFG call.

        Args:
            sdfg: The SDFG where storage needs to be allocated.
            known_symbols: Set of GT4Py scalars that must be represented as SDFG symbols.
            name: Symbol Name to be allocated.
            gt_type: GT4Py symbol type.
            transient: True when the data symbol has to be allocated as internal storage.

        Returns:
            List of data containers or symbols allocated as storage. This is a list,
            not a single value, because in case of tuples we flat the tuple fields
            (eventually nested) and allocate storage for each tuple element.
        """
        if isinstance(gt_type, ts.TupleType):
            tuple_fields = []
            for tname, tsymbol_type in dace_gtir_utils.get_tuple_fields(
                name, gt_type, flatten=True
            ):
                tuple_fields.extend(
                    self._add_storage(sdfg, known_symbols, tname, tsymbol_type, transient)
                )
            return tuple_fields

        elif isinstance(gt_type, ts.FieldType):
            dtype = dace_utils.as_dace_type(gt_type.dtype)
            # use symbolic shape, which allows to invoke the program with fields of different size;
            # and symbolic strides, which enables decoupling the memory layout from generated code.
            sym_shape, sym_strides = self._make_array_shape_and_strides(name, gt_type.dims)
            sdfg.add_array(name, sym_shape, dtype, strides=sym_strides, transient=transient)

            return [(name, gt_type)]

        elif isinstance(gt_type, ts.ScalarType):
            dtype = dace_utils.as_dace_type(gt_type)
            if name in known_symbols:
                sdfg.add_symbol(name, dtype)
            elif dace_utils.is_field_symbol(name):
                # Sometimes the IR contains the field size as a scalar program argument,
                # so we have to check if the shape symbol was already allocated by
                # `_make_array_shape_and_strides`. We assume that the scalar argument
                # for field size always follows the field argument.
                if name in sdfg.symbols:
                    assert sdfg.symbols[name].dtype == dtype
                else:
                    sdfg.add_symbol(name, dtype)
            else:
                sdfg.add_scalar(name, dtype, transient=transient)

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
    ) -> list[gtir_builtin_translators.Field]:
        """
        Specialized visit method for fieldview expressions.

        This method represents the entry point to visit `ir.Stmt` expressions.
        As such, it must preserve the property of single exit state in the SDFG.

        Returns:
            A list of array nodes containing the result fields.
        """
        result = self.visit(node, sdfg=sdfg, head_state=head_state, reduce_identity=None)

        # sanity check: each statement should preserve the property of single exit state (aka head state),
        # i.e. eventually only introduce internal branches, and keep the same head state
        sink_states = sdfg.sink_nodes()
        assert len(sink_states) == 1
        assert sink_states[0] == head_state

        def make_temps(field: gtir_builtin_translators.Field) -> gtir_builtin_translators.Field:
            desc = sdfg.arrays[field.data_node.data]
            if desc.transient or not use_temp:
                return field
            else:
                temp, _ = sdfg.add_temp_transient_like(desc)
                temp_node = head_state.add_access(temp)
                head_state.add_nedge(
                    field.data_node, temp_node, sdfg.make_array_memlet(field.data_node.data)
                )
                return gtir_builtin_translators.Field(temp_node, field.data_type)

        temp_result = gtx_utils.tree_map(make_temps)(result)
        return list(gtx_utils.flatten_nested_tuple((temp_result,)))

    def _add_sdfg_params(
        self,
        sdfg: dace.SDFG,
        node_params: Sequence[gtir.Sym],
        known_symbols: set[str],
    ) -> list[str]:
        """
        Helper function to add storage for node parameters and connectivity tables.

        By default all scalar arguments are represented as `dace.data.Scalar` objects.
        However, some arguments that are related to domain size and array layout
        (shape and strides) must be represented as SDFG symbols (`dace.Symbol` objects).
        The names passed as `known_symbols` will be represented as SDFG symbols.
        """
        # add non-transient arrays and/or SDFG symbols for the program arguments
        sdfg_args = []
        for param in node_params:
            pname = str(param.id)
            assert isinstance(param.type, (ts.DataType))
            sdfg_args += self._add_storage(sdfg, known_symbols, pname, param.type, transient=False)
            self.global_symbols[pname] = param.type

        # add SDFG storage for connectivity tables
        for offset, offset_provider in dace_utils.filter_connectivities(
            self.offset_provider
        ).items():
            scalar_kind = tt.get_scalar_kind(offset_provider.index_type)
            local_dim = gtx_common.Dimension(offset, kind=gtx_common.DimensionKind.LOCAL)
            gt_type = ts.FieldType(
                [offset_provider.origin_axis, local_dim], ts.ScalarType(scalar_kind)
            )
            # We store all connectivity tables as transient arrays here; later, while building
            # the field operator expressions, we change to non-transient (i.e. allocated externally)
            # the tables that are actually used. This way, we avoid adding SDFG arguments for
            # the connectivity tables that are not used. The remaining unused transient arrays
            # are removed by the dace simplify pass.
            self._add_storage(
                sdfg, known_symbols, dace_utils.connectivity_identifier(offset), gt_type
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

        sdfg = dace.SDFG(node.id)
        sdfg.debuginfo = dace_utils.debug_info(node, default=sdfg.debuginfo)
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
        sdfg_arg_names = self._add_sdfg_params(sdfg, node.params, known_symbols=domain_symbols)

        # visit one statement at a time and expand the SDFG from the current head state
        for i, stmt in enumerate(node.body):
            # include `debuginfo` only for `ir.Program` and `ir.Stmt` nodes: finer granularity would be too messy
            head_state = sdfg.add_state_after(head_state, f"stmt_{i}")
            head_state._debuginfo = dace_utils.debug_info(stmt, default=sdfg.debuginfo)
            head_state = self.visit(stmt, sdfg=sdfg, state=head_state)

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

        temp_fields = self._visit_expression(stmt.expr, sdfg, state)

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
        for temp, target in zip(temp_fields, target_fields, strict=True):
            target_desc = sdfg.arrays[target.data_node.data]
            assert not target_desc.transient

            if isinstance(target.data_type, ts.FieldType):
                subset = ",".join(
                    f"{domain[dim][0]}:{domain[dim][1]}" for dim in target.data_type.dims
                )
            else:
                assert len(domain) == 0
                subset = "0"

            if target.data_node.data in state_input_data:
                # if inout argument, write the result in separate next state
                # this is needed to avoid undefined behavior for expressions like: X, Y = X + 1, X
                if not target_state:
                    target_state = sdfg.add_state_after(state, f"post_{state.label}")
                # create new access nodes in the target state
                target_state.add_nedge(
                    target_state.add_access(temp.data_node.data),
                    target_state.add_access(target.data_node.data),
                    dace.Memlet(data=target.data_node.data, subset=subset, other_subset=subset),
                )
                # remove isolated access node
                state.remove_node(target.data_node)
            else:
                state.add_nedge(
                    temp.data_node,
                    target.data_node,
                    dace.Memlet(data=target.data_node.data, subset=subset, other_subset=subset),
                )

        return target_state or state

    def visit_FunCall(
        self,
        node: gtir.FunCall,
        sdfg: dace.SDFG,
        head_state: dace.SDFGState,
        reduce_identity: Optional[gtir_dataflow.SymbolExpr],
    ) -> gtir_builtin_translators.FieldopResult:
        # use specialized dataflow builder classes for each builtin function
        if cpm.is_call_to(node, "if_"):
            return gtir_builtin_translators.translate_if(
                node, sdfg, head_state, self, reduce_identity
            )
        elif cpm.is_call_to(node, "make_tuple"):
            return gtir_builtin_translators.translate_make_tuple(
                node, sdfg, head_state, self, reduce_identity
            )
        elif cpm.is_call_to(node, "tuple_get"):
            return gtir_builtin_translators.translate_tuple_get(
                node, sdfg, head_state, self, reduce_identity
            )
        elif cpm.is_applied_as_fieldop(node):
            return gtir_builtin_translators.translate_as_fieldop(
                node, sdfg, head_state, self, reduce_identity
            )
        elif isinstance(node.fun, gtir.Lambda):
            lambda_args = [
                self.visit(
                    arg,
                    sdfg=sdfg,
                    head_state=head_state,
                    reduce_identity=reduce_identity,
                )
                for arg in node.args
            ]

            return self.visit(
                node.fun,
                sdfg=sdfg,
                head_state=head_state,
                reduce_identity=reduce_identity,
                args=lambda_args,
            )
        elif isinstance(node.type, ts.ScalarType):
            return gtir_builtin_translators.translate_scalar_expr(
                node, sdfg, head_state, self, reduce_identity
            )
        else:
            raise NotImplementedError(f"Unexpected 'FunCall' expression ({node}).")

    def visit_Lambda(
        self,
        node: gtir.Lambda,
        sdfg: dace.SDFG,
        head_state: dace.SDFGState,
        reduce_identity: Optional[gtir_dataflow.SymbolExpr],
        args: list[gtir_builtin_translators.FieldopResult],
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

        # inherit symbols from parent scope but eventually override with local symbols
        lambda_symbols = {
            sym: self.global_symbols[sym]
            for sym in symbol_ref_utils.collect_symbol_refs(node.expr, self.global_symbols.keys())
        } | {
            pname: dace_gtir_utils.get_tuple_type(arg) if isinstance(arg, tuple) else arg.data_type
            for pname, arg in lambda_args_mapping
        }

        # lower let-statement lambda node as a nested SDFG
        lambda_translator = GTIRToSDFG(self.offset_provider, lambda_symbols)
        nsdfg = dace.SDFG(f"{sdfg.label}_lambda")
        nstate = nsdfg.add_state("lambda")

        # add sdfg storage for the symbols that need to be passed as input parameters
        lambda_params = [
            gtir.Sym(id=p_name, type=p_type) for p_name, p_type in lambda_symbols.items()
        ]
        lambda_domain_symbols = _collect_symbols_in_domain_expressions(node, lambda_params)
        lambda_translator._add_sdfg_params(
            nsdfg, node_params=lambda_params, known_symbols=lambda_domain_symbols
        )

        lambda_result = lambda_translator.visit(
            node.expr,
            sdfg=nsdfg,
            head_state=nstate,
            reduce_identity=reduce_identity,
        )

        def _flatten_tuples(
            name: str,
            arg: gtir_builtin_translators.FieldopResult,
        ) -> list[tuple[str, gtir_builtin_translators.Field]]:
            if isinstance(arg, tuple):
                tuple_type = dace_gtir_utils.get_tuple_type(arg)
                tuple_field_names = [
                    arg_name for arg_name, _ in dace_gtir_utils.get_tuple_fields(name, tuple_type)
                ]
                tuple_args = zip(tuple_field_names, arg, strict=True)
                return list(
                    itertools.chain(*[_flatten_tuples(fname, farg) for fname, farg in tuple_args])
                )
            else:
                return [(name, arg)]

        # Process lambda inputs
        #
        lambda_arg_nodes = dict(
            itertools.chain(*[_flatten_tuples(pname, arg) for pname, arg in lambda_args_mapping])
        )
        connectivity_arrays = {
            dace_utils.connectivity_identifier(offset)
            for offset in dace_utils.filter_connectivities(self.offset_provider)
        }

        input_memlets = {}
        nsdfg_symbols_mapping: dict[str, dace.symbolic.SymExpr] = {}
        for nsdfg_dataname, nsdfg_datadesc in nsdfg.arrays.items():
            if nsdfg_datadesc.transient:
                continue
            datadesc: Optional[dace.dtypes.Data] = None
            if nsdfg_dataname in lambda_arg_nodes:
                src_node = lambda_arg_nodes[nsdfg_dataname].data_node
                dataname = src_node.data
                datadesc = src_node.desc(sdfg)
            else:
                dataname = nsdfg_dataname
                datadesc = sdfg.arrays[nsdfg_dataname]

            # ensure that connectivity tables are non-transient arrays in parent SDFG
            if dataname in connectivity_arrays:
                datadesc.transient = False

            input_memlets[nsdfg_dataname] = sdfg.make_array_memlet(dataname)

            nsdfg_symbols_mapping |= {
                str(nested_symbol): parent_symbol
                for nested_symbol, parent_symbol in zip(
                    [*nsdfg_datadesc.shape, *nsdfg_datadesc.strides],
                    [*datadesc.shape, *datadesc.strides],
                    strict=True,
                )
                if isinstance(nested_symbol, dace.symbol)
            }

        # Process lambda outputs
        #
        lambda_output_nodes: Iterable[gtir_builtin_translators.Field] = (
            gtx_utils.flatten_nested_tuple(lambda_result)
        )
        # sanity check on isolated nodes
        assert all(
            nstate.degree(x.data_node) == 0
            for x in lambda_output_nodes
            if x.data_node.data in input_memlets
        )
        # keep only non-isolated output nodes
        lambda_outputs = {
            x.data_node.data for x in lambda_output_nodes if x.data_node.data not in input_memlets
        }

        if lambda_outputs:
            nsdfg_node = head_state.add_nested_sdfg(
                nsdfg,
                parent=sdfg,
                inputs=set(input_memlets.keys()),
                outputs=lambda_outputs,
                symbol_mapping=nsdfg_symbols_mapping,
                debuginfo=dace_utils.debug_info(node, default=sdfg.debuginfo),
            )

            for connector, memlet in input_memlets.items():
                if connector in lambda_arg_nodes:
                    src_node = lambda_arg_nodes[connector].data_node
                else:
                    src_node = head_state.add_access(memlet.data)

                head_state.add_edge(src_node, None, nsdfg_node, connector, memlet)

        def make_temps(
            x: gtir_builtin_translators.Field,
        ) -> gtir_builtin_translators.Field:
            if x.data_node.data in lambda_outputs:
                connector = x.data_node.data
                desc = x.data_node.desc(nsdfg)
                # make lambda result non-transient and map it to external temporary
                desc.transient = False
                # isolated access node will make validation fail
                if nstate.degree(x.data_node) == 0:
                    nstate.remove_node(x.data_node)
                temp, _ = sdfg.add_temp_transient_like(desc)
                dst_node = head_state.add_access(temp)
                head_state.add_edge(
                    nsdfg_node, connector, dst_node, None, sdfg.make_array_memlet(temp)
                )
                return gtir_builtin_translators.Field(dst_node, x.data_type)
            elif x.data_node.data in lambda_arg_nodes:
                nstate.remove_node(x.data_node)
                return lambda_arg_nodes[x.data_node.data]
            else:
                nstate.remove_node(x.data_node)
                data_node = head_state.add_access(x.data_node.data)
                return gtir_builtin_translators.Field(data_node, x.data_type)

        return gtx_utils.tree_map(make_temps)(lambda_result)

    def visit_Literal(
        self,
        node: gtir.Literal,
        sdfg: dace.SDFG,
        head_state: dace.SDFGState,
        reduce_identity: Optional[gtir_dataflow.SymbolExpr],
    ) -> gtir_builtin_translators.FieldopResult:
        return gtir_builtin_translators.translate_literal(
            node, sdfg, head_state, self, reduce_identity=None
        )

    def visit_SymRef(
        self,
        node: gtir.SymRef,
        sdfg: dace.SDFG,
        head_state: dace.SDFGState,
        reduce_identity: Optional[gtir_dataflow.SymbolExpr],
    ) -> gtir_builtin_translators.FieldopResult:
        return gtir_builtin_translators.translate_symbol_ref(
            node, sdfg, head_state, self, reduce_identity=None
        )


def build_sdfg_from_gtir(
    ir: gtir.Program,
    offset_provider: gtx_common.OffsetProvider,
) -> dace.SDFG:
    """
    Receives a GTIR program and lowers it to a DaCe SDFG.

    The lowering to SDFG requires that the program node is type-annotated, therefore this function
    runs type ineference as first step.

    Arguments:
        ir: The GTIR program node to be lowered to SDFG
        offset_provider: The definitions of offset providers used by the program node

    Returns:
        An SDFG in the DaCe canonical form (simplified)
    """

    ir = gtir_type_inference.infer(ir, offset_provider=offset_provider)
    ir = ir_prune_casts.PruneCasts().visit(ir)
    sdfg_genenerator = GTIRToSDFG(offset_provider)
    sdfg = sdfg_genenerator.visit(ir)
    assert isinstance(sdfg, dace.SDFG)

    return sdfg
