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
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple, Union

import dace

from gt4py import eve
from gt4py.eve import concepts
from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.iterator.type_system import inference as gtir_type_inference
from gt4py.next.program_processors.runners.dace_common import utility as dace_utils
from gt4py.next.program_processors.runners.dace_fieldview import (
    gtir_builtin_translators,
    gtir_to_tasklet,
    transformations as gtx_transformations,
    utility as dace_gtir_utils,
)
from gt4py.next.type_system import type_specifications as ts, type_translation as tt


class DataflowBuilder(Protocol):
    """Visitor interface to build a dataflow subgraph."""

    @abc.abstractmethod
    def get_offset_provider(self, offset: str) -> gtx_common.Connectivity | gtx_common.Dimension:
        pass

    @abc.abstractmethod
    def unique_map_name(self, name: str) -> str:
        pass

    @abc.abstractmethod
    def unique_tasklet_name(self, name: str) -> str:
        pass

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


class SDFGBuilder(DataflowBuilder, Protocol):
    """Visitor interface available to GTIR-primitive translators."""

    @abc.abstractmethod
    def get_symbol_type(self, symbol_name: str) -> ts.FieldType | ts.ScalarType:
        """Retrieve the GT4Py type of a symbol used in the program."""
        pass

    @abc.abstractmethod
    def visit(self, node: concepts.RootNode, **kwargs: Any) -> Any:
        """Visit a node of the GT4Py IR."""
        pass


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

    offset_provider: dict[str, gtx_common.Connectivity | gtx_common.Dimension]
    global_symbols: dict[str, ts.FieldType | ts.ScalarType] = dataclasses.field(
        default_factory=lambda: {}
    )
    map_uids: eve.utils.UIDGenerator = dataclasses.field(
        init=False, repr=False, default_factory=lambda: eve.utils.UIDGenerator(prefix="map")
    )
    tesklet_uids: eve.utils.UIDGenerator = dataclasses.field(
        init=False, repr=False, default_factory=lambda: eve.utils.UIDGenerator(prefix="tlet")
    )

    def get_offset_provider(self, offset: str) -> gtx_common.Connectivity | gtx_common.Dimension:
        return self.offset_provider[offset]

    def get_symbol_type(self, symbol_name: str) -> ts.FieldType | ts.ScalarType:
        return self.global_symbols[symbol_name]

    def unique_map_name(self, name: str) -> str:
        return f"{self.map_uids.sequential_id()}_{name}"

    def unique_tasklet_name(self, name: str) -> str:
        return f"{self.tesklet_uids.sequential_id()}_{name}"

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
        self, sdfg: dace.SDFG, name: str, symbol_type: ts.DataType, transient: bool = True
    ) -> None:
        """
        Add storage for data containers used in the SDFG. For fields, it allocates dace arrays,
        while scalars are stored as SDFG symbols.

        The fields used as temporary arrays, when `transient = True`, are allocated and exist
        only within the SDFG; when `transient = False`, the fields have to be allocated outside
        and have to be passed as array arguments to the SDFG.
        """
        if isinstance(symbol_type, ts.FieldType):
            dtype = dace_utils.as_dace_type(symbol_type.dtype)
            # use symbolic shape, which allows to invoke the program with fields of different size;
            # and symbolic strides, which enables decoupling the memory layout from generated code.
            sym_shape, sym_strides = self._make_array_shape_and_strides(name, symbol_type.dims)
            sdfg.add_array(name, sym_shape, dtype, strides=sym_strides, transient=transient)
        elif isinstance(symbol_type, ts.ScalarType):
            assert isinstance(symbol_type, ts.ScalarType)
            dtype = dace_utils.as_dace_type(symbol_type)
            # Scalar arguments passed to the program are represented as symbols in DaCe SDFG.
            # The field size is sometimes passed as scalar argument to the program, so we have to
            # check if the shape symbol was already allocated by `_make_array_shape_and_strides`.
            # We assume that the scalar argument for field size always follows the field argument.
            if name in sdfg.symbols:
                assert sdfg.symbols[name].dtype == dtype
            else:
                sdfg.add_symbol(name, dtype)
        else:
            raise RuntimeError(f"Data type '{type(symbol_type)}' not supported.")

    def _add_storage_for_temporary(self, temp_decl: gtir.Temporary) -> dict[str, str]:
        """
        Add temporary storage (aka transient) for data containers used as GTIR temporaries.

        Assume all temporaries to be fields, therefore represented as dace arrays.
        """
        raise NotImplementedError("Temporaries not supported yet by GTIR DaCe backend.")

    def _visit_expression(
        self, node: gtir.Expr, sdfg: dace.SDFG, head_state: dace.SDFGState
    ) -> list[dace.nodes.AccessNode]:
        """
        Specialized visit method for fieldview expressions.

        This method represents the entry point to visit `ir.Stmt` expressions.
        As such, it must preserve the property of single exit state in the SDFG.

        Returns a list of array nodes containing the result fields.

        TODO: Do we need to return the GT4Py `FieldType`/`ScalarType`? It is needed
        in case the transient arrays containing the expression result are not guaranteed
        to have the same memory layout as the target array.
        """
        results: list[gtir_builtin_translators.TemporaryData] = self.visit(
            node, sdfg=sdfg, head_state=head_state, reduce_identity=None
        )

        field_nodes = []
        for node, _ in results:
            assert isinstance(node, dace.nodes.AccessNode)
            field_nodes.append(node)

        # sanity check: each statement should preserve the property of single exit state (aka head state),
        # i.e. eventually only introduce internal branches, and keep the same head state
        sink_states = sdfg.sink_nodes()
        assert len(sink_states) == 1
        assert sink_states[0] == head_state

        return field_nodes

    def _add_sdfg_params(self, sdfg: dace.SDFG, node_params: Sequence[gtir.Sym]) -> None:
        """Helper function to add storage for node parameters and connectivity tables."""

        # add non-transient arrays and/or SDFG symbols for the program arguments
        for param in node_params:
            pname = str(param.id)
            assert isinstance(param.type, (ts.FieldType, ts.ScalarType))
            self._add_storage(sdfg, pname, param.type, transient=False)
            self.global_symbols[pname] = param.type

        # add SDFG storage for connectivity tables
        for offset, offset_provider in dace_utils.filter_connectivities(
            self.offset_provider
        ).items():
            scalar_kind = tt.get_scalar_kind(offset_provider.index_type)
            local_dim = gtx_common.Dimension(offset, kind=gtx_common.DimensionKind.LOCAL)
            type_ = ts.FieldType(
                [offset_provider.origin_axis, local_dim], ts.ScalarType(scalar_kind)
            )
            # We store all connectivity tables as transient arrays here; later, while building
            # the field operator expressions, we change to non-transient (i.e. allocated externally)
            # the tables that are actually used. This way, we avoid adding SDFG arguments for
            # the connectivity tables that are not used. The remaining unused transient arrays
            # are removed by the dace simplify pass.
            self._add_storage(sdfg, dace_utils.connectivity_identifier(offset), type_)

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

        self._add_sdfg_params(sdfg, node.params)

        # visit one statement at a time and expand the SDFG from the current head state
        for i, stmt in enumerate(node.body):
            # include `debuginfo` only for `ir.Program` and `ir.Stmt` nodes: finer granularity would be too messy
            head_state = sdfg.add_state_after(head_state, f"stmt_{i}")
            head_state._debuginfo = dace_utils.debug_info(stmt, default=sdfg.debuginfo)
            self.visit(stmt, sdfg=sdfg, state=head_state)

        # Create the call signature for the SDFG.
        #  Only the arguments required by the GT4Py program, i.e. `node.params`, are added
        #  as positional arguments. The implicit arguments, such as the offset providers or
        #  the arguments created by the translation process, must be passed as keyword arguments.
        sdfg.arg_names = [str(a) for a in node.params]

        sdfg.validate()
        return sdfg

    def visit_SetAt(self, stmt: gtir.SetAt, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Visits a `SetAt` statement expression and writes the local result to some external storage.

        Each statement expression results in some sort of dataflow gragh writing to temporary storage.
        The translation of `SetAt` ensures that the result is written back to the target external storage.
        """

        expr_nodes = self._visit_expression(stmt.expr, sdfg, state)

        # the target expression could be a `SymRef` to an output node or a `make_tuple` expression
        # in case the statement returns more than one field
        target_nodes = self._visit_expression(stmt.target, sdfg, state)

        # convert domain expression to dictionary to ease access to dimension boundaries
        domain = dace_gtir_utils.get_domain_ranges(stmt.domain)

        for expr_node, target_node in zip(expr_nodes, target_nodes, strict=True):
            target_array = sdfg.arrays[target_node.data]
            assert not target_array.transient
            target_symbol_type = self.global_symbols[target_node.data]

            if isinstance(target_symbol_type, ts.FieldType):
                subset = ",".join(
                    f"{domain[dim][0]}:{domain[dim][1]}" for dim in target_symbol_type.dims
                )
            else:
                assert len(domain) == 0
                subset = "0"

            state.add_nedge(
                expr_node,
                target_node,
                dace.Memlet(data=target_node.data, subset=subset, other_subset=subset),
            )

    def visit_FunCall(
        self,
        node: gtir.FunCall,
        sdfg: dace.SDFG,
        head_state: dace.SDFGState,
        reduce_identity: Optional[gtir_to_tasklet.SymbolExpr],
    ) -> list[gtir_builtin_translators.TemporaryData]:
        # use specialized dataflow builder classes for each builtin function
        if cpm.is_call_to(node, "if_"):
            return gtir_builtin_translators.translate_if(
                node, sdfg, head_state, self, reduce_identity
            )
        elif cpm.is_call_to(node.fun, "as_fieldop"):
            return gtir_builtin_translators.translate_as_field_op(
                node, sdfg, head_state, self, reduce_identity
            )
        elif isinstance(node.fun, gtir.Lambda):
            node_args = []
            for arg in node.args:
                node_args.extend(
                    self.visit(
                        arg,
                        sdfg=sdfg,
                        head_state=head_state,
                        reduce_identity=reduce_identity,
                    )
                )

            return self.visit(
                node.fun,
                sdfg=sdfg,
                head_state=head_state,
                reduce_identity=reduce_identity,
                args=node_args,
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
        reduce_identity: Optional[gtir_to_tasklet.SymbolExpr],
        args: list[gtir_builtin_translators.TemporaryData],
    ) -> list[gtir_builtin_translators.TemporaryData]:
        """
        Translates a `Lambda` node to a nested SDFG in the current state.

        All arguments to lambda functions are fields (i.e. `as_fieldop`, field or scalar `gtir.SymRef`,
        nested let-lambdas thereof). The reason for creating a nested SDFG is to define local symbols
        (the lambda paremeters) that map to parent fields, either program arguments or temporary fields.

        If the lambda has a parameter whose name is already present in `GTIRToSDFG.global_symbols`,
        i.e. a lambda parameter with the same name as a symbol in scope, the parameter will shadow
        the previous symbol during traversal of the lambda expression.
        """

        lambda_args_mapping = {str(p.id): arg for p, arg in zip(node.params, args, strict=True)}

        # inherit symbols from parent scope but eventually override with local symbols
        lambda_symbols = self.global_symbols | {
            pname: type_ for pname, (_, type_) in lambda_args_mapping.items()
        }

        nsdfg = dace.SDFG(f"{sdfg.label}_nested")
        nstate = nsdfg.add_state("lambda")

        # add sdfg storage for the symbols that need to be passed as input parameters,
        # that are only the symbols used in the context of the lambda node
        self._add_sdfg_params(
            nsdfg,
            [gtir.Sym(id=p_name, type=p_type) for p_name, p_type in lambda_symbols.items()],
        )

        lambda_nodes = GTIRToSDFG(self.offset_provider, lambda_symbols.copy()).visit(
            node.expr,
            sdfg=nsdfg,
            head_state=nstate,
            reduce_identity=reduce_identity,
        )

        connectivity_arrays = {
            dace_utils.connectivity_identifier(offset)
            for offset in dace_utils.filter_connectivities(self.offset_provider)
        }
        nsdfg_symbols_mapping: dict[str, dace.symbolic.SymExpr] = {}

        input_memlets = {}
        for nsdfg_dataname, nsdfg_datadesc in nsdfg.arrays.items():
            if nsdfg_datadesc.transient:
                continue
            datadesc: Optional[dace.dtypes.Array] = None
            if nsdfg_dataname in lambda_args_mapping:
                src_node, _ = lambda_args_mapping[nsdfg_dataname]
                dataname = src_node.data
                datadesc = src_node.desc(sdfg)

                nsdfg_symbols_mapping |= {
                    str(nested_symbol): parent_symbol
                    for nested_symbol, parent_symbol in zip(
                        [*nsdfg_datadesc.shape, *nsdfg_datadesc.strides],
                        [*datadesc.shape, *datadesc.strides],
                        strict=True,
                    )
                    if isinstance(nested_symbol, dace.symbol)
                }
            else:
                dataname = nsdfg_dataname
                datadesc = sdfg.arrays[nsdfg_dataname]
                # ensure that connectivity tables are non-transient arrays in parent SDFG
                if dataname in connectivity_arrays:
                    datadesc.transient = False

            if datadesc:
                input_memlets[nsdfg_dataname] = dace.Memlet.from_array(dataname, datadesc)

        nsdfg_node = head_state.add_nested_sdfg(
            nsdfg,
            parent=sdfg,
            inputs=set(input_memlets.keys()),
            outputs=set(node.data for node, _ in lambda_nodes),
            symbol_mapping=nsdfg_symbols_mapping,
            debuginfo=dace_utils.debug_info(node, default=sdfg.debuginfo),
        )

        for connector, memlet in input_memlets.items():
            if connector in lambda_args_mapping:
                src_node, _ = lambda_args_mapping[connector]
            else:
                src_node = head_state.add_access(memlet.data)

            head_state.add_edge(src_node, None, nsdfg_node, connector, memlet)

        results = []
        for lambda_node, type_ in lambda_nodes:
            connector = lambda_node.data
            desc = lambda_node.desc(nsdfg)
            # make lambda result non-transient and map it to external temporary
            desc.transient = False
            # isolated access node will make validation fail
            if nstate.degree(lambda_node) == 0:
                nstate.remove_node(lambda_node)
            temp, _ = sdfg.add_temp_transient_like(desc)
            dst_node = head_state.add_access(temp)
            head_state.add_edge(
                nsdfg_node, connector, dst_node, None, dace.Memlet.from_array(temp, desc)
            )
            results.append((dst_node, type_))

        return results

    def visit_Literal(
        self,
        node: gtir.Literal,
        sdfg: dace.SDFG,
        head_state: dace.SDFGState,
        reduce_identity: Optional[gtir_to_tasklet.SymbolExpr],
    ) -> list[gtir_builtin_translators.TemporaryData]:
        return gtir_builtin_translators.translate_literal(
            node, sdfg, head_state, self, reduce_identity=None
        )

    def visit_SymRef(
        self,
        node: gtir.SymRef,
        sdfg: dace.SDFG,
        head_state: dace.SDFGState,
        reduce_identity: Optional[gtir_to_tasklet.SymbolExpr],
    ) -> list[gtir_builtin_translators.TemporaryData]:
        return gtir_builtin_translators.translate_symbol_ref(
            node, sdfg, head_state, self, reduce_identity=None
        )


def build_sdfg_from_gtir(
    program: gtir.Program,
    offset_provider: dict[str, gtx_common.Connectivity | gtx_common.Dimension],
) -> dace.SDFG:
    """
    Receives a GTIR program and lowers it to a DaCe SDFG.

    The lowering to SDFG requires that the program node is type-annotated, therefore this function
    runs type ineference as first step.
    As a final step, it runs the `simplify` pass to ensure that the SDFG is in the DaCe canonical form.

    Arguments:
        program: The GTIR program node to be lowered to SDFG
        offset_provider: The definitions of offset providers used by the program node

    Returns:
        An SDFG in the DaCe canonical form (simplified)
    """
    program = gtir_type_inference.infer(program, offset_provider=offset_provider)
    sdfg_genenerator = GTIRToSDFG(offset_provider)
    sdfg = sdfg_genenerator.visit(program)
    assert isinstance(sdfg, dace.SDFG)

    gtx_transformations.gt_simplify(sdfg)
    return sdfg
