# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Class to lower GTIR to a DaCe SDFG.

Note: this module covers the fieldview flavour of GTIR.
"""

from typing import Any, Callable, Mapping, Sequence, final

import dace

import gt4py.eve as eve
from gt4py.next.common import Connectivity, Dimension, DimensionKind
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace_fieldview import gtir_builtins
from gt4py.next.program_processors.runners.dace_fieldview.utility import (
    as_dace_type,
    filter_connectivities,
)
from gt4py.next.type_system import type_specifications as ts


class GTIRToSDFG(eve.NodeVisitor):
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

    _param_types: list[ts.DataType]
    _data_types: dict[str, ts.FieldType | ts.ScalarType]
    _offset_providers: Mapping[str, Any]

    def __init__(
        self,
        param_types: list[ts.DataType],
        offset_providers: dict[str, Connectivity | Dimension],
    ):
        self._param_types = param_types
        self._data_types = {}
        self._offset_providers = offset_providers

    def _make_array_shape_and_strides(
        self, name: str, dims: Sequence[Dimension]
    ) -> tuple[Sequence[dace.symbol], Sequence[dace.symbol]]:
        """
        Parse field dimensions and allocate symbols for array shape and strides.

        For local dimensions, the size is known at compile-time and therefore
        the corresponding array shape dimension is set to an integer literal value.

        Returns
        -------
        tuple(shape, strides)
            The output tuple fields are arrays of dace symbolic expressions.
        """
        dtype = dace.int32
        neighbor_tables = filter_connectivities(self._offset_providers)
        shape = [
            (
                neighbor_tables[dim.value].max_neighbors
                if dim.kind == DimensionKind.LOCAL
                # we reuse the same symbol for field size passed as scalar argument to the gt4py program
                else dace.symbol(f"__{name}_size_{i}", dtype)
            )
            for i, dim in enumerate(dims)
        ]
        strides = [dace.symbol(f"__{name}_stride_{i}", dtype) for i in range(len(dims))]
        return shape, strides

    def _add_storage(self, sdfg: dace.SDFG, name: str, data_type: ts.DataType) -> None:
        """
        Add external storage (aka non-transient) for data containers passed as arguments to the SDFG.

        For fields, it allocates dace arrays, while scalars are stored as SDFG symbols.
        """
        if isinstance(data_type, ts.FieldType):
            dtype = as_dace_type(data_type.dtype)
            # use symbolic shape, which allows to invoke the program with fields of different size;
            # and symbolic strides, which enables decoupling the memory layout from generated code.
            sym_shape, sym_strides = self._make_array_shape_and_strides(name, data_type.dims)
            sdfg.add_array(name, sym_shape, dtype, strides=sym_strides, transient=False)
        elif isinstance(data_type, ts.ScalarType):
            dtype = as_dace_type(data_type)
            # scalar arguments passed to the program are represented as symbols in DaCe SDFG
            sdfg.add_symbol(name, dtype)
        else:
            raise RuntimeError(f"Data type '{type(data_type)}' not supported.")

        # TODO: unclear why mypy complains about incompatible types
        assert isinstance(data_type, (ts.FieldType, ts.ScalarType))
        self._data_types[name] = data_type

    def _add_storage_for_temporary(self, temp_decl: itir.Temporary) -> Mapping[str, str]:
        """
        Add temporary storage (aka transient) for data containers used as GTIR temporaries.

        Assume all temporaries to be fields, therefore represented as dace arrays.
        """
        raise NotImplementedError("Temporaries not supported yet by GTIR DaCe backend.")

    def _visit_expression(
        self, node: itir.Expr, sdfg: dace.SDFG, head_state: dace.SDFGState
    ) -> list[dace.nodes.AccessNode]:
        """
        Specialized visit method for fieldview expressions.

        This method represents the entry point to visit 'Stmt' expressions.
        As such, it must preserve the property of single exit state in the SDFG.

        TODO: do we need to return the GT4Py `FieldType`/`ScalarType`?
        """
        expr_builder = self.visit(node, sdfg=sdfg, head_state=head_state)
        assert callable(expr_builder)
        results = expr_builder()

        expressions_nodes = []
        for node, _ in results:
            assert isinstance(node, dace.nodes.AccessNode)
            expressions_nodes.append(node)

        # sanity check: each statement should preserve the property of single exit state (aka head state),
        # i.e. eventually only introduce internal branches, and keep the same head state
        sink_states = sdfg.sink_nodes()
        assert len(sink_states) == 1
        assert sink_states[0] == head_state

        return expressions_nodes

    def visit_Program(self, node: itir.Program) -> dace.SDFG:
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
        entry_state = sdfg.add_state("program_entry", is_start_block=True)

        # declarations of temporaries result in transient array definitions in the SDFG
        if node.declarations:
            temp_symbols: dict[str, str] = {}
            for decl in node.declarations:
                temp_symbols |= self._add_storage_for_temporary(decl)

            # define symbols for shape and offsets of temporary arrays as interstate edge symbols
            # TODO(edopao): use new `add_state_after` function in next dace release
            head_state = sdfg.add_state_after(entry_state, "init_temps")
            sdfg.edges_between(entry_state, head_state)[0].assignments = temp_symbols
        else:
            head_state = entry_state

        # add non-transient arrays and/or SDFG symbols for the program arguments
        for param, type_ in zip(node.params, self._param_types, strict=True):
            self._add_storage(sdfg, str(param.id), type_)

        # visit one statement at a time and expand the SDFG from the current head state
        for i, stmt in enumerate(node.body):
            head_state = sdfg.add_state_after(head_state, f"stmt_{i}")
            self.visit(stmt, sdfg=sdfg, state=head_state)

        sdfg.validate()
        return sdfg

    def visit_SetAt(self, stmt: itir.SetAt, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Visits a `SetAt` statement expression and writes the local result to some external storage.

        Each statement expression results in some sort of dataflow gragh writing to temporary storage.
        The translation of `SetAt` ensures that the result is written to some global storage.
        """

        expr_nodes = self._visit_expression(stmt.expr, sdfg, state)

        # the target expression could be a `SymRef` to an output node or a `make_tuple` expression
        # in case the statement returns more than one field
        target_nodes = self._visit_expression(stmt.target, sdfg, state)

        domain = gtir_builtins.FieldDomain(sdfg, state).visit_domain(stmt.domain)
        # convert domain to dictionary to ease access to dimension boundaries
        domain_map = {dim: (lb, ub) for dim, lb, ub in domain}

        for expr_node, target_node in zip(expr_nodes, target_nodes, strict=True):
            target_array = sdfg.arrays[target_node.data]
            assert not target_array.transient
            target_field_type = self._data_types[target_node.data]

            if isinstance(target_field_type, ts.FieldType):
                subset = ",".join(
                    f"{domain_map[dim][0]}:{domain_map[dim][1]}" for dim in target_field_type.dims
                )
            else:
                assert len(domain) == 0
                subset = "0"

            state.add_nedge(
                expr_node,
                target_node,
                dace.Memlet(data=target_node.data, subset=subset),
            )

    def visit_FunCall(
        self, node: itir.FunCall, sdfg: dace.SDFG, head_state: dace.SDFGState
    ) -> Callable:
        arg_builders: list[Callable] = []
        for arg in node.args:
            arg_builder = self.visit(arg, sdfg=sdfg, head_state=head_state)
            assert callable(arg_builder)
            arg_builders.append(arg_builder)

        if cpm.is_call_to(node.fun, "as_fieldop"):
            return gtir_builtins.AsFieldOp(sdfg, head_state, node, arg_builders)

        elif cpm.is_call_to(node.fun, "select"):
            assert len(arg_builders) == 0
            return gtir_builtins.Select(sdfg, head_state, self, node)

        else:
            raise NotImplementedError(f"Unexpected 'FunCall' expression ({node}).")

    @final
    def visit_Lambda(self, node: itir.Lambda) -> Any:
        """
        This visitor class should never encounter `itir.Lambda` expressions
        because a lambda represents a stencil, which translates from iterator to values.
        In fieldview, lambdas should only be arguments to field operators (`as_field_op`).
        """
        raise RuntimeError("Unexpected 'itir.Lambda' node encountered by 'GtirTaskletCodegen'.")

    def visit_SymRef(
        self, node: itir.SymRef, sdfg: dace.SDFG, head_state: dace.SDFGState
    ) -> Callable:
        sym_name = str(node.id)
        assert sym_name in self._data_types
        sym_type = self._data_types[sym_name]
        return gtir_builtins.SymbolRef(sdfg, head_state, sym_name, sym_type)
