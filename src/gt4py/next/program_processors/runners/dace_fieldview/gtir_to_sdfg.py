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
Class to lower GTIR to DaCe SDFG.

Note: this module covers the fieldview flavour of GTIR.
"""

from typing import Any, Sequence

import dace

from gt4py import eve
from gt4py.next.common import Connectivity, Dimension, DimensionKind
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace_fieldview import (
    gtir_builtin_translators,
    utility as dace_fieldview_util,
)
from gt4py.next.type_system import type_specifications as ts, type_translation as tt


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

    param_types: list[ts.DataType]
    offset_provider: dict[str, Connectivity | Dimension]
    symbol_types: dict[str, ts.FieldType | ts.ScalarType]

    def __init__(
        self,
        param_types: list[ts.DataType],
        offset_provider: dict[str, Connectivity | Dimension],
    ):
        self.param_types = param_types
        self.offset_provider = offset_provider
        self.symbol_types = {}

    def _make_array_shape_and_strides(
        self, name: str, dims: Sequence[Dimension]
    ) -> tuple[list[dace.symbol], list[dace.symbol]]:
        """
        Parse field dimensions and allocate symbols for array shape and strides.

        For local dimensions, the size is known at compile-time and therefore
        the corresponding array shape dimension is set to an integer literal value.

        Returns:
            Two lists of symbols, one for the shape and the other for the strides of the array.
        """
        dtype = dace.int32
        neighbor_tables = dace_fieldview_util.filter_connectivities(self.offset_provider)
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

    def _add_storage(
        self, sdfg: dace.SDFG, name: str, symbol_type: ts.DataType, transient: bool = False
    ) -> None:
        """
        Add external storage (aka non-transient) for data containers passed as arguments to the SDFG.

        For fields, it allocates dace arrays, while scalars are stored as SDFG symbols.
        """
        if isinstance(symbol_type, ts.FieldType):
            dtype = dace_fieldview_util.as_dace_type(symbol_type.dtype)
            # use symbolic shape, which allows to invoke the program with fields of different size;
            # and symbolic strides, which enables decoupling the memory layout from generated code.
            sym_shape, sym_strides = self._make_array_shape_and_strides(name, symbol_type.dims)
            sdfg.add_array(name, sym_shape, dtype, strides=sym_strides, transient=transient)
        elif isinstance(symbol_type, ts.ScalarType):
            dtype = dace_fieldview_util.as_dace_type(symbol_type)
            # scalar arguments passed to the program are represented as symbols in DaCe SDFG
            sdfg.add_symbol(name, dtype)
        else:
            raise RuntimeError(f"Data type '{type(symbol_type)}' not supported.")

        # TODO: unclear why mypy complains about incompatible types
        assert isinstance(symbol_type, (ts.FieldType, ts.ScalarType))
        self.symbol_types[name] = symbol_type

    def _add_storage_for_temporary(self, temp_decl: itir.Temporary) -> dict[str, str]:
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

        This method represents the entry point to visit `ir.Stmt` expressions.
        As such, it must preserve the property of single exit state in the SDFG.

        Returns a list of array nodes containing the result fields.

        TODO: do we need to return the GT4Py `FieldType`/`ScalarType`?
        """
        field_builder: gtir_builtin_translators.SDFGFieldBuilder = self.visit(
            node, sdfg=sdfg, head_state=head_state
        )
        results = field_builder()

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

        if len(node.params) != len(self.param_types):
            raise RuntimeError(
                "The provided list of parameter types has different length than SDFG parameter list."
            )

        sdfg = dace.SDFG(node.id)
        sdfg.debuginfo = dace_fieldview_util.debuginfo(node)
        entry_state = sdfg.add_state("program_entry", is_start_block=True)

        # declarations of temporaries result in transient array definitions in the SDFG
        if node.declarations:
            temp_symbols: dict[str, str] = {}
            for decl in node.declarations:
                temp_symbols |= self._add_storage_for_temporary(decl)

            # define symbols for shape and offsets of temporary arrays as interstate edge symbols
            # TODO(edopao): use new `add_state_after` function available in next dace release
            head_state = sdfg.add_state_after(entry_state, "init_temps")
            sdfg.edges_between(entry_state, head_state)[0].assignments = temp_symbols
        else:
            head_state = entry_state

        # add non-transient arrays and/or SDFG symbols for the program arguments
        for param, type_ in zip(node.params, self.param_types, strict=True):
            self._add_storage(sdfg, str(param.id), type_)

        # add SDFG storage for connectivity tables
        for offset, offset_provider in dace_fieldview_util.filter_connectivities(
            self.offset_provider
        ).items():
            scalar_kind = tt.get_scalar_kind(offset_provider.index_type)
            local_dim = Dimension(offset, kind=DimensionKind.LOCAL)
            type_ = ts.FieldType(
                [offset_provider.origin_axis, local_dim], ts.ScalarType(scalar_kind)
            )
            # We store all connectivity tables as transient arrays here; later, while building
            # the field operator expressions, we change to non transient the tables
            # that are actually used. This way, we avoid adding SDFG arguments for
            # the connectivity tabkes that are not used.
            self._add_storage(
                sdfg, dace_fieldview_util.connectivity_identifier(offset), type_, transient=True
            )

        # visit one statement at a time and expand the SDFG from the current head state
        for i, stmt in enumerate(node.body):
            # include `debuginfo` only for `ir.Program` and `ir.Stmt` nodes: finer granularity would be too messy
            head_state = sdfg.add_state_after(head_state, f"stmt_{i}")
            head_state._debuginfo = dace_fieldview_util.debuginfo(stmt)
            self.visit(stmt, sdfg=sdfg, state=head_state)

        sdfg.validate()
        return sdfg

    def visit_SetAt(self, stmt: itir.SetAt, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Visits a `SetAt` statement expression and writes the local result to some external storage.

        Each statement expression results in some sort of dataflow gragh writing to temporary storage.
        The translation of `SetAt` ensures that the result is written back to the target external storage.
        """

        expr_nodes = self._visit_expression(stmt.expr, sdfg, state)

        # the target expression could be a `SymRef` to an output node or a `make_tuple` expression
        # in case the statement returns more than one field
        target_nodes = self._visit_expression(stmt.target, sdfg, state)

        # convert domain expression to dictionary to ease access to dimension boundaries
        domain = dace_fieldview_util.get_domain(stmt.domain)

        for expr_node, target_node in zip(expr_nodes, target_nodes, strict=True):
            target_array = sdfg.arrays[target_node.data]
            assert not target_array.transient
            target_symbol_type = self.symbol_types[target_node.data]

            if isinstance(target_symbol_type, ts.FieldType):
                subset = ",".join(
                    f"{domain[dim.value][0]}:{domain[dim.value][1]}"
                    for dim in target_symbol_type.dims
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
    ) -> gtir_builtin_translators.SDFGFieldBuilder:
        # first visit the argument nodes
        arg_builders = []
        for arg in node.args:
            arg_builder: gtir_builtin_translators.SDFGFieldBuilder = self.visit(
                arg, sdfg=sdfg, head_state=head_state
            )
            arg_builders.append(arg_builder)

        # use specialized dataflow builder classes for each builtin function
        if cpm.is_call_to(node.fun, "as_fieldop"):
            return gtir_builtin_translators.AsFieldOp(
                sdfg, head_state, node, arg_builders, self.offset_provider
            )
        elif cpm.is_call_to(node.fun, "select"):
            assert len(arg_builders) == 0
            return gtir_builtin_translators.Select(sdfg, head_state, self, node)
        else:
            raise NotImplementedError(f"Unexpected 'FunCall' expression ({node}).")

    def visit_Lambda(self, node: itir.Lambda) -> Any:
        """
        This visitor class should never encounter `itir.Lambda` expressions
        because a lambda represents a stencil, which operates from iterator to values.
        In fieldview, lambdas should only be arguments to field operators (`as_field_op`).
        """
        raise RuntimeError("Unexpected 'itir.Lambda' node encountered in GTIR.")

    def visit_SymRef(
        self, node: itir.SymRef, sdfg: dace.SDFG, head_state: dace.SDFGState
    ) -> gtir_builtin_translators.SDFGFieldBuilder:
        symbol_name = str(node.id)
        assert symbol_name in self.symbol_types
        symbol_type = self.symbol_types[symbol_name]
        return gtir_builtin_translators.SymbolRef(sdfg, head_state, symbol_name, symbol_type)
