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

from typing import Any, Mapping, Sequence

import dace

from gt4py import eve
from gt4py.next.common import Connectivity, Dimension, DimensionKind
from gt4py.next.iterator import ir as itir
from gt4py.next.type_system import type_specifications as ts

from .gtir_dataflow_builder import GtirDataflowBuilder as DataflowBuilder
from .utility import as_dace_type, filter_connectivities


class GtirToSDFG(eve.NodeVisitor):
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

    _param_types: list[ts.TypeSpec]
    _data_types: dict[str, ts.FieldType | ts.ScalarType]
    _offset_providers: Mapping[str, Any]

    def __init__(
        self,
        param_types: list[ts.TypeSpec],
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
                # we reuse the same gt4py symbol for field size passed as scalar argument which is used in closure domain
                neighbor_tables[dim.value].max_neighbors
                if dim.kind == DimensionKind.LOCAL
                # we reuse the same gt4py symbol for field size passed as scalar argument which is used in closure domain
                else dace.symbol(f"__{name}_size_{i}", dtype)
            )
            for i, dim in enumerate(dims)
        ]
        strides = [dace.symbol(f"__{name}_stride_{i}", dtype) for i, _ in enumerate(dims)]
        return shape, strides

    def _add_storage(self, sdfg: dace.SDFG, name: str, type_: ts.TypeSpec) -> None:
        """
        Add external storage (aka non-transient) for data containers passed as arguments to the SDFG.

        For fields, it allocates dace arrays, while scalars are stored as SDFG symbols.
        """
        assert isinstance(type_, (ts.FieldType, ts.ScalarType))
        self._data_types[name] = type_

        if isinstance(type_, ts.FieldType):
            dtype = as_dace_type(type_.dtype)
            # use symbolic shape, which allows to invoke the program with fields of different size;
            # and symbolic strides, which enables decoupling the memory layout from generated code.
            sym_shape, sym_strides = self._make_array_shape_and_strides(name, type_.dims)
            sdfg.add_array(name, sym_shape, dtype, strides=sym_strides, transient=False)
        else:
            dtype = as_dace_type(type_)
            # scalar arguments passed to the program are represented as symbols in DaCe SDFG
            sdfg.add_symbol(name, dtype)

    def _add_storage_for_temporary(self, temp_decl: itir.Temporary) -> Mapping[str, str]:
        """
        Add temporary storage (aka transient) for data containers used as GTIR temporaries.

        Assume all temporaries to be fields, therefore represented as dace arrays.
        """
        raise NotImplementedError("Temporaries not supported yet by GTIR DaCe backend.")
        return {}

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
        assert len(node.params) == len(self._param_types)
        for param, type_ in zip(node.params, self._param_types):
            self._add_storage(sdfg, str(param.id), type_)

        # visit one statement at a time and expand the SDFG from the current head state
        for i, stmt in enumerate(node.body):
            head_state = sdfg.add_state_after(head_state, f"stmt_{i}")
            # the statement could eventually modify the head state by appending new states
            # however, it should preserve the property of single exit state (aka head state)
            head_state = self.visit(stmt, sdfg=sdfg, state=head_state)

        sdfg.validate()
        return sdfg

    def visit_SetAt(
        self, stmt: itir.SetAt, sdfg: dace.SDFG, state: dace.SDFGState
    ) -> dace.SDFGState:
        """Visits a `SetAt` statement expression and writes the local result to some external storage.

        Each statement expression results in some sort of dataflow gragh writing to temporary storage.
        The translation of `SetAt` ensures that the result is written to some global storage.
        """

        dataflow_builder = DataflowBuilder(sdfg, state, self._data_types)
        head_state, expr_nodes = dataflow_builder.visit_expression(stmt.expr)

        # the target expression could be a `SymRef` to an output node or a `make_tuple` expression
        # in case the statement returns more than one field
        target_builder = DataflowBuilder(sdfg, head_state, self._data_types)
        head_state, target_nodes = target_builder.visit_expression(stmt.target)
        assert len(expr_nodes) == len(target_nodes)

        domain = dataflow_builder.visit_domain(stmt.domain)
        # convert domain to dictionary to ease access to dimension boundaries
        domain_map = {dim: (lb, ub) for dim, lb, ub in domain}

        for expr_node, target_node in zip(expr_nodes, target_nodes):
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

            head_state.add_nedge(
                expr_node,
                target_node,
                dace.Memlet(data=target_node.data, subset=subset),
            )

        return head_state
