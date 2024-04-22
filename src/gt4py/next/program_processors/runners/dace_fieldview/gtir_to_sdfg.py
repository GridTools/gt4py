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
Class to lower GTIR to SDFG.

Note: this module covers the fieldview flavour of GTIR.
"""

from typing import Dict

import dace

from gt4py import eve
from gt4py.next.iterator import ir as itir
from gt4py.next.type_system import type_specifications as ts

from .gtir_fieldview_builder import GtirFieldviewBuilder as FieldviewBuilder
from .utility import as_dace_type


class GtirToSDFG(eve.NodeVisitor):
    """Provides translation capability from an GTIR program to a DaCe SDFG.

    This class is responsible for translation of `ir.Program`, that is the top level representation
    of a GT4Py program as a sequence of `it.Stmt` statements.
    Each statement is translated to a taskgraph inside a separate state. The parent SDFG and
    the translation state define the translation context, implemented by `ItirTaskgenContext`.
    Statement states are chained one after the other: potential concurrency between states should be
    extracted by the DaCe SDFG transformations.
    The program translation keeps track of entry and exit states: each statement is translated as
    a new state inserted just before the exit state. Note that statements with branch execution might
    result in more than one state. However, each statement should provide a single termination state
    (e.g. a join state for an if/else branch execution) on the exit state of the program SDFG.
    """

    _param_types: list[ts.TypeSpec]
    _field_types: dict[str, ts.FieldType]

    def __init__(
        self,
        param_types: list[ts.TypeSpec],
    ):
        self._param_types = param_types
        self._field_types = {}

    def _add_storage(self, sdfg: dace.SDFG, name: str, type_: ts.TypeSpec) -> None:
        if isinstance(type_, ts.FieldType):
            # TODO define shape based on domain and dtype based on type inference
            shape = [10]
            dtype = dace.float64
            sdfg.add_array(name, shape, dtype, transient=False)
            self._field_types[name] = type_
        else:
            assert isinstance(type_, ts.ScalarType)
            dtype = as_dace_type(type_.kind)
            sdfg.add_symbol(name, dtype)

    def _add_storage_for_temporary(self, temp_decl: itir.Temporary) -> Dict[str, str]:
        raise NotImplementedError()
        return {}

    def visit_Program(self, node: itir.Program) -> dace.SDFG:
        """Translates `ir.Program` to `dace.SDFG`.

        First, it will allocate array and scalar storage for external (aka non-transient)
        and local (aka transient) data. The local data, at this stage, is used
        for temporary declarations, which should be available everywhere in the SDFG
        but not outside.
        Then, all statements are translated, one after the other in separate states.
        """
        if node.function_definitions:
            raise NotImplementedError("Functions expected to be inlined as lambda calls.")

        sdfg = dace.SDFG(node.id)

        # we use entry/exit state to keep track of entry/exit point of graph execution
        entry_state = sdfg.add_state("program_entry", is_start_block=True)

        # declarations of temporaries result in local (aka transient) array definitions in the SDFG
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

        # add global arrays (aka non-transient) to the SDFG
        assert len(node.params) == len(self._param_types)
        for param, type_ in zip(node.params, self._param_types):
            self._add_storage(sdfg, str(param.id), type_)

        # visit one statement at a time and put it into separate state
        for i, stmt in enumerate(node.body):
            head_state = sdfg.add_state_after(head_state, f"stmt_{i}")
            self.visit(stmt, sdfg=sdfg, state=head_state)
            # sanity check below: each statement should have a single exit state -- aka no branches
            sink_states = sdfg.sink_nodes()
            assert len(sink_states) == 1
            head_state = sink_states[0]

        sdfg.validate()
        return sdfg

    def visit_SetAt(self, stmt: itir.SetAt, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Visits a statement expression and writes the local result to some external storage.

        Each statement expression results in some sort of taskgraph writing to local (aka transient) storage.
        The translation of `SetAt` ensures that the result is written to the external storage.
        """

        fieldview_builder = FieldviewBuilder(sdfg, state, self._field_types)
        fieldview_builder.visit(stmt.expr)

        # the target expression could be a `SymRef` to an output node or a `make_tuple` expression
        # in case the statement returns more than one field
        fieldview_builder.write_to(stmt.target)
