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
Class to lower ITIR to SDFG.

Note: this module covers the fieldview flavour of ITIR.
"""

from collections import deque
from typing import Dict, List, Sequence, Tuple

import dace

from gt4py import eve
from gt4py.next.iterator import ir as itir
from gt4py.next.type_system import type_specifications as ts

from .itir_taskgen import ItirTaskgenContext as TaskgenContext
from .itir_to_tasklet import ItirToTasklet


class ItirToSDFG(eve.NodeVisitor):
    """Provides translation capability from an ITIR program to a DaCe SDFG.

    This class is responsible for translation of `ir.Program`, that is the top level representation
    of a GT4Py program as a sequence of `it.Stmt` statements.
    Each statement is translated to a taskgraph inside a separate state. The parent SDFG and
    the translation state define the translation context, implemented by `TaskgenContext`.
    Statement states are chained one after the other: potential concurrency between states should be
    extracted by the DaCe SDFG transformations.
    The program translation keeps track of entry and exit states: each statement is translated as
    a new state inserted just before the exit state. Note that statements with branch execution might
    result in more than one state.
    """

    _ctx_stack: deque[TaskgenContext]
    _param_types: list[ts.TypeSpec]

    def __init__(
        self,
        param_types: list[ts.TypeSpec],
    ):
        self._ctx_stack = deque()
        self._param_types = param_types

    def _add_storage(self, sdfg: dace.SDFG, name: str, type_: ts.TypeSpec) -> None:
        # TODO define shape based on domain and dtype based on type inference
        shape = [10]
        dtype = dace.float64
        sdfg.add_array(name, shape, dtype, transient=False)
        return

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
            temp_state = sdfg.add_state("init_symbols_for_temporaries")
            sdfg.add_edge(entry_state, temp_state, dace.InterstateEdge(assignments=temp_symbols))

            exit_state = sdfg.add_state_after(temp_state, "program_exit")
        else:
            exit_state = sdfg.add_state_after(entry_state, "program_exit")

        # add global arrays (aka non-transient) to the SDFG
        for param, type_ in zip(node.params, self._param_types):
            self._add_storage(sdfg, str(param.id), type_)

        # create root context with exit state
        root_ctx = TaskgenContext(sdfg, exit_state)
        self._ctx_stack.append(root_ctx)

        # visit one statement at a time and put it into separate state
        for i, stmt in enumerate(node.body):
            stmt_state = sdfg.add_state_before(exit_state, f"stmt_{i}")
            stmt_ctx = TaskgenContext(sdfg, stmt_state)
            self._ctx_stack.append(stmt_ctx)
            self.visit(stmt)
            self._ctx_stack.pop()

        assert len(self._ctx_stack) == 1
        assert self._ctx_stack[-1] == root_ctx

        sdfg.validate()
        return sdfg

    def visit_SetAt(self, stmt: itir.SetAt) -> None:
        """Visits a statement expression and writes the local result to some external storage.

        Each statement expression results in some sort of taskgraph writing to local (aka transient) storage.
        The translation of `SetAt` ensures that the result is written to the external storage.
        """

        assert len(self._ctx_stack) > 0
        ctx = self._ctx_stack[-1]

        # the statement expression will result in a tasklet writing to one or more local data nodes
        self.visit(stmt.expr)

        # sanity check on stack status
        assert ctx == self._ctx_stack[-1]

        # reset the list of visited symrefs to only discover output symrefs
        tasklet_symrefs = ctx.symrefs.copy()
        ctx.symrefs.clear()

        # the statement target will result in one or more access nodes to external data
        self.visit(stmt.target)
        target_symrefs = ctx.symrefs.copy()

        # sanity check on stack status
        assert ctx == self._ctx_stack[-1]

        assert len(tasklet_symrefs) == len(target_symrefs)
        for tasklet_sym, target_sym in zip(tasklet_symrefs, target_symrefs):
            target_array = ctx.sdfg.arrays[target_sym]
            assert not target_array.transient

            # TODO: visit statement domain to define the memlet subset
            ctx.state.add_nedge(
                ctx.node_mapping[tasklet_sym],
                ctx.node_mapping[target_sym],
                dace.Memlet.from_array(target_sym, target_array),
            )

    def _make_fieldop(
        self, fun_node: itir.FunCall, fun_args: List[itir.Expr]
    ) -> Sequence[Tuple[str, dace.nodes.AccessNode]]:
        assert len(self._ctx_stack) != 0
        prev_ctx = self._ctx_stack[-1]
        ctx = prev_ctx.clone()
        self._ctx_stack.append(ctx)

        self.visit(fun_args)

        # create ordered list of input nodes
        input_arrays = [(name, ctx.sdfg.arrays[name]) for name in ctx.symrefs]

        # TODO: define shape based on domain and dtype based on type inference
        shape = [10]
        dtype = dace.float64
        output_name, output_array = ctx.sdfg.add_array(
            ctx.var_name(), shape, dtype, transient=True, find_new_name=True
        )
        output_arrays = [(output_name, output_array)]

        assert len(fun_node.args) == 1
        tletgen = ItirToTasklet()
        tlet_code, tlet_inputs, tlet_outputs = tletgen.visit(fun_node.args[0])

        # TODO: define map range based on domain
        map_ranges = dict(i="0:10")

        input_memlets: dict[str, dace.Memlet] = {}
        for connector, (dname, _) in zip(tlet_inputs, input_arrays):
            # TODO: define memlet subset based on domain
            input_memlets[connector] = dace.Memlet(data=dname, subset="i")

        output_memlets: dict[str, dace.Memlet] = {}
        output_nodes: list[Tuple[str, dace.nodes.AccessNode]] = []
        for connector, (dname, _) in zip(tlet_outputs, output_arrays):
            # TODO: define memlet subset based on domain
            output_memlets[connector] = dace.Memlet(data=dname, subset="i")
            output_nodes.append((dname, ctx.add_node(dname)))

        ctx.state.add_mapped_tasklet(
            ctx.tasklet_name(),
            map_ranges,
            input_memlets,
            tlet_code,
            output_memlets,
            input_nodes=ctx.node_mapping,
            output_nodes=ctx.node_mapping,
            external_edges=True,
        )

        self._ctx_stack.pop()
        assert prev_ctx == self._ctx_stack[-1]

        return output_nodes

    def visit_FunCall(self, node: itir.FunCall) -> None:
        assert len(self._ctx_stack) > 0
        ctx = self._ctx_stack[-1]

        if isinstance(node.fun, itir.FunCall) and isinstance(node.fun.fun, itir.SymRef):
            if node.fun.fun.id == "as_fieldop":
                arg_nodes = self._make_fieldop(node.fun, node.args)
                ctx.symrefs.extend([dname for dname, _ in arg_nodes])
            else:
                raise NotImplementedError(f"Unexpected 'FunCall' with function {node.fun.fun.id}.")
        else:
            raise NotImplementedError(f"Unexpected 'FunCall' with type {type(node.fun)}.")

    def visit_SymRef(self, node: itir.SymRef) -> None:
        dname = str(node.id)
        ctx = self._ctx_stack[-1]
        ctx.add_node(dname)
