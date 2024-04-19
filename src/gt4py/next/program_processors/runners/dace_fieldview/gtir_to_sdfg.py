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

from typing import Any, Callable, Dict, List, Optional

import dace

from gt4py import eve
from gt4py.next.iterator import ir as itir
from gt4py.next.type_system import type_specifications as ts

from .fieldview_dataflow import FieldviewRegion
from .gtir_to_tasklet import GtirToTasklet


def create_ctx_in_new_state(new_state_name: Optional[str] = None) -> Callable:
    """Decorator to execute the visit function in a separate context, in a new state."""

    def decorator(func: Callable) -> Callable:
        def newf(self: "GtirToSDFG", *args: Any, **kwargs: Optional[Any]) -> FieldviewRegion:
            prev_ctx = self._ctx
            assert prev_ctx is not None
            new_ctx = prev_ctx.clone()
            if new_state_name:
                new_ctx.state = prev_ctx.sdfg.add_state_after(prev_ctx.state, new_state_name)
            self._ctx = new_ctx

            child_ctx = func(self, *args, **kwargs)

            assert self._ctx == new_ctx
            self._ctx = prev_ctx

            return child_ctx

        return newf

    return decorator


def create_ctx(func: Callable) -> Callable:
    """Decorator to execute the visit function in a separate context, in current state."""

    return create_ctx_in_new_state()(func)


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

    _ctx: Optional[FieldviewRegion]
    _param_types: list[ts.TypeSpec]

    def __init__(
        self,
        param_types: list[ts.TypeSpec],
    ):
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
        exit_state = sdfg.add_state_after(entry_state, "program_exit")

        # declarations of temporaries result in local (aka transient) array definitions in the SDFG
        if node.declarations:
            temp_symbols: dict[str, str] = {}
            for decl in node.declarations:
                temp_symbols |= self._add_storage_for_temporary(decl)

            # define symbols for shape and offsets of temporary arrays as interstate edge symbols
            # TODO(edopao): use new `add_state_after` function in next dace release
            head_state = sdfg.add_state_before(exit_state, "init_symbols_for_temporaries")
            (sdfg.edges_between(entry_state, head_state))[0].assignments = temp_symbols
        else:
            head_state = entry_state

        # add global arrays (aka non-transient) to the SDFG
        for param, type_ in zip(node.params, self._param_types):
            self._add_storage(sdfg, str(param.id), type_)

        self._ctx = FieldviewRegion(sdfg, head_state)
        # visit one statement at a time and put it into separate state
        for stmt in node.body:
            self.visit(stmt)

        assert self._ctx.state == head_state
        self._ctx = None

        sdfg.validate()
        return sdfg

    @create_ctx_in_new_state(new_state_name="set_at")
    def visit_SetAt(self, stmt: itir.SetAt) -> None:
        """Visits a statement expression and writes the local result to some external storage.

        Each statement expression results in some sort of taskgraph writing to local (aka transient) storage.
        The translation of `SetAt` ensures that the result is written to the external storage.
        """

        stmt_ctx = self._ctx
        assert stmt_ctx is not None

        self.visit(stmt.expr)

        # the target expression could be a `SymRef` to an output node or a `make_tuple` expression
        # in case the statement returns more than one field
        # TODO: Use GtirToTasklet with new context without updating self._ctx
        self._ctx = stmt_ctx.clone()
        self.visit(stmt.target)
        # the visit of a target expression should only produce a set of access nodes (no tasklets, no output nodes)
        assert len(self._ctx.output_nodes) == 0
        stmt_ctx.output_nodes.extend(self._ctx.input_nodes)
        self._ctx = stmt_ctx

        assert len(stmt_ctx.input_nodes) == len(stmt_ctx.output_nodes)
        for tasklet_node, target_node in zip(stmt_ctx.input_nodes, stmt_ctx.output_nodes):
            target_array = stmt_ctx.sdfg.arrays[target_node]
            target_array.transient = False

            # TODO: visit statement domain to define the memlet subset
            stmt_ctx.state.add_nedge(
                stmt_ctx.node_mapping[tasklet_node],
                stmt_ctx.node_mapping[target_node],
                dace.Memlet.from_array(target_node, target_array),
            )

    @create_ctx
    def _make_fieldop(self, fun_node: itir.FunCall, fun_args: List[itir.Expr]) -> FieldviewRegion:
        ctx = self._ctx
        assert ctx is not None

        self.visit(fun_args)

        # create ordered list of input nodes
        input_arrays = [(name, ctx.sdfg.arrays[name]) for name in ctx.input_nodes]

        # TODO: define shape based on domain and dtype based on type inference
        shape = [10]
        dtype = dace.float64
        output_name, output_array = ctx.sdfg.add_array(
            ctx.var_name(), shape, dtype, transient=True, find_new_name=True
        )
        output_arrays = [(output_name, output_array)]

        assert len(fun_node.args) == 1
        assert isinstance(fun_node.args[0], itir.Lambda)

        tletgen = GtirToTasklet(ctx)
        tlet_code, tlet_inputs, tlet_outputs = tletgen.visit(fun_node.args[0])

        # TODO: define map range based on domain
        map_ranges = dict(i="0:10")

        input_memlets: dict[str, dace.Memlet] = {}
        for connector, (dname, _) in zip(tlet_inputs, input_arrays):
            # TODO: define memlet subset based on domain
            input_memlets[connector] = dace.Memlet(data=dname, subset="i")

        output_memlets: dict[str, dace.Memlet] = {}
        for connector, (dname, _) in zip(tlet_outputs, output_arrays):
            # TODO: define memlet subset based on domain
            output_memlets[connector] = dace.Memlet(data=dname, subset="i")
            ctx.add_output_node(dname)

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

        return ctx

    def visit_FunCall(self, node: itir.FunCall) -> None:
        assert self._ctx is not None
        if isinstance(node.fun, itir.FunCall) and isinstance(node.fun.fun, itir.SymRef):
            if node.fun.fun.id == "as_fieldop":
                child_ctx = self._make_fieldop(node.fun, node.args)
                assert child_ctx.state == self._ctx.state
                self._ctx.input_nodes.extend(child_ctx.output_nodes)
            else:
                raise NotImplementedError(f"Unexpected 'FunCall' with function {node.fun.fun.id}.")
        else:
            raise NotImplementedError(f"Unexpected 'FunCall' with type {type(node.fun)}.")

    def visit_SymRef(self, node: itir.SymRef) -> None:
        dname = str(node.id)
        assert self._ctx is not None
        self._ctx.add_input_node(dname)
