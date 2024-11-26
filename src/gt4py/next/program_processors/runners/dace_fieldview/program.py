# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import itertools
import typing
from typing import Any, ClassVar, Optional, Sequence

import dace
import numpy as np

from gt4py import eve
from gt4py.next import backend as next_backend, common
from gt4py.next.ffront import decorator
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import arguments, recipes, toolchain
from gt4py.next.program_processors.runners.dace_common import utility as dace_utils
from gt4py.next.type_system import type_specifications as ts


@dataclasses.dataclass(frozen=True)
class Program(decorator.Program, dace.frontend.python.common.SDFGConvertible):
    """Extension of GT4Py Program implementing the SDFGConvertible interface via GTIR."""

    sdfg_closure_cache: dict[str, Any] = dataclasses.field(default_factory=dict)
    # Being a ClassVar ensures that in an SDFG with multiple nested GT4Py Programs,
    # there is no name mangling of the connectivity tables used across the nested SDFGs
    # since they share the same memory address.
    connectivity_tables_data_descriptors: ClassVar[
        dict[str, dace.data.Array]
    ] = {}  # symbolically defined

    def __sdfg__(self, *args: Any, **kwargs: Any) -> dace.sdfg.sdfg.SDFG:
        if (self.backend is None) or "dace" not in self.backend.name.lower():
            raise ValueError("The SDFG can be generated only for the DaCe backend.")

        offset_provider = (self.connectivities or {}) | self._implicit_offset_provider
        column_axis = kwargs.get("column_axis", None)

        gtir_stage = typing.cast(next_backend.Transforms, self.backend.transforms).past_to_itir(
            toolchain.CompilableProgram(
                data=self.past_stage,
                args=arguments.CompileTimeArgs(
                    args=tuple(p.type for p in self.past_stage.past_node.params),
                    kwargs={},
                    column_axis=column_axis,
                    offset_provider=offset_provider,
                ),
            )
        )

        compile_workflow = typing.cast(
            recipes.OTFCompileWorkflow,
            self.backend.executor
            if not hasattr(self.backend.executor, "step")
            else self.backend.executor.step,
        )  # We know which backend we are using, but we don't know if the compile workflow is cached.
        sdfg = dace.SDFG.from_json(compile_workflow.translation(gtir_stage).source_code)

        self.sdfg_closure_cache["arrays"] = sdfg.arrays

        # Halo exchange related metadata, i.e. gt4py_program_input_fields, gt4py_program_output_fields, offset_providers_per_input_field
        # Add them as dynamic properties to the SDFG
        program = typing.cast(
            itir.Program, gtir_stage.data
        )  # we already checked that our backend uses GTIR
        field_params = {
            str(param.id): param for param in program.params if isinstance(param.type, ts.FieldType)
        }

        def single_horizontal_dim_per_field(
            fields: typing.Iterable[itir.Sym],
        ) -> typing.Iterator[tuple[str, common.Dimension]]:
            for field in fields:
                assert isinstance(field.type, ts.FieldType)
                horizontal_dims = [
                    dim for dim in field.type.dims if dim.kind is common.DimensionKind.HORIZONTAL
                ]
                # do nothing for fields with multiple horizontal dimensions
                # or without horizontal dimensions
                # this is only meant for use with unstructured grids
                if len(horizontal_dims) == 1:
                    yield str(field.id), horizontal_dims[0]

        input_fields = (field_params[name] for name in InputNamesExtractor.only_fields(program))
        sdfg.gt4py_program_input_fields = dict(single_horizontal_dim_per_field(input_fields))

        output_fields = (field_params[name] for name in OutputNamesExtractor.only_fields(program))
        sdfg.gt4py_program_output_fields = dict(single_horizontal_dim_per_field(output_fields))

        # TODO (ricoh): bring back sdfg.offset_providers_per_input_field.
        #               A starting point would be to use the "trace_shifts" pass on GTIR
        #               and associate the extracted shifts with each input field.
        #               Analogous to the version in `runners.dace_iterator.__init__`, which
        #               was removed when merging #1742.

        return sdfg

    def __sdfg_closure__(self, reevaluate: Optional[dict[str, str]] = None) -> dict[str, Any]:
        """
        Return the closure arrays of the SDFG represented by this object
        as a mapping between array name and the corresponding value.

        The connectivity tables are defined symbolically, i.e. table sizes & strides are DaCe symbols.
        The need to define the connectivity tables in the `__sdfg_closure__` arises from the fact that
        the offset providers are not part of GT4Py Program's arguments.
        Keep in mind, that `__sdfg_closure__` is called after `__sdfg__` method.
        """
        closure_dict: dict[str, Any] = {}

        if self.connectivities:
            symbols = {}
            with_table = [
                name for name, conn in self.connectivities.items() if hasattr(conn, "table")
            ]
            in_arrays_with_id = [
                (name, conn_id)
                for name in with_table
                if (conn_id := dace_utils.connectivity_identifier(name))
                in self.sdfg_closure_cache["arrays"]
            ]
            in_arrays = (name for name, _ in in_arrays_with_id)
            name_axis = list(itertools.product(in_arrays, [0, 1]))

            def size_symbol_name(name: str, axis: int) -> str:
                return dace_utils.field_size_symbol_name(
                    dace_utils.connectivity_identifier(name), axis
                )

            connectivity_tables_size_symbols = {
                (sname := size_symbol_name(name, axis)): dace.symbol(sname)
                for name, axis in name_axis
            }

            def stride_symbol_name(name: str, axis: int) -> str:
                return dace_utils.field_stride_symbol_name(
                    dace_utils.connectivity_identifier(name), axis
                )

            connectivity_table_stride_symbols = {
                (sname := stride_symbol_name(name, axis)): dace.symbol(sname)
                for name, axis in name_axis
            }

            symbols = connectivity_tables_size_symbols | connectivity_table_stride_symbols

            # Define the storage location (e.g. CPU, GPU) of the connectivity tables
            if "storage" not in self.connectivity_tables_data_descriptors:
                for _, conn_id in in_arrays_with_id:
                    self.connectivity_tables_data_descriptors["storage"] = self.sdfg_closure_cache[
                        "arrays"
                    ][conn_id].storage
                    break

            # Build the closure dictionary
            for name, conn_id in in_arrays_with_id:
                if conn_id not in self.connectivity_tables_data_descriptors:
                    conn = self.connectivities[name]
                    self.connectivity_tables_data_descriptors[conn_id] = dace.data.Array(
                        dtype=dace.int64 if conn.index_type == np.int64 else dace.int32,
                        shape=[
                            symbols[dace_utils.field_size_symbol_name(conn_id, 0)],
                            symbols[dace_utils.field_size_symbol_name(conn_id, 1)],
                        ],
                        strides=[
                            symbols[dace_utils.field_stride_symbol_name(conn_id, 0)],
                            symbols[dace_utils.field_stride_symbol_name(conn_id, 1)],
                        ],
                        storage=Program.connectivity_tables_data_descriptors["storage"],
                    )
                closure_dict[conn_id] = self.connectivity_tables_data_descriptors[conn_id]

        return closure_dict

    def __sdfg_signature__(self) -> tuple[Sequence[str], Sequence[str]]:
        return [p.id for p in self.past_stage.past_node.params], []


class SymbolNameSetExtractor(eve.NodeVisitor):
    """Extract a set of symbol names"""

    def generic_visitor(self, node: itir.Node) -> set[str]:
        input_fields: set[str] = set()
        for child in eve.trees.iter_children_values(node):
            input_fields |= self.visit(child)
        return input_fields

    @classmethod
    def only_fields(cls, program: itir.Program) -> set[str]:
        field_param_names = [
            str(param.id) for param in program.params if isinstance(param.type, ts.FieldType)
        ]
        return {name for name in cls().visit(program) if name in field_param_names}


class InputNamesExtractor(SymbolNameSetExtractor):
    """Extract the set of symbol names passed into field operators within a program."""

    def visit_Program(self, node: itir.Program) -> set[str]:
        input_fields = set()
        for stmt in node.body:
            input_fields |= self.visit(stmt)
        return input_fields

    def visit_IfStmt(self, node: itir.IfStmt) -> set[str]:
        input_fields = set()
        for stmt in node.true_branch + node.false_branch:
            input_fields |= self.visit(stmt)
        return input_fields

    def visit_Temporary(self, node: itir.Temporary) -> set[str]:
        return set()

    def visit_SetAt(self, node: itir.SetAt) -> set[str]:
        return self.visit(node.expr)

    def visit_FunCall(self, node: itir.FunCall) -> set[str]:
        input_fields = set()
        for arg in node.args:
            input_fields |= self.visit(arg)
        return input_fields

    def visit_SymRef(self, node: itir.SymRef) -> set[str]:
        return {str(node.id)}


class OutputNamesExtractor(SymbolNameSetExtractor):
    """Extract the set of symbol names written to within a program"""

    def visit_Program(self, node: itir.Program) -> set[str]:
        output_fields = set()
        for stmt in node.body:
            output_fields |= self.visit(stmt)
        return output_fields

    def visit_IfStmt(self, node: itir.IfStmt) -> set[str]:
        output_fields = set()
        for stmt in node.true_branch + node.false_branch:
            output_fields |= self.visit(stmt)
        return output_fields

    def visit_Temporary(self, node: itir.Temporary) -> set[str]:
        return set()

    def visit_SetAt(self, node: itir.SetAt) -> set[str]:
        return self.visit(node.target)

    def visit_SymRef(self, node: itir.SymRef) -> set[str]:
        return {str(node.id)}
