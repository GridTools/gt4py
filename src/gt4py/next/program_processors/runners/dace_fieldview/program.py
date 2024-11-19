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

from gt4py._core import definitions as core_defs
from gt4py.next import allocators, backend as next_backend
from gt4py.next.ffront import decorator
from gt4py.next.otf import arguments, toolchain
from gt4py.next.program_processors.runners.dace_common import utility as dace_utils


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

        on_gpu = not allocators.is_field_allocator_for(
            self.backend.allocator, core_defs.DeviceType.CPU
        ) and not allocators.is_field_allocator_for(
            self.backend.allocator, core_defs.DeviceType.CPU_PINNED
        )

        sdfg = self.backend.executor.step.translation.generate_sdfg(  # type: ignore[attr-defined] # we can assume to get a DaCeTranslationStep here
            gtir_stage.data,
            offset_provider=gtir_stage.args.offset_provider,
            column_axis=gtir_stage.args.column_axis,
            auto_opt=False,
            on_gpu=on_gpu,
        )

        self.sdfg_closure_cache["arrays"] = sdfg.arrays

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
