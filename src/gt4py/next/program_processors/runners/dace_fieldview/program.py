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
from typing import Any, Optional, Sequence

import dace

from gt4py._core import definitions as core_defs
from gt4py.next import allocators, backend as next_backend
from gt4py.next.ffront import decorator
from gt4py.next.otf import arguments, toolchain
from gt4py.next.program_processors.runners.dace_common import utility as dace_utils


@dataclasses.dataclass(frozen=True)
class Program(decorator.Program, dace.frontend.python.common.SDFGConvertible):
    """Extension of GT4Py Program implementing the SDFGConvertible interface via GTIR."""

    sdfg_closure_cache: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __sdfg__(self, *args: Any, **kwargs: Any) -> dace.sdfg.sdfg.SDFG:
        if (self.backend is None) or "dace" not in self.backend.name.lower():
            raise ValueError("The SDFG can be generated only for the DaCe backend.")

        offset_provider = (self.connectivities or {}) | self._implicit_offset_provider
        column_axis = kwargs.get("column_axis", None)

        gtir_stage = typing.cast(next_backend.Transforms, self.backend.transforms).past_to_itir(
            toolchain.CompilableProgram(
                data=self.past_stage,
                args=arguments.CompileTimeArgs(
                    args=tuple(p.type for p in self.past_stage.past_node.definition.params),
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
            offset_provider=offset_provider,
            column_axis=gtir_stage.args.column_axis,
            auto_opt=True,
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
        symbols = {}

        if self.connectivities:
            with_table = (
                name for name, conn in self.connectivities.items() if hasattr(conn, "table")
            )
            in_arrays = (
                name
                for name in with_table
                if dace_utils.connectivity_identifier(name) in self.sdfg_closure_cache["arrays"]
            )
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

        closure_dict: dict[str, Any] = {}

        return closure_dict

    def __sdfg_signature__(self) -> tuple[Sequence[str], Sequence[str]]:
        return [p.id for p in self.past_stage.past_node.params], []
