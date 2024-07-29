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
import dataclasses
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import field
from typing import Any, ClassVar, Optional

import dace
import numpy as np

import gt4py.next.iterator.ir as itir
from gt4py.next import common
from gt4py.next.ffront import decorator
from gt4py.next.iterator import transforms as itir_transforms
from gt4py.next.type_system import type_specifications as ts

from .utility import (
    as_dace_type,
    connectivity_identifier,
    field_size_symbol_name,
    field_stride_symbol_name,
)


@dataclasses.dataclass(frozen=True)
class Program(decorator.Program, dace.frontend.python.common.SDFGConvertible):
    """Extension of GT4Py Program implementing the SDFGConvertible interface."""

    sdfg_closure_vars: dict[str, Any] = field(default_factory=dict)

    # Being a ClassVar ensures that in an SDFG with multiple nested GT4Py Programs,
    # there is no name mangling of the connectivity tables used across the nested SDFGs
    # since they share the same memory address.
    connectivity_tables_data_descriptors: ClassVar[
        dict[str, dace.data.Array]
    ] = {}  # symbolically defined

    def __sdfg__(self, *args, **kwargs) -> dace.sdfg.sdfg.SDFG:
        if "dace" not in self.backend.executor.name.lower():  # type: ignore[union-attr]
            raise ValueError("The SDFG can be generated only for the DaCe backend.")

        params = {str(p.id): p.type for p in self.itir.params}
        fields = {str(p.id): p.type for p in self.itir.params if hasattr(p.type, "dims")}
        arg_types = [*params.values()]

        dace_parsed_args = [*args, *kwargs.values()]
        gt4py_program_args = [*params.values()]
        _crosscheck_dace_parsing(dace_parsed_args, gt4py_program_args)

        if self.connectivities is None:
            raise ValueError(
                "[DaCe Orchestration] Connectivities -at compile time- are required to generate the SDFG. Use `with_connectivities` method."
            )
        offset_provider = self.connectivities  # tables are None at this point

        sdfg = self.backend.executor.otf_workflow.step.translation.generate_sdfg(  # type: ignore[union-attr]
            self.itir,
            arg_types,
            offset_provider=offset_provider,
            column_axis=kwargs.get("column_axis", None),
        )
        self.sdfg_closure_vars["sdfg.arrays"] = sdfg.arrays  # use it in __sdfg_closure__

        # Halo exchange related metadata, i.e. gt4py_program_input_fields, gt4py_program_output_fields, offset_providers_per_input_field
        # Add them as dynamic properties to the SDFG

        input_fields = [
            str(in_field.id)
            for closure in self.itir.closures
            for in_field in closure.inputs
            if str(in_field.id) in fields
        ]
        sdfg.gt4py_program_input_fields = {
            in_field: dim
            for in_field in input_fields
            for dim in fields[in_field].dims  # type: ignore[union-attr]
            if dim.kind == common.DimensionKind.HORIZONTAL
        }

        output_fields = []
        for closure in self.itir.closures:
            output = closure.output
            if isinstance(output, itir.SymRef):
                if str(output.id) in fields:
                    output_fields.append(str(output.id))
            else:
                for arg in output.args:
                    if str(arg.id) in fields:  # type: ignore[attr-defined]
                        output_fields.append(str(arg.id))  # type: ignore[attr-defined]
        sdfg.gt4py_program_output_fields = {
            output: dim
            for output in output_fields
            for dim in fields[output].dims  # type: ignore[union-attr]
            if dim.kind == common.DimensionKind.HORIZONTAL
        }

        sdfg.offset_providers_per_input_field = {}
        itir_tmp = itir_transforms.apply_common_transforms(
            self.itir, offset_provider=offset_provider
        )
        if not isinstance(itir_tmp, itir.FencilDefinition):
            # FIXME[#1582](kotsaloscv): Update dace orchestration to work with GTIR program
            raise NotImplementedError
        for closure in itir_tmp.closures:
            shifts = itir_transforms.trace_shifts.TraceShifts.apply(closure)
            for k, v in shifts.items():
                if not isinstance(k, str):
                    continue
                if k not in sdfg.gt4py_program_input_fields:
                    continue
                sdfg.offset_providers_per_input_field.setdefault(k, []).extend(list(v))

        return sdfg

    def __sdfg_closure__(self, reevaluate: Optional[dict[str, str]] = None) -> dict[str, Any]:
        """
        Returns the closure arrays of the SDFG represented by this object
        as a mapping between array name and the corresponding value.

        The connectivity tables are defined symbolically, i.e. table sizes & strides are DaCe symbols.
        The need to define the connectivity tables in the `__sdfg_closure__` arises from the fact that
        the offset providers are not part of GT4Py Program's arguments.
        Keep in mind, that `__sdfg_closure__` is called after `__sdfg__` method.
        """
        offset_provider = self.connectivities

        # Define DaCe symbols
        connectivity_table_size_symbols = {
            field_size_symbol_name(connectivity_identifier(k), axis): dace.symbol(
                field_size_symbol_name(connectivity_identifier(k), axis)
            )
            for k, v in offset_provider.items()  # type: ignore[union-attr]
            for axis in [0, 1]
            if hasattr(v, "table")
            and connectivity_identifier(k) in self.sdfg_closure_vars["sdfg.arrays"]
        }

        connectivity_table_stride_symbols = {
            field_stride_symbol_name(connectivity_identifier(k), axis): dace.symbol(
                field_stride_symbol_name(connectivity_identifier(k), axis)
            )
            for k, v in offset_provider.items()  # type: ignore[union-attr]
            for axis in [0, 1]
            if hasattr(v, "table")
            and connectivity_identifier(k) in self.sdfg_closure_vars["sdfg.arrays"]
        }

        symbols = {**connectivity_table_size_symbols, **connectivity_table_stride_symbols}

        # Define the storage location (e.g. CPU, GPU) of the connectivity tables
        if "storage" not in Program.connectivity_tables_data_descriptors:
            for k, v in offset_provider.items():  # type: ignore[union-attr]
                if not hasattr(v, "table"):
                    continue
                if connectivity_identifier(k) in self.sdfg_closure_vars["sdfg.arrays"]:
                    Program.connectivity_tables_data_descriptors["storage"] = (
                        self.sdfg_closure_vars["sdfg.arrays"][connectivity_identifier(k)].storage
                    )
                    break

        # Build the closure dictionary
        closure_dict = {}
        for k, v in offset_provider.items():  # type: ignore[union-attr]
            conn_id = connectivity_identifier(k)
            if hasattr(v, "table") and conn_id in self.sdfg_closure_vars["sdfg.arrays"]:
                if conn_id not in Program.connectivity_tables_data_descriptors:
                    Program.connectivity_tables_data_descriptors[conn_id] = dace.data.Array(
                        dtype=dace.int64 if v.index_type == np.int64 else dace.int32,
                        shape=[
                            symbols[field_size_symbol_name(conn_id, 0)],
                            symbols[field_size_symbol_name(conn_id, 1)],
                        ],
                        strides=[
                            symbols[field_stride_symbol_name(conn_id, 0)],
                            symbols[field_stride_symbol_name(conn_id, 1)],
                        ],
                        storage=Program.connectivity_tables_data_descriptors["storage"],
                    )
                closure_dict[conn_id] = Program.connectivity_tables_data_descriptors[conn_id]

        return closure_dict

    def __sdfg_signature__(self) -> tuple[Sequence[str], Sequence[str]]:
        args = []
        for arg in self.past_stage.past_node.params:
            args.append(arg.id)
        return (args, [])


def _crosscheck_dace_parsing(dace_parsed_args: list[Any], gt4py_program_args: list[Any]) -> bool:
    for dace_parsed_arg, gt4py_program_arg in zip(dace_parsed_args, gt4py_program_args):
        if isinstance(dace_parsed_arg, dace.data.Scalar):
            assert dace_parsed_arg.dtype == as_dace_type(gt4py_program_arg)
        elif isinstance(
            dace_parsed_arg, (bool, int, float, str, np.bool_, np.integer, np.floating, np.str_)
        ):  # compile-time constant scalar
            assert isinstance(gt4py_program_arg, ts.ScalarType)
            if isinstance(dace_parsed_arg, (bool, np.bool_)):
                assert gt4py_program_arg.kind == ts.ScalarKind.BOOL
            elif isinstance(dace_parsed_arg, (int, np.integer)):
                assert gt4py_program_arg.kind in [ts.ScalarKind.INT32, ts.ScalarKind.INT64]
            elif isinstance(dace_parsed_arg, (float, np.floating)):
                assert gt4py_program_arg.kind in [ts.ScalarKind.FLOAT32, ts.ScalarKind.FLOAT64]
            elif isinstance(dace_parsed_arg, (str, np.str_)):
                assert gt4py_program_arg.kind == ts.ScalarKind.STRING
        elif isinstance(dace_parsed_arg, dace.data.Array):
            assert isinstance(gt4py_program_arg, ts.FieldType)
            assert len(dace_parsed_arg.shape) == len(gt4py_program_arg.dims)
            assert dace_parsed_arg.dtype == as_dace_type(gt4py_program_arg.dtype)
        elif isinstance(
            dace_parsed_arg, (dace.data.Structure, dict, OrderedDict)
        ):  # offset_provider
            continue
        else:
            raise ValueError(f"Unresolved case for {dace_parsed_arg} (==, !=) {gt4py_program_arg}")

    return True
