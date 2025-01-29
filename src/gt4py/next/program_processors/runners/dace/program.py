# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import collections
import dataclasses
import itertools
import typing
from typing import Any, ClassVar, Optional, Sequence

import dace
import numpy as np

from gt4py.next import backend as next_backend, common
from gt4py.next.ffront import decorator
from gt4py.next.iterator import ir as itir, transforms as itir_transforms
from gt4py.next.iterator.transforms import extractors as extractors
from gt4py.next.otf import arguments, recipes, toolchain
from gt4py.next.program_processors.runners.dace import utils as gtx_dace_utils
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

        offset_provider: common.OffsetProvider = {
            **(self.connectivities or {}),
            **self._implicit_offset_provider,
        }
        column_axis = kwargs.get("column_axis", None)

        # TODO(ricoh): connectivity tables required here for now.
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
        program = gtir_stage.data
        program = itir_transforms.apply_fieldview_transforms(  # run the transforms separately because they require the runtime info
            program, offset_provider=offset_provider
        )
        object.__setattr__(
            gtir_stage,
            "data",
            program,
        )
        object.__setattr__(
            gtir_stage.args, "offset_provider", gtir_stage.args.offset_provider_type
        )  # TODO(ricoh): currently this is circumventing the frozenness of CompileTimeArgs
        # in order to isolate DaCe from the runtime tables in connectivities.offset_provider.
        # These are needed at the time of writing for mandatory GTIR passes.
        # Remove this as soon as Program does not expect connectivity tables anymore.

        _crosscheck_dace_parsing(
            dace_parsed_args=[*args, *kwargs.values()],
            gt4py_program_args=[p.type for p in program.params],
        )

        compile_workflow = typing.cast(
            recipes.OTFCompileWorkflow,
            self.backend.executor
            if not hasattr(self.backend.executor, "step")
            else self.backend.executor.step,
        )  # We know which backend we are using, but we don't know if the compile workflow is cached.
        # TODO(ricoh): switch 'itir_transforms_off=True' because we ran them separately previously
        # and so we can ensure the SDFG does not know any runtime info it shouldn't know. Remove with
        # the other parts of the workaround when possible.
        sdfg = dace.SDFG.from_json(
            compile_workflow.translation.replace(itir_transforms_off=True)(gtir_stage).source_code
        )

        self.sdfg_closure_cache["arrays"] = sdfg.arrays

        # Halo exchange related metadata, i.e. gt4py_program_input_fields, gt4py_program_output_fields,
        # offset_providers_per_input_field. Add them as dynamic attributes to the SDFG
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

        input_fields = (
            field_params[name] for name in extractors.InputNamesExtractor.only_fields(program)
        )
        sdfg.gt4py_program_input_fields = dict(single_horizontal_dim_per_field(input_fields))

        output_fields = (
            field_params[name] for name in extractors.OutputNamesExtractor.only_fields(program)
        )
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
                name for name, conn in self.connectivities.items() if common.is_neighbor_table(conn)
            ]
            in_arrays_with_id = [
                (name, conn_id)
                for name in with_table
                if (conn_id := gtx_dace_utils.connectivity_identifier(name))
                in self.sdfg_closure_cache["arrays"]
            ]
            in_arrays = (name for name, _ in in_arrays_with_id)
            name_axis = list(itertools.product(in_arrays, [0, 1]))

            def size_symbol_name(name: str, axis: int) -> str:
                return gtx_dace_utils.field_size_symbol_name(
                    gtx_dace_utils.connectivity_identifier(name), axis
                )

            connectivity_tables_size_symbols = {
                (sname := size_symbol_name(name, axis)): dace.symbol(sname)
                for name, axis in name_axis
            }

            def stride_symbol_name(name: str, axis: int) -> str:
                return gtx_dace_utils.field_stride_symbol_name(
                    gtx_dace_utils.connectivity_identifier(name), axis
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
                    assert common.is_neighbor_table(conn)
                    self.connectivity_tables_data_descriptors[conn_id] = dace.data.Array(
                        dtype=dace.dtypes.dtype_to_typeclass(conn.dtype.dtype.type),
                        shape=[
                            symbols[gtx_dace_utils.field_size_symbol_name(conn_id, 0)],
                            symbols[gtx_dace_utils.field_size_symbol_name(conn_id, 1)],
                        ],
                        strides=[
                            symbols[gtx_dace_utils.field_stride_symbol_name(conn_id, 0)],
                            symbols[gtx_dace_utils.field_stride_symbol_name(conn_id, 1)],
                        ],
                        storage=Program.connectivity_tables_data_descriptors["storage"],
                    )
                closure_dict[conn_id] = self.connectivity_tables_data_descriptors[conn_id]

        return closure_dict

    def __sdfg_signature__(self) -> tuple[Sequence[str], Sequence[str]]:
        return [p.id for p in self.past_stage.past_node.params], []


def _crosscheck_dace_parsing(dace_parsed_args: list[Any], gt4py_program_args: list[Any]) -> None:
    for dace_parsed_arg, gt4py_program_arg in zip(
        dace_parsed_args,
        gt4py_program_args,
        strict=False,  # dace does not see implicit size args
    ):
        match dace_parsed_arg:
            case dace.data.Scalar():
                assert dace_parsed_arg.dtype == gtx_dace_utils.as_dace_type(gt4py_program_arg)
            case bool() | np.bool_():
                assert isinstance(gt4py_program_arg, ts.ScalarType)
                assert gt4py_program_arg.kind == ts.ScalarKind.BOOL
            case int() | np.integer():
                assert isinstance(gt4py_program_arg, ts.ScalarType)
                assert gt4py_program_arg.kind in [ts.ScalarKind.INT32, ts.ScalarKind.INT64]
            case float() | np.floating():
                assert isinstance(gt4py_program_arg, ts.ScalarType)
                assert gt4py_program_arg.kind in [ts.ScalarKind.FLOAT32, ts.ScalarKind.FLOAT64]
            case str() | np.str_():
                assert isinstance(gt4py_program_arg, ts.ScalarType)
                assert gt4py_program_arg.kind == ts.ScalarKind.STRING
            case dace.data.Array():
                assert isinstance(gt4py_program_arg, ts.FieldType)
                assert isinstance(gt4py_program_arg.dtype, ts.ScalarType)
                assert len(dace_parsed_arg.shape) == len(gt4py_program_arg.dims)
                assert dace_parsed_arg.dtype == gtx_dace_utils.as_dace_type(gt4py_program_arg.dtype)
            case dace.data.Structure() | dict() | collections.OrderedDict():
                # offset provider
                pass
            case _:
                raise ValueError(
                    f"Unresolved case for {dace_parsed_arg} (==, !=) {gt4py_program_arg}"
                )
