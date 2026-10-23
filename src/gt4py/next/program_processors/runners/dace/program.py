# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import collections
import dataclasses
import typing
from typing import Any, ClassVar, Optional, Sequence

import dace
import numpy as np

from gt4py.next import backend as gtx_backend, common as gtx_common
from gt4py.next.ffront import decorator
from gt4py.next.iterator import ir as itir, transforms as itir_transforms
from gt4py.next.iterator.transforms import extractors as extractors
from gt4py.next.otf import arguments, recipes, workflow
from gt4py.next.program_processors.runners.dace import sdfg_args as gtx_dace_args
from gt4py.next.program_processors.runners.dace.workflow import translation as gtx_dace_translation
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

        offset_provider: gtx_common.OffsetProvider = self.compilation_options.connectivities or {}
        column_axis = kwargs.get("column_axis", None)

        # TODO(ricoh): connectivity tables required here for now.
        gtir_stage = typing.cast(gtx_backend.Transforms, self.backend.frontend).past_to_itir(
            workflow.ProgramWithArgs(
                definition=self.past_stage,
                args=arguments.CompileTimeArgs(
                    args=tuple(p.type for p in self.past_stage.past_node.params),
                    kwargs={},
                    column_axis=column_axis,
                    offset_provider=offset_provider,
                    argument_descriptor_contexts={},
                ),
            )
        )
        # Run the field-view transforms separately: they need the runtime
        # connectivity tables, which the SDFG itself must not capture. The
        # translation step then receives an already-transformed program whose
        # args only carry the offset-provider *types*.
        # TODO(ricoh): remove this workaround as soon as the mandatory GTIR
        #   passes do not need the connectivity tables anymore.
        program = itir_transforms.apply_fieldview_transforms(
            gtir_stage.definition, offset_provider=offset_provider
        )
        aot_args = dataclasses.replace(
            gtir_stage.args,
            # `CompileTimeArgs.offset_provider` is still typed as the runtime
            # mapping, even though the mandatory GTIR passes are the only
            # consumers needing the tables and they already ran above.
            offset_provider=gtir_stage.args.offset_provider_type,  # type: ignore[arg-type]
        )

        _crosscheck_dace_parsing(
            dace_parsed_args=[*args, *kwargs.values()],
            gt4py_program_args=[p.type for p in program.params],
        )

        sdfg = dace.SDFG.from_json(
            _translation_only_toolchain(self.backend).translate(program, aot_args).source_code
        )

        self.sdfg_closure_cache["arrays"] = sdfg.arrays

        # Halo exchange related metadata, i.e. gt4py_program_input_fields, gt4py_program_output_fields,
        # offset_providers_per_input_field. Add them as dynamic attributes to the SDFG
        field_params = {
            str(param.id): param for param in program.params if isinstance(param.type, ts.FieldType)
        }

        def single_horizontal_dim_per_field(
            fields: typing.Iterable[itir.Sym],
        ) -> typing.Iterator[tuple[str, gtx_common.Dimension]]:
            for field in fields:
                assert isinstance(field.type, ts.FieldType)
                horizontal_dims = [
                    dim
                    for dim in field.type.dims
                    if dim.kind is gtx_common.DimensionKind.HORIZONTAL
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
        if not self.compilation_options.connectivities:
            return {}

        used_connectivities: dict[str, gtx_common.NeighborTable] = {
            conn_id: conn
            for offset, conn in self.compilation_options.connectivities.items()
            if gtx_common.is_neighbor_table(conn)
            and (conn_id := gtx_dace_args.connectivity_identifier(offset))
            in self.sdfg_closure_cache["arrays"]
        }

        # Define the storage location (e.g. CPU, GPU) of the connectivity tables
        if "storage" not in self.connectivity_tables_data_descriptors:
            for conn_id in used_connectivities.keys():
                self.connectivity_tables_data_descriptors["storage"] = self.sdfg_closure_cache[
                    "arrays"
                ][conn_id].storage
                break

        # Build the closure dictionary
        closure_dict: dict[str, dace.data.Array] = {}
        offset_provider_type = gtx_common.offset_provider_to_type(
            self.compilation_options.connectivities
        )
        for conn_id, conn in used_connectivities.items():
            if conn_id not in self.connectivity_tables_data_descriptors:
                self.connectivity_tables_data_descriptors[conn_id] = dace.data.Array(
                    dtype=dace.dtypes.dtype_to_typeclass(conn.dtype.dtype.type),
                    shape=[
                        gtx_dace_args.field_size_symbol(
                            conn_id, conn.domain.dims[0], offset_provider_type
                        ),
                        gtx_dace_args.field_size_symbol(
                            conn_id, conn.domain.dims[1], offset_provider_type
                        ),
                    ],
                    strides=[
                        gtx_dace_args.field_stride_symbol(
                            conn_id, conn.domain.dims[0], offset_provider_type
                        ),
                        gtx_dace_args.field_stride_symbol(
                            conn_id, conn.domain.dims[1], offset_provider_type
                        ),
                    ],
                    storage=Program.connectivity_tables_data_descriptors["storage"],
                )
            closure_dict[conn_id] = self.connectivity_tables_data_descriptors[conn_id]

        return closure_dict

    def __sdfg_signature__(self) -> tuple[Sequence[str], Sequence[str]]:
        return [p.id for p in self.past_stage.past_node.params], []


def _translation_only_toolchain(backend: gtx_backend.Toolchain) -> gtx_backend.Toolchain:
    """
    Derive the translate-only toolchain variant used for SDFG conversion.

    The variant reuses the translation settings of `backend` but disables the
    ITIR transforms (the caller runs them separately, with the runtime
    connectivity tables), the field origins and the metrics instrumentation,
    and bypasses the persistent translation cache.

    Args:
        backend: The dace toolchain the program is bound to.

    Returns:
        A toolchain whose `translate` produces an SDFG source for an
        already-transformed program.

    Raises:
        NotImplementedError: If `backend` is not shaped like the standard dace
            toolchain, i.e. an 'OTFCompileWorkflow' whose translation step is a
            'DaCeTranslator' (optionally wrapped in a 'CachedStep').
    """
    pipeline = backend.backend
    if not isinstance(pipeline, recipes.OTFCompileWorkflow):
        raise NotImplementedError(
            f"Toolchain '{backend.name}' cannot be converted to an SDFG: SDFG"
            " conversion requires the standard 'OTFCompileWorkflow' compile"
            f" pipeline, but this toolchain's backend is a '{type(pipeline).__name__}'."
        )
    translation: workflow.Workflow[Any, Any] = pipeline.translation
    if isinstance(translation, workflow.CachedStep):
        # The persistent translation cache is keyed on the untransformed program,
        # so it must not see the pre-transformed one handed to this variant.
        translation = translation.step
    if not isinstance(translation, gtx_dace_translation.DaCeTranslator):
        raise NotImplementedError(
            f"Toolchain '{backend.name}' cannot be converted to an SDFG: SDFG"
            " conversion requires a 'DaCeTranslator' translation step, but this"
            f" toolchain's is a '{type(translation).__name__}'."
        )
    return dataclasses.replace(
        backend,
        backend=dataclasses.replace(
            pipeline,
            translation=dataclasses.replace(
                translation,
                disable_itir_transforms=True,
                disable_field_origin_on_program_arguments=True,
                use_metrics=False,
            ),
        ),
    )


def _crosscheck_dace_parsing(dace_parsed_args: list[Any], gt4py_program_args: list[Any]) -> None:
    for dace_parsed_arg, gt4py_program_arg in zip(
        dace_parsed_args,
        gt4py_program_args,
        strict=False,  # dace does not see implicit size args
    ):
        match dace_parsed_arg:
            case dace.data.Scalar():
                assert dace_parsed_arg.dtype == gtx_dace_args.as_dace_type(gt4py_program_arg)
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
                assert dace_parsed_arg.dtype == gtx_dace_args.as_dace_type(gt4py_program_arg.dtype)
            case dace.data.Structure() | dict() | collections.OrderedDict():
                # offset provider
                pass
            case _:
                raise ValueError(
                    f"Unresolved case for {dace_parsed_arg} (==, !=) {gt4py_program_arg}"
                )
