# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Iterator, Sequence, TypeAlias

from gt4py.next import common, errors
from gt4py.next.ffront import (
    program_ast as past,
    stages as ffront_stages,
    type_specifications as ts_ffront,
)
from gt4py.next.otf import arguments, toolchain, workflow
from gt4py.next.type_system import type_info, type_specifications as ts


AOT_PRG: TypeAlias = toolchain.CompilableProgram[
    ffront_stages.PastProgramDefinition, arguments.CompileTimeArgs
]


def transform_program_args(inp: AOT_PRG) -> AOT_PRG:
    rewritten_args, rewritten_kwargs = _process_args(
        past_node=inp.data.past_node, args=inp.args.args, kwargs=inp.args.kwargs
    )
    return toolchain.CompilableProgram(
        data=inp.data,
        args=arguments.CompileTimeArgs(
            args=rewritten_args,
            kwargs=rewritten_kwargs,
            offset_provider=inp.args.offset_provider,
            column_axis=inp.args.column_axis,
            argument_descriptor_contexts=inp.args.argument_descriptor_contexts,
        ),
    )


def transform_program_args_factory(cached: bool = True) -> workflow.Workflow[AOT_PRG, AOT_PRG]:
    wf = transform_program_args
    if cached:
        wf = workflow.CachedStep(wf, hash_function=ffront_stages.fingerprint_stage)
    return wf


def _validate_args(
    past_node: past.Program, arg_types: Sequence[ts.TypeSpec], kwarg_types: dict[str, ts.TypeSpec]
) -> None:
    if not isinstance(past_node.type, ts_ffront.ProgramType):
        raise TypeError("Can not validate arguments for PAST programs prior to type inference.")

    try:
        type_info.accepts_args(
            past_node.type, with_args=arg_types, with_kwargs=kwarg_types, raise_exception=True
        )
    except ValueError as err:
        raise errors.DSLError(
            None, f"Invalid argument types in call to '{past_node.id}'.\n{err}"
        ) from err


def _process_args(
    past_node: past.Program,
    args: Sequence[ts.TypeSpec],
    kwargs: dict[str, ts.TypeSpec],
) -> tuple[tuple[ts.TypeSpec], dict[str, ts.TypeSpec]]:
    if not isinstance(past_node.type, ts_ffront.ProgramType):
        raise TypeError("Can not process arguments for PAST programs prior to type inference.")

    args, kwargs = type_info.canonicalize_arguments(past_node.type, args, kwargs)
    _validate_args(past_node=past_node, arg_types=args, kwarg_types=kwargs)

    return args, kwargs


def _field_constituents_range_and_dims(
    arg: Any,  # TODO(havogt): improve typing
    arg_type: ts.DataType,
) -> Iterator[tuple[tuple[tuple[int, int], ...], list[common.Dimension]]]:
    match arg_type:
        case ts.TupleType():
            for el, el_type in zip(arg, arg_type.types):
                assert isinstance(el_type, ts.DataType)
                yield from _field_constituents_range_and_dims(el, el_type)
        case ts.FieldType():
            dims = type_info.extract_dims(arg_type)
            if isinstance(arg, ts.TypeSpec):  # TODO
                yield (tuple(), dims)
            elif dims:
                assert (
                    hasattr(arg, "domain")
                    and isinstance(arg.domain, common.Domain)
                    and len(arg.domain.dims) == len(dims)
                )
                yield (tuple((r.start, r.stop) for r in arg.domain.ranges), dims)
            else:
                yield from []  # ignore 0-dim fields
        case ts.ScalarType():
            yield from []  # ignore scalars
        case _:
            raise ValueError("Expected 'FieldType' or 'TupleType' thereof.")
