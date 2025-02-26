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
    rewritten_args, size_args, kwargs = _process_args(
        past_node=inp.data.past_node, args=inp.args.args, kwargs=inp.args.kwargs
    )
    return toolchain.CompilableProgram(
        data=inp.data,
        args=arguments.CompileTimeArgs(
            args=tuple((*rewritten_args, *(size_args))),
            kwargs=kwargs,
            offset_provider=inp.args.offset_provider,
            column_axis=inp.args.column_axis,
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
    past_node: past.Program, args: Sequence[ts.TypeSpec], kwargs: dict[str, ts.TypeSpec]
) -> tuple[tuple, tuple, dict[str, Any]]:
    if not isinstance(past_node.type, ts_ffront.ProgramType):
        raise TypeError("Can not process arguments for PAST programs prior to type inference.")

    _validate_args(past_node=past_node, arg_types=args, kwarg_types=kwargs)

    args, kwargs = type_info.canonicalize_arguments(past_node.type, args, kwargs)

    implicit_domain = any(
        isinstance(stmt, past.Call) and "domain" not in stmt.kwargs for stmt in past_node.body
    )

    # extract size of all field arguments
    size_args: list[ts.TypeSpec] = []
    rewritten_args = list(args)
    for param_idx, param in enumerate(past_node.params):
        if implicit_domain and isinstance(param.type, (ts.FieldType, ts.TupleType)):
            # TODO(tehrengruber): Previously this function was called with the actual arguments
            #  not their type. The check using the shape here is not functional anymore and
            #  should instead be placed in a proper location.
            ranges_and_dims = [*_field_constituents_range_and_dims(args[param_idx], param.type)]
            # check that all non-scalar like constituents have the same shape and dimension, e.g.
            # for `(scalar, (field1, field2))` the two fields need to have the same shape and
            # dimension
            if ranges_and_dims:
                range_, dims = ranges_and_dims[0]
                if not all(
                    el_range == range_ and el_dims == dims
                    for (el_range, el_dims) in ranges_and_dims
                ):
                    raise ValueError(
                        "Constituents of composite arguments (e.g. the elements of a"
                        " tuple) need to have the same shape and dimensions."
                    )
                index_type = ts.ScalarType(kind=ts.ScalarKind.INT32)
                size_args.extend(
                    range_ if range_ else [ts.TupleType(types=[index_type, index_type])] * len(dims)  # type: ignore[arg-type]  # shape is always empty
                )
    return tuple(rewritten_args), tuple(size_args), kwargs


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
