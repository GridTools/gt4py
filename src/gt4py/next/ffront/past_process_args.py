# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Iterator, Optional

from gt4py.next import common, errors
from gt4py.next.ffront import (
    program_ast as past,
    stages as ffront_stages,
    type_specifications as ts_ffront,
)
from gt4py.next.otf import workflow
from gt4py.next.type_system import type_info, type_specifications as ts, type_translation


@workflow.make_step
def past_process_args(inp: ffront_stages.PastClosure) -> ffront_stages.PastClosure:
    extra_kwarg_names = ["offset_provider", "column_axis"]
    extra_kwargs = {k: v for k, v in inp.kwargs.items() if k in extra_kwarg_names}
    kwargs = {k: v for k, v in inp.kwargs.items() if k not in extra_kwarg_names}
    rewritten_args, size_args, kwargs = _process_args(
        past_node=inp.past_node, args=list(inp.args), kwargs=kwargs
    )
    return ffront_stages.PastClosure(
        past_node=inp.past_node,
        closure_vars=inp.closure_vars,
        grid_type=inp.grid_type,
        args=tuple([*rewritten_args, *size_args]),
        kwargs=kwargs | extra_kwargs,
    )


def _validate_args(past_node: past.Program, args: list, kwargs: dict[str, Any]) -> None:
    arg_types = [type_translation.from_value(arg) for arg in args]
    kwarg_types = {k: type_translation.from_value(v) for k, v in kwargs.items()}

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
    past_node: past.Program, args: list, kwargs: dict[str, Any]
) -> tuple[tuple, tuple, dict[str, Any]]:
    if not isinstance(past_node.type, ts_ffront.ProgramType):
        raise TypeError("Can not process arguments for PAST programs prior to type inference.")

    _validate_args(past_node=past_node, args=args, kwargs=kwargs)

    args, kwargs = type_info.canonicalize_arguments(past_node.type, args, kwargs)

    implicit_domain = any(
        isinstance(stmt, past.Call) and "domain" not in stmt.kwargs for stmt in past_node.body
    )

    # extract size of all field arguments
    size_args: list[Optional[int]] = []
    rewritten_args = list(args)
    for param_idx, param in enumerate(past_node.params):
        if implicit_domain and isinstance(param.type, (ts.FieldType, ts.TupleType)):
            shapes_and_dims = [*_field_constituents_shape_and_dims(args[param_idx], param.type)]
            shape, dims = shapes_and_dims[0]
            if not all(
                el_shape == shape and el_dims == dims for (el_shape, el_dims) in shapes_and_dims
            ):
                raise ValueError(
                    "Constituents of composite arguments (e.g. the elements of a"
                    " tuple) need to have the same shape and dimensions."
                )
            size_args.extend(shape if shape else [None] * len(dims))
    return tuple(rewritten_args), tuple(size_args), kwargs


def _field_constituents_shape_and_dims(
    arg: Any,  # TODO(havogt): improve typing
    arg_type: ts.DataType,
) -> Iterator[tuple[tuple[int, ...], list[common.Dimension]]]:
    match arg_type:
        case ts.TupleType():
            for el, el_type in zip(arg, arg_type.types):
                yield from _field_constituents_shape_and_dims(el, el_type)
        case ts.FieldType():
            dims = type_info.extract_dims(arg_type)
            if hasattr(arg, "shape"):
                assert len(arg.shape) == len(dims)
                yield (arg.shape, dims)
            else:
                yield (tuple(), dims)
        case ts.ScalarType():
            yield (tuple(), [])
        case _:
            raise ValueError("Expected 'FieldType' or 'TupleType' thereof.")
