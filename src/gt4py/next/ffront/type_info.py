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

from functools import reduce
from typing import Iterator, cast

import gt4py.next.ffront.type_specifications as ts_ffront
import gt4py.next.type_system.type_specifications as ts
from gt4py.next.common import Dimension, GTTypeError
from gt4py.next.type_system import type_info


def _is_zero_dim_field(field: ts.TypeSpec) -> bool:
    return isinstance(field, ts.FieldType) and len(field.dims) == 0


def promote_zero_dims(
    function_type: ts.FunctionType, args: list[ts.TypeSpec], kwargs: dict[str, ts.TypeSpec]
) -> tuple[list, dict]:
    """Promote arg types to zero dimensional fields if compatible and required by function signature."""
    args, kwargs = type_info.canonicalize_function_arguments(
        function_type, args, kwargs, ignore_errors=True
    )

    def promote_arg(param: ts.TypeSpec, arg: ts.TypeSpec):
        def _as_field(arg_el: ts.TypeSpec, path: tuple):
            param_el = reduce(lambda type_, idx: type_.types[idx], path, param)  # noqa: B023

            if _is_zero_dim_field(param_el) and type_info.is_number(arg_el):
                if type_info.extract_dtype(param_el) == type_info.extract_dtype(arg_el):
                    return param_el
                else:
                    raise GTTypeError(f"{arg_el} is not compatible with {param_el}.")
            return arg_el

        return type_info.apply_to_primitive_constituents(arg, _as_field, with_path_arg=True)

    new_args = [*args]
    for i, (param, arg) in enumerate(
        zip(function_type.pos_only_args + list(function_type.pos_or_kw_args.values()), args)
    ):
        new_args[i] = promote_arg(param, arg)
    new_kwargs = {**kwargs}
    for name in set(function_type.kw_only_args.keys()) & set(kwargs.keys()):
        new_kwargs[name] = promote_arg(function_type.kw_only_args[name], kwargs[name])

    return new_args, new_kwargs


@type_info.return_type.register
def return_type_fieldop(
    fieldop_type: ts_ffront.FieldOperatorType,
    *,
    with_args: list[ts.TypeSpec],
    with_kwargs: dict[str, ts.TypeSpec],
):
    ret_type = type_info.return_type(
        fieldop_type.definition, with_args=with_args, with_kwargs=with_kwargs
    )
    return ret_type


@type_info.function_signature_incompatibilities.register
def function_signature_incompatibilities_fieldop(
    fieldop_type: ts_ffront.FieldOperatorType,
    args: list[ts.TypeSpec],
    kwargs: dict[str, ts.TypeSpec],
) -> Iterator[str]:
    new_args, new_kwargs = promote_zero_dims(fieldop_type.definition, args, kwargs)
    yield from type_info.function_signature_incompatibilities_func(
        fieldop_type.definition, new_args, new_kwargs
    )


@type_info.function_signature_incompatibilities.register
def function_signature_incompatibilities_scanop(
    scanop_type: ts_ffront.ScanOperatorType, args: list[ts.TypeSpec], kwargs: dict[str, ts.TypeSpec]
) -> Iterator[str]:
    if not all(
        type_info.is_type_or_tuple_of_type(arg, (ts.ScalarType, ts.FieldType)) for arg in args
    ):
        yield "Arguments to scan operator must be fields, scalars or tuples thereof."
        return

    arg_dims = [
        type_info.extract_dims(el) for arg in args for el in type_info.primitive_constituents(arg)
    ]
    try:
        type_info.promote_dims(*arg_dims)
    except GTTypeError as e:
        yield e.args[0]

    if len(args) != len(scanop_type.definition.pos_only_args) - 1:
        yield f"Scan operator takes {len(scanop_type.definition.pos_only_args) - 1} arguments, but {len(args)} were given."
        return

    promoted_args = []
    for i, scan_pass_arg in enumerate(scanop_type.definition.pos_only_args[1:]):
        # Helper function that given a scalar type in the signature of the scan
        # pass return a field type with that dtype and the dimensions of the
        # corresponding field type in the requested `pos_only_args` type. Defined here
        # as we capture `i`.
        def _as_field(dtype: ts.ScalarType, path: tuple[int, ...]) -> ts.FieldType:
            try:
                el_type = reduce(lambda type_, idx: type_.types[idx], path, args[i])  # type: ignore[attr-defined] # noqa: B023
                return ts.FieldType(dims=type_info.extract_dims(el_type), dtype=dtype)
            except (IndexError, AttributeError):
                # The structure of the scan passes argument and the requested
                # argument type differ. As such we can not extract the dimensions
                # and just return a generic field shown in the error later on.
                # TODO: we want some generic field type here, but our type system does not support it yet.
                return ts.FieldType(dims=[Dimension("...")], dtype=dtype)

        promoted_args.append(
            type_info.apply_to_primitive_constituents(scan_pass_arg, _as_field, with_path_arg=True)  # type: ignore[arg-type]
        )

    # build a function type to leverage the already existing signature checking
    #  capabilities
    function_type = ts.FunctionType(
        pos_only_args=promoted_args,
        pos_or_kw_args={},
        kw_only_args={},
        returns=ts.DeferredType(constraint=None),
    )

    yield from type_info.function_signature_incompatibilities(
        function_type, *promote_zero_dims(function_type, args, kwargs)
    )


@type_info.function_signature_incompatibilities.register
def function_signature_incompatibilities_program(
    program_type: ts_ffront.ProgramType, args: list[ts.TypeSpec], kwargs: dict[str, ts.TypeSpec]
) -> Iterator[str]:
    new_args, new_kwargs = promote_zero_dims(program_type.definition, args, kwargs)
    yield from type_info.function_signature_incompatibilities_func(
        program_type.definition, new_args, new_kwargs
    )


@type_info.return_type.register
def return_type_scanop(
    callable_type: ts_ffront.ScanOperatorType,
    *,
    with_args: list[ts.TypeSpec],
    with_kwargs: dict[str, ts.TypeSpec],
):
    carry_dtype = callable_type.definition.returns
    promoted_dims = type_info.promote_dims(
        *(
            type_info.extract_dims(el)
            for arg in with_args
            for el in type_info.primitive_constituents(arg)
        ),
        # the vertical axis is always added to the dimension of the returned
        #  field
        [callable_type.axis],
    )
    return type_info.apply_to_primitive_constituents(
        carry_dtype, lambda arg: ts.FieldType(dims=promoted_dims, dtype=cast(ts.ScalarType, arg))
    )
