# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either vervisit_Constantsion 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from functools import reduce
from typing import Iterable, Iterator, cast

import gt4py.next.ffront.type_specifications as ts
from gt4py.next.common import Dimension, GTTypeError
from gt4py.next.type_system.type_info import (  # noqa: F401
    accepts_args as accepts_args,
    apply_to_primitive_constituents as apply_to_primitive_constituents,
    extract_dims as extract_dims,
    extract_dtype as extract_dtype,
    function_signature_incompatibilities as function_signature_incompatibilities,
    function_signature_incompatibilities_field as function_signature_incompatibilities_field,
    function_signature_incompatibilities_func as function_signature_incompatibilities_func,
    is_arithmetic as is_arithmetic,
    is_concrete as is_concrete,
    is_concretizable as is_concretizable,
    is_floating_point as is_floating_point,
    is_integral as is_integral,
    is_logical as is_logical,
    is_number as is_number,
    is_type_or_tuple_of_type as is_type_or_tuple_of_type,
    primitive_constituents as primitive_constituents,
    promote as promote,
    promote_dims as promote_dims,
    return_type as return_type,
    return_type_field as return_type_field,
    return_type_func as return_type_func,
    type_class as type_class,
)


def _is_zero_dim_field(field: ts.TypeSpec) -> bool:
    return (
        isinstance(field, ts.FieldType)
        and isinstance(field.dims, Iterable)
        and len(field.dims) == 0
    )


def promote_zero_dims(
    args: list[ts.TypeSpec], function_type: ts.FieldOperatorType | ts.ProgramType | ts.FunctionType
) -> list:
    """Promote arg types to zero dimensional fields if compatible and required by function signature."""
    new_args = []
    for arg_i, arg in enumerate(args):
        def_type = (
            function_type.args[arg_i]
            if isinstance(function_type, ts.FunctionType)
            else function_type.definition.args[arg_i]
        )

        def _as_field(arg: ts.TypeSpec, path: tuple):
            el_def_type = reduce(lambda type_, idx: type_.types[idx], path, def_type)  # noqa: B023

            if _is_zero_dim_field(el_def_type) and is_number(arg):
                if extract_dtype(el_def_type) == extract_dtype(arg):
                    return el_def_type
                else:
                    raise GTTypeError(f"{arg} is not compatible with {el_def_type}.")
            return arg

        new_args.append(apply_to_primitive_constituents(arg, _as_field, with_path_arg=True))

    return new_args


@return_type.register
def return_type_fieldop(
    fieldop_type: ts.FieldOperatorType,
    *,
    with_args: list[ts.TypeSpec],
    with_kwargs: dict[str, ts.TypeSpec],
):
    ret_type = return_type(fieldop_type.definition, with_args=with_args, with_kwargs=with_kwargs)
    return ret_type


@function_signature_incompatibilities.register
def function_signature_incompatibilities_fieldop(
    fieldop_type: ts.FieldOperatorType, args: list[ts.TypeSpec], kwargs: dict[str, ts.TypeSpec]
) -> Iterator[str]:
    new_args = promote_zero_dims(args, fieldop_type)
    yield from function_signature_incompatibilities_func(fieldop_type.definition, new_args, kwargs)


@function_signature_incompatibilities.register
def function_signature_incompatibilities_scanop(
    scanop_type: ts.ScanOperatorType, args: list[ts.TypeSpec], kwargs: dict[str, ts.TypeSpec]
) -> Iterator[str]:
    if not all(is_type_or_tuple_of_type(arg, (ts.ScalarType, ts.FieldType)) for arg in args):
        yield "Arguments to scan operator must be fields, scalars or tuples thereof."
        return

    arg_dims = [extract_dims(el) for arg in args for el in primitive_constituents(arg)]
    try:
        promote_dims(*arg_dims)
    except GTTypeError as e:
        yield e.args[0]

    if len(args) != len(scanop_type.definition.args) - 1:
        yield f"Scan operator takes {len(scanop_type.definition.args)-1} arguments, but {len(args)} were given."
        return

    promoted_args = []
    for i, scan_pass_arg in enumerate(scanop_type.definition.args[1:]):
        # Helper function that given a scalar type in the signature of the scan
        # pass return a field type with that dtype and the dimensions of the
        # corresponding field type in the requested `args` type. Defined here
        # as we capture `i`.
        def _as_field(dtype: ts.ScalarType, path: tuple[int, ...]) -> ts.FieldType:
            try:
                el_type = reduce(lambda type_, idx: type_.types[idx], path, args[i])  # type: ignore[attr-defined] # noqa: B023
                return ts.FieldType(dims=extract_dims(el_type), dtype=dtype)
            except (IndexError, AttributeError):
                # The structure of the scan passes argument and the requested
                # argument type differ. As such we can not extract the dimensions
                # and just return a generic field shown in the error later on.
                # TODO: we want some generic field type here, but our type system does not support it yet.
                return ts.FieldType(dims=[Dimension("...")], dtype=dtype)

        promoted_args.append(
            apply_to_primitive_constituents(scan_pass_arg, _as_field, with_path_arg=True)  # type: ignore[arg-type]
        )

    # build a function type to leverage the already existing signature checking
    #  capabilities
    function_type = ts.FunctionType(
        args=promoted_args,
        kwargs={},
        returns=ts.DeferredType(constraint=None),
    )

    yield from function_signature_incompatibilities(
        function_type, promote_zero_dims(args, function_type), kwargs
    )


@function_signature_incompatibilities.register
def function_signature_incompatibilities_program(
    program_type: ts.ProgramType, args: list[ts.TypeSpec], kwargs: dict[str, ts.TypeSpec]
) -> Iterator[str]:
    new_args = promote_zero_dims(args, program_type)
    yield from function_signature_incompatibilities_func(program_type.definition, new_args, kwargs)


@return_type.register
def return_type_scanop(
    callable_type: ts.ScanOperatorType,
    *,
    with_args: list[ts.TypeSpec],
    with_kwargs: dict[str, ts.TypeSpec],
):
    carry_dtype = callable_type.definition.returns
    promoted_dims = promote_dims(
        *(extract_dims(el) for arg in with_args for el in primitive_constituents(arg)),
        # the vertical axis is always added to the dimension of the returned
        #  field
        [callable_type.axis],
    )
    return apply_to_primitive_constituents(
        carry_dtype, lambda arg: ts.FieldType(dims=promoted_dims, dtype=cast(ts.ScalarType, arg))
    )
