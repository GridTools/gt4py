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
from gt4py.next import common
from gt4py.next.type_system import type_info


def _is_zero_dim_field(field: ts.TypeSpec) -> bool:
    return isinstance(field, ts.FieldType) and len(field.dims) == 0


def promote_zero_dims(
    function_type: ts.FunctionType, args: list[ts.TypeSpec], kwargs: dict[str, ts.TypeSpec]
) -> tuple[list, dict]:
    """
    Promote arg types to zero dimensional fields if compatible and required by function signature.

    This function expects the arguments to already be canonicalized using `canonicalize_arguments`.
    """

    def promote_arg(param: ts.TypeSpec, arg: ts.TypeSpec) -> ts.TypeSpec:
        def _as_field(arg_el: ts.TypeSpec, path: tuple[int, ...]) -> ts.TypeSpec:
            param_el = param
            for idx in path:
                if not isinstance(param_el, ts.TupleType):
                    # The parameter has a different structure than the actual argument. Just return
                    # the argument unpromoted and let the further error handling take care of printing
                    # a meaningful error.
                    return arg_el
                param_el = param_el.types[idx]

            if _is_zero_dim_field(param_el) and (
                type_info.is_number(arg_el) or type_info.is_logical(arg_el)
            ):
                if type_info.extract_dtype(param_el) == type_info.extract_dtype(arg_el):
                    return param_el
                else:
                    raise ValueError(f"'{arg_el}' is not compatible with '{param_el}'.")
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


@type_info.canonicalize_arguments.register(ts_ffront.FieldOperatorType)
@type_info.canonicalize_arguments.register(ts_ffront.ProgramType)
def canonicalize_program_or_fieldop_arguments(
    program_type: ts_ffront.ProgramType,
    args: tuple | list,
    kwargs: dict,
    *,
    ignore_errors=False,
    use_signature_ordering=False,
) -> tuple[list, dict]:
    return type_info.canonicalize_arguments(
        program_type.definition,
        args,
        kwargs,
        ignore_errors=ignore_errors,
        use_signature_ordering=use_signature_ordering,
    )


@type_info.canonicalize_arguments.register
def canonicalize_scanop_arguments(
    scanop_type: ts_ffront.ScanOperatorType,
    args: tuple | list,
    kwargs: dict,
    *,
    ignore_errors=False,
    use_signature_ordering=False,
) -> tuple[list, dict]:
    (_, *cargs), ckwargs = type_info.canonicalize_arguments(
        scanop_type.definition,
        (None, *args),
        kwargs,
        ignore_errors=ignore_errors,
        use_signature_ordering=use_signature_ordering,
    )
    return cargs, ckwargs


@type_info.function_signature_incompatibilities.register
def function_signature_incompatibilities_fieldop(
    fieldop_type: ts_ffront.FieldOperatorType,
    args: list[ts.TypeSpec],
    kwargs: dict[str, ts.TypeSpec],
) -> Iterator[str]:
    args, kwargs = type_info.canonicalize_arguments(
        fieldop_type.definition, args, kwargs, ignore_errors=True
    )

    error_list = list(
        type_info.structural_function_signature_incompatibilities(
            fieldop_type.definition, args, kwargs
        )
    )
    if len(error_list) > 0:
        yield from error_list
        return

    new_args, new_kwargs = promote_zero_dims(fieldop_type.definition, args, kwargs)
    yield from type_info.function_signature_incompatibilities_func(
        fieldop_type.definition, new_args, new_kwargs, skip_canonicalization=True
    )


def _scan_param_promotion(param: ts.TypeSpec, arg: ts.TypeSpec) -> ts.FieldType | ts.TupleType:
    """
    Promote parameter of a scan pass to match dimensions of respective scan operator argument.

    More specifically: Given a scalar type `param` and a field type `arg` return a field with the
    dtype of the `param` and the dimensions of `arg`. If `param` is a composite of scalars
    the promotion is element-wise.

    Example:
    --------
    >>> _scan_param_promotion(
    ...     ts.ScalarType(kind=ts.ScalarKind.INT64),
    ...     ts.FieldType(dims=[common.Dimension("I")], dtype=ts.ScalarKind.FLOAT64)
    ... )
    FieldType(dims=[Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)], dtype=ScalarType(kind=<ScalarKind.INT64: 64>, shape=None))
    """

    def _as_field(dtype: ts.TypeSpec, path: tuple[int, ...]) -> ts.FieldType:
        assert isinstance(dtype, ts.ScalarType)
        try:
            el_type = reduce(
                lambda type_, idx: type_.types[idx], path, arg  # type: ignore[attr-defined]
            )
            return ts.FieldType(dims=type_info.extract_dims(el_type), dtype=dtype)
        except (IndexError, AttributeError):
            # The structure of the scan passes argument and the requested
            # argument type differ. As such we can not extract the dimensions
            # and just return a generic field shown in the error later on.
            # TODO: we want some generic field type here, but our type system does not support it yet.
            return ts.FieldType(dims=[common.Dimension("...")], dtype=dtype)

    return type_info.apply_to_primitive_constituents(param, _as_field, with_path_arg=True)


@type_info.function_signature_incompatibilities.register
def function_signature_incompatibilities_scanop(
    scanop_type: ts_ffront.ScanOperatorType, args: list[ts.TypeSpec], kwargs: dict[str, ts.TypeSpec]
) -> Iterator[str]:
    if not all(
        type_info.is_type_or_tuple_of_type(arg, (ts.ScalarType, ts.FieldType)) for arg in args
    ):
        yield "Arguments to scan operator must be fields, scalars or tuples thereof."
        return

    scan_pass_type: ts.FunctionType = scanop_type.definition
    assert len(scan_pass_type.pos_only_args) == 0

    # canonicalize function arguments
    cargs, ckwargs = type_info.canonicalize_arguments(scanop_type, args, kwargs, ignore_errors=True)

    # check for structural errors
    num_pos_args = len(cargs) - cargs.count(type_info.UNDEFINED_ARG)
    if num_pos_args != len(scan_pass_type.pos_or_kw_args) - 1:
        yield f"Scan operator takes {len(scan_pass_type.pos_or_kw_args) - 1} positional arguments, but {num_pos_args} were given."
        return
    error_list = list(
        type_info.structural_function_signature_incompatibilities(
            scan_pass_type, [None, *cargs], ckwargs
        )
    )
    if len(error_list) > 0:
        yield from error_list
        return
    assert ckwargs.keys() == scan_pass_type.kw_only_args.keys()

    # ensure the dimensions of all arguments can be promoted to a common list of dimensions
    arg_dims = [
        type_info.extract_dims(el)
        for arg in [*cargs, *ckwargs.values()]
        for el in type_info.primitive_constituents(arg)
    ]
    try:
        common.promote_dims(*arg_dims)
    except ValueError as e:
        yield e.args[0]

    assert len(scan_pass_type.pos_only_args) == 0

    # promote parameters
    promoted_params = {}
    for (name, param), arg in zip(list(scan_pass_type.pos_or_kw_args.items())[1:], cargs):
        promoted_params[name] = _scan_param_promotion(param, arg)
    promoted_kwparams = {}
    for name, param, arg in zip(
        ckwargs.keys(), scan_pass_type.kw_only_args.values(), ckwargs.values()
    ):
        promoted_kwparams[name] = _scan_param_promotion(param, arg)

    # build a function type to leverage the already existing signature checking capabilities
    function_type = ts.FunctionType(
        pos_only_args=[],
        pos_or_kw_args=promoted_params,  # type: ignore[arg-type] # dict is invariant, but we don't care here.
        kw_only_args=promoted_kwparams,  # type: ignore[arg-type] # same as above
        returns=ts.DeferredType(constraint=None),
    )

    yield from type_info.function_signature_incompatibilities_func(
        function_type,
        *promote_zero_dims(function_type, cargs, ckwargs),
        skip_canonicalization=True,
        skip_structural_checks=True,
    )


@type_info.function_signature_incompatibilities.register
def function_signature_incompatibilities_program(
    program_type: ts_ffront.ProgramType, args: list[ts.TypeSpec], kwargs: dict[str, ts.TypeSpec]
) -> Iterator[str]:
    args, kwargs = type_info.canonicalize_arguments(
        program_type.definition, args, kwargs, ignore_errors=True
    )

    error_list = list(
        type_info.structural_function_signature_incompatibilities(
            program_type.definition, args, kwargs
        )
    )
    if len(error_list) > 0:
        yield from error_list
        return

    new_args, new_kwargs = promote_zero_dims(program_type.definition, args, kwargs)
    yield from type_info.function_signature_incompatibilities_func(
        program_type.definition, new_args, new_kwargs, skip_canonicalization=True
    )


@type_info.return_type.register
def return_type_scanop(
    callable_type: ts_ffront.ScanOperatorType,
    *,
    with_args: list[ts.TypeSpec],
    with_kwargs: dict[str, ts.TypeSpec],
):
    carry_dtype = callable_type.definition.returns
    promoted_dims = common.promote_dims(
        *(
            type_info.extract_dims(el)
            for arg in with_args + list(with_kwargs.values())
            for el in type_info.primitive_constituents(arg)
        ),
        # the vertical axis is always added to the dimension of the returned
        #  field
        [callable_type.axis],
    )
    return type_info.apply_to_primitive_constituents(
        carry_dtype, lambda arg: ts.FieldType(dims=promoted_dims, dtype=cast(ts.ScalarType, arg))
    )
