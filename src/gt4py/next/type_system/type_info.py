# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import types
from collections.abc import Callable, Iterator
from typing import (
    Any,
    Generic,
    Literal,
    Protocol,
    Sequence,
    Type,
    TypeGuard,
    TypeVar,
    cast,
    overload,
)

import numpy as np
try:
    import ml_dtypes
except ModuleNotFoundError:
    ml_dtypes = None

from gt4py.eve.utils import XIterable, xiter
from gt4py.next import common
from gt4py.next.type_system import type_specifications as ts


def _number_to_ordinal_number(number: int) -> str:
    """
    Convert number into ordinal number.

    >>> for i in range(0, 5):
    ...     print(_number_to_ordinal_number(i))
    0th
    1st
    2nd
    3rd
    4th
    """
    number_as_string = str(number)
    if len(number_as_string) > 1 and number_as_string[-2] == "1":
        return number_as_string + "th"
    last_digit = number_as_string[-1]
    if last_digit == "1":
        return number_as_string + "st"
    if last_digit == "2":
        return number_as_string + "nd"
    if last_digit == "3":
        return number_as_string + "rd"
    return number_as_string + "th"


def is_concrete(symbol_type: ts.TypeSpec) -> TypeGuard[ts.TypeSpec]:
    """Figure out if the foast type is completely deduced."""
    if isinstance(symbol_type, ts.DeferredType):
        return False
    elif isinstance(symbol_type, ts.TypeSpec):
        return True
    return False


def type_class(symbol_type: ts.TypeSpec) -> Type[ts.TypeSpec]:
    """
    Determine which class should be used to create a compatible concrete type.

    Examples:
    ---------
    >>> type_class(ts.DeferredType(constraint=ts.ScalarType)).__name__
    'ScalarType'

    >>> type_class(ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.BOOL))).__name__
    'FieldType'

    >>> type_class(ts.TupleType(types=[])).__name__
    'TupleType'
    """
    match symbol_type:
        case ts.DeferredType(constraint):
            if constraint is None:
                raise ValueError(f"No type information available for '{symbol_type}'.")
            elif isinstance(constraint, tuple):
                raise ValueError(f"Not sufficient type information available for '{symbol_type}'.")
            return constraint
        case ts.TypeSpec() as concrete_type:
            return concrete_type.__class__
    raise ValueError(
        f"Invalid type for TypeInfo: requires '{ts.TypeSpec}', got '{type(symbol_type)}'."
    )


@overload
def primitive_constituents(
    symbol_type: ts.TypeSpec, with_path_arg: Literal[False] = False
) -> XIterable[ts.TypeSpec]: ...


@overload
def primitive_constituents(
    symbol_type: ts.TypeSpec, with_path_arg: Literal[True]
) -> XIterable[tuple[ts.TypeSpec, tuple[int, ...]]]: ...


def primitive_constituents(
    symbol_type: ts.TypeSpec, with_path_arg: bool = False
) -> XIterable[ts.TypeSpec] | XIterable[tuple[ts.TypeSpec, tuple[int, ...]]]:
    """
    Return the primitive types contained in a composite type.

    >>> from gt4py.next import common
    >>> I = common.Dimension(value="I")
    >>> int_type = ts.ScalarType(kind=ts.ScalarKind.INT64)
    >>> field_type = ts.FieldType(dims=[I], dtype=int_type)

    >>> tuple_type = ts.TupleType(types=[int_type, field_type])
    >>> primitive_constituents(tuple_type).to_list()  # doctest: +ELLIPSIS
    [ScalarType(...), FieldType(...)]

    >>> nested_tuple = ts.TupleType(types=[field_type, tuple_type])
    >>> primitive_constituents(nested_tuple).to_list()  # doctest: +ELLIPSIS
    [FieldType(...), ScalarType(...), FieldType(...)]
    """

    def constituents_yielder(
        symbol_type: ts.TypeSpec, path: tuple[int, ...]
    ) -> Iterator[ts.TypeSpec] | Iterator[tuple[ts.TypeSpec, tuple[int, ...]]]:
        if isinstance(symbol_type, ts.TupleType):
            for i, el_type in enumerate(symbol_type.types):
                yield from constituents_yielder(el_type, (*path, i))
        else:
            if with_path_arg:
                yield (symbol_type, path)
            else:
                yield symbol_type

    return xiter(constituents_yielder(symbol_type, ()))  # type: ignore[return-value] # why resolved to XIterable[object]?


_R = TypeVar("_R", covariant=True)
_T = TypeVar("_T")


class TupleConstructorType(Protocol, Generic[_R]):
    def __call__(self, *args: Any) -> _R: ...


def apply_to_primitive_constituents(
    fun: Callable[..., _T],
    *symbol_types: ts.TypeSpec,
    with_path_arg: bool = False,
    tuple_constructor: TupleConstructorType[_R] = lambda *elements: ts.TupleType(types=[*elements]),  # type: ignore[assignment] # probably related to https://github.com/python/mypy/issues/10854
    _path: tuple[int, ...] = (),
) -> _T | _R:
    """
    Apply function to all primitive constituents of a type.

    >>> int_type = ts.ScalarType(kind=ts.ScalarKind.INT64)
    >>> tuple_type = ts.TupleType(types=[int_type, int_type])
    >>> print(
    ...     apply_to_primitive_constituents(
    ...         lambda primitive_type: ts.FieldType(dims=[], dtype=primitive_type),
    ...         tuple_type,
    ...     )
    ... )
    tuple[Field[[], int64], Field[[], int64]]

    >>> apply_to_primitive_constituents(
    ...     lambda primitive_type, path: (path, primitive_type),
    ...     tuple_type,
    ...     with_path_arg=True,
    ...     tuple_constructor=lambda *elements: dict(elements),
    ... )
    {(0,): ScalarType(kind=<ScalarKind.INT64: 64>, shape=None), (1,): ScalarType(kind=<ScalarKind.INT64: 64>, shape=None)}
    """
    if isinstance(symbol_types[0], ts.TupleType):
        assert all(isinstance(symbol_type, ts.TupleType) for symbol_type in symbol_types)
        return tuple_constructor(
            *[
                apply_to_primitive_constituents(
                    fun,
                    *el_types,
                    _path=(*_path, i),
                    with_path_arg=with_path_arg,
                    tuple_constructor=tuple_constructor,
                )
                for i, el_types in enumerate(
                    zip(*(symbol_type.types for symbol_type in symbol_types))  # type: ignore[attr-defined]  # ensured by assert above
                )
            ]
        )
    if with_path_arg:
        return fun(*symbol_types, path=_path)
    else:
        return fun(*symbol_types)


def extract_dtype(symbol_type: ts.TypeSpec) -> ts.ScalarType:
    """
    Extract the data type from ``symbol_type`` if it is either `FieldType` or `ScalarType`.

    Raise an error if no dtype can be found or the result would be ambiguous.

    Examples:
    ---------
    >>> print(extract_dtype(ts.ScalarType(kind=ts.ScalarKind.FLOAT64)))
    float64

    >>> print(extract_dtype(ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.BOOL))))
    bool
    """
    match symbol_type:
        case ts.FieldType(dtype=dtype):
            return dtype
        case ts.ScalarType() as dtype:
            return dtype
    raise ValueError(f"Can not unambiguosly extract data type from '{symbol_type}'.")


def is_floating_point(symbol_type: ts.TypeSpec) -> bool:
    """
    Check if the dtype of ``symbol_type`` is a floating point type.

    Examples:
    ---------
    >>> is_floating_point(ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
    True
    >>> is_floating_point(ts.ScalarType(kind=ts.ScalarKind.FLOAT32))
    True
    >>> is_floating_point(ts.ScalarType(kind=ts.ScalarKind.INT32))
    False
    >>> is_floating_point(ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32)))
    True
    """
    return extract_dtype(symbol_type).kind in [
        ts.ScalarKind.FLOAT16,
        ts.ScalarKind.BFLOAT16,
        ts.ScalarKind.FLOAT32,
        ts.ScalarKind.FLOAT64,
    ]


def is_integer(symbol_type: ts.TypeSpec) -> bool:
    """
    Check if ``symbol_type`` is an integral type.

    Examples:
    ---------
    >>> is_integer(ts.ScalarType(kind=ts.ScalarKind.INT32))
    True
    >>> is_integer(ts.ScalarType(kind=ts.ScalarKind.FLOAT32))
    False
    >>> is_integer(ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32)))
    False
    """
    return isinstance(symbol_type, ts.ScalarType) and symbol_type.kind in {
        ts.ScalarKind.INT8,
        ts.ScalarKind.UINT8,
        ts.ScalarKind.INT16,
        ts.ScalarKind.UINT16,
        ts.ScalarKind.INT32,
        ts.ScalarKind.UINT32,
        ts.ScalarKind.INT64,
    }


def is_integral(symbol_type: ts.TypeSpec) -> bool:
    """
    Check if the dtype of ``symbol_type`` is an integral type.

    Examples:
    ---------
    >>> is_integral(ts.ScalarType(kind=ts.ScalarKind.INT32))
    True
    >>> is_integral(ts.ScalarType(kind=ts.ScalarKind.FLOAT32))
    False
    >>> is_integral(ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32)))
    True
    """
    return is_integer(extract_dtype(symbol_type))


def is_number(symbol_type: ts.TypeSpec) -> bool:
    """
    Check if ``symbol_type`` is either intergral or float.

    Examples:
    ---------
    >>> is_number(ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
    True
    >>> is_number(ts.ScalarType(kind=ts.ScalarKind.INT32))
    True
    >>> is_number(ts.ScalarType(kind=ts.ScalarKind.BOOL))
    False
    >>> is_number(ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.INT64)))
    False
    """
    if not isinstance(symbol_type, ts.ScalarType):
        return False
    # TODO(nfarabullini): re-factor is_arithmetic such that it only checks for scalars
    #  and the emtpy field pass in an another function
    return is_arithmetic(symbol_type)


def is_logical(symbol_type: ts.TypeSpec) -> bool:
    return extract_dtype(symbol_type).kind is ts.ScalarKind.BOOL


def is_arithmetic(symbol_type: ts.TypeSpec) -> bool:
    """
    Check if ``symbol_type`` is compatible with arithmetic operations.

    Examples:
    ---------
    >>> is_arithmetic(ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
    True
    >>> is_arithmetic(ts.ScalarType(kind=ts.ScalarKind.BOOL))
    False
    >>> is_arithmetic(ts.ScalarType(kind=ts.ScalarKind.STRING))
    False
    >>> is_arithmetic(ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32)))
    True
    """
    return is_floating_point(symbol_type) or is_integral(symbol_type)


def arithmetic_bounds(arithmetic_type: ts.ScalarType) -> tuple[np.number, np.number]:
    assert is_arithmetic(arithmetic_type)
    bounds = {  # type: ignore[return-value] # why resolved to `tuple[object, object]`?
        ts.ScalarKind.FLOAT16: (np.finfo(np.float16).min, np.finfo(np.float16).max),  # todo: cleanup?
        ts.ScalarKind.FLOAT32: (np.finfo(np.float32).min, np.finfo(np.float32).max),
        ts.ScalarKind.FLOAT64: (np.finfo(np.float64).min, np.finfo(np.float64).max),
        ts.ScalarKind.INT8: (np.iinfo(np.int8).min, np.iinfo(np.int8).max),
        ts.ScalarKind.UINT8: (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max),
        ts.ScalarKind.INT16: (np.iinfo(np.int16).min, np.iinfo(np.int16).max),
        ts.ScalarKind.UINT16: (np.iinfo(np.uint16).min, np.iinfo(np.uint16).max),
        ts.ScalarKind.INT32: (np.iinfo(np.int32).min, np.iinfo(np.int32).max),
        ts.ScalarKind.UINT32: (np.iinfo(np.uint32).min, np.iinfo(np.uint32).max),
        ts.ScalarKind.INT64: (np.iinfo(np.int64).min, np.iinfo(np.int64).max),
        ts.ScalarKind.UINT64: (np.iinfo(np.uint64).min, np.iinfo(np.uint64).max),
    }
    if ml_dtypes:
        bounds[ts.ScalarKind.BFLOAT16] = (ml_dtypes.finfo(ml_dtypes.bfloat16).min, ml_dtypes.finfo(ml_dtypes.bfloat16).max)
    return bounds[arithmetic_type.kind]


def is_type_or_tuple_of_type(type_: ts.TypeSpec, expected_type: type | tuple) -> bool:
    """
    Return True if ``type_`` matches any of the expected or is a tuple of them.

    Examples:
    ---------
    >>> scalar_type = ts.ScalarType(kind=ts.ScalarKind.INT64)
    >>> field_type = ts.FieldType(dims=[], dtype=scalar_type)
    >>> is_type_or_tuple_of_type(field_type, ts.FieldType)
    True
    >>> is_type_or_tuple_of_type(
    ...     ts.TupleType(types=[scalar_type, field_type]), (ts.ScalarType, ts.FieldType)
    ... )
    True
    >>> is_type_or_tuple_of_type(scalar_type, ts.FieldType)
    False
    >>> is_type_or_tuple_of_type(ts.TupleType(types=[scalar_type, field_type]), ts.FieldType)
    False
    """
    return all(isinstance(t, expected_type) for t in primitive_constituents(type_))


def is_tuple_of_type(type_: ts.TypeSpec, expected_type: type | tuple) -> TypeGuard[ts.TupleType]:
    """
    Return True if ``type_`` matches (nested) tuple of ``expected_type``.

    Examples:
    ---------
    >>> scalar_type = ts.ScalarType(kind=ts.ScalarKind.INT64)
    >>> field_type = ts.FieldType(dims=[], dtype=scalar_type)
    >>> is_tuple_of_type(field_type, ts.FieldType)
    False
    >>> is_tuple_of_type(
    ...     ts.TupleType(types=[scalar_type, field_type]), (ts.ScalarType, ts.FieldType)
    ... )
    True
    >>> is_tuple_of_type(ts.TupleType(types=[scalar_type]), ts.FieldType)
    False
    >>> is_tuple_of_type(ts.TupleType(types=[scalar_type, field_type]), ts.FieldType)
    False
    """
    return isinstance(type_, ts.TupleType) and is_type_or_tuple_of_type(type_, expected_type)


def extract_dims(symbol_type: ts.TypeSpec) -> list[common.Dimension]:
    """
    Try to extract field dimensions if possible.

    Scalars are treated as zero-dimensional

    Examples:
    ---------
    >>> extract_dims(ts.ScalarType(kind=ts.ScalarKind.INT64, shape=[3, 4]))
    []
    >>> I = common.Dimension(value="I")
    >>> J = common.Dimension(value="J")
    >>> extract_dims(ts.FieldType(dims=[I, J], dtype=ts.ScalarType(kind=ts.ScalarKind.INT64)))
    [Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>), Dimension(value='J', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)]
    """
    match symbol_type:
        case ts.ScalarType():
            return []
        case ts.FieldType(dims):
            return dims
    raise ValueError(f"Can not extract dimensions from '{symbol_type}'.")


def is_local_field(type_: ts.FieldType) -> bool:
    """
    Return if `type_` is a field defined on a local dimension.

    Examples:
    ---------
    >>> V = common.Dimension(value="V")
    >>> V2E = common.Dimension(value="V2E", kind=common.DimensionKind.LOCAL)
    >>> is_local_field(ts.FieldType(dims=[V, V2E], dtype=ts.ScalarType(kind=ts.ScalarKind.INT64)))
    True
    >>> is_local_field(ts.FieldType(dims=[V], dtype=ts.ScalarType(kind=ts.ScalarKind.INT64)))
    False
    """
    return any(dim.kind == common.DimensionKind.LOCAL for dim in type_.dims)


def contains_local_field(type_: ts.TypeSpec) -> bool:
    """Return if primitive constitutens of `type_` contains a field defined on a local dimension."""
    return any(
        isinstance(t, ts.FieldType) and is_local_field(t) for t in primitive_constituents(type_)
    )


def is_concretizable(symbol_type: ts.TypeSpec, to_type: ts.TypeSpec) -> bool:
    """
    Check if ``symbol_type`` can be concretized to ``to_type``.

    Examples:
    ---------
    >>> is_concretizable(
    ...     ts.ScalarType(kind=ts.ScalarKind.INT64), to_type=ts.ScalarType(kind=ts.ScalarKind.INT64)
    ... )
    True

    >>> is_concretizable(
    ...     ts.ScalarType(kind=ts.ScalarKind.INT64),
    ...     to_type=ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
    ... )
    False

    >>> is_concretizable(
    ...     ts.DeferredType(constraint=None),
    ...     to_type=ts.FieldType(dtype=ts.ScalarType(kind=ts.ScalarKind.BOOL), dims=[]),
    ... )
    True

    >>> is_concretizable(
    ...     ts.DeferredType(constraint=ts.DataType),
    ...     to_type=ts.FieldType(dtype=ts.ScalarType(kind=ts.ScalarKind.BOOL), dims=[]),
    ... )
    True

    >>> is_concretizable(
    ...     ts.DeferredType(constraint=ts.OffsetType),
    ...     to_type=ts.FieldType(dtype=ts.ScalarType(kind=ts.ScalarKind.BOOL), dims=[]),
    ... )
    False

    >>> is_concretizable(
    ...     ts.DeferredType(constraint=ts.TypeSpec),
    ...     to_type=ts.DeferredType(constraint=ts.ScalarType),
    ... )
    True

    """
    if isinstance(symbol_type, ts.DeferredType) and (
        symbol_type.constraint is None
        or (isinstance(to_type, ts.DeferredType) and to_type.constraint is None)
        or issubclass(type_class(to_type), symbol_type.constraint)
    ):
        return True
    elif is_concrete(symbol_type):
        return symbol_type == to_type
    return False


def promote(
    *types: ts.FieldType | ts.ScalarType, always_field: bool = False
) -> ts.FieldType | ts.ScalarType:
    """
    Promote a set of field or scalar types to a common type.

    The resulting type is defined on all dimensions of the arguments, respecting
    the individual order of the dimensions of each argument (see
    :func:`common.promote_dims` for more details).

    >>> dtype = ts.ScalarType(kind=ts.ScalarKind.INT64)
    >>> I, J, K = (common.Dimension(value=dim) for dim in ["I", "J", "K"])
    >>> promoted: ts.FieldType = promote(
    ...     ts.FieldType(dims=[I, J], dtype=dtype), ts.FieldType(dims=[I, J, K], dtype=dtype), dtype
    ... )
    >>> promoted.dims == [I, J, K] and promoted.dtype == dtype
    True

    >>> promote(
    ...     ts.FieldType(dims=[I, J], dtype=dtype), ts.FieldType(dims=[K], dtype=dtype)
    ... )  # doctest: +ELLIPSIS
    Traceback (most recent call last):
     ...
    ValueError: Dimensions can not be promoted. Could not determine order of the following dimensions: J, K.
    """
    if not always_field and all(isinstance(type_, ts.ScalarType) for type_ in types):
        if not all(type_ == types[0] for type_ in types):
            raise ValueError("Could not promote scalars of different dtype (not implemented).")
        if not all(type_.shape is None for type_ in types):  # type: ignore[union-attr]
            raise NotImplementedError("Shape promotion not implemented.")
        return types[0]
    elif all(isinstance(type_, (ts.ScalarType, ts.FieldType)) for type_ in types):
        dims = common.promote_dims(*(extract_dims(type_) for type_ in types))
        dtype = cast(ts.ScalarType, promote(*(extract_dtype(type_) for type_ in types)))

        return ts.FieldType(dims=dims, dtype=dtype)
    raise TypeError("Expected a 'FieldType' or 'ScalarType'.")


@functools.singledispatch
def return_type(
    callable_type: ts.CallableType,
    *,
    with_args: Sequence[ts.TypeSpec],
    with_kwargs: dict[str, ts.TypeSpec],
) -> ts.TypeSpec:
    raise NotImplementedError(
        f"Return type deduction of type " f"'{type(callable_type).__name__}' not implemented."
    )


@return_type.register
def return_type_func(
    func_type: ts.FunctionType,
    *,
    with_args: Sequence[ts.TypeSpec],
    with_kwargs: dict[str, ts.TypeSpec],
) -> ts.TypeSpec:
    return func_type.returns


@return_type.register
def return_type_field(
    field_type: ts.FieldType,
    *,
    with_args: Sequence[ts.TypeSpec],
    with_kwargs: dict[str, ts.TypeSpec],
) -> ts.FieldType:
    try:
        accepts_args(
            field_type,
            with_args=with_args,
            with_kwargs=with_kwargs,
            raise_exception=True,
        )
    except ValueError as ex:
        raise ValueError("Could not deduce return type of invalid remap operation.") from ex

    if not isinstance(with_args[0], ts.OffsetType):
        raise ValueError(f"First argument must be of type '{ts.OffsetType}', got '{with_args[0]}'.")

    source_dim = with_args[0].source
    target_dims = with_args[0].target
    new_dims = []
    # TODO: This code does not handle ellipses for dimensions. Fix it.
    assert field_type.dims is not ...
    for d in field_type.dims:
        if d != source_dim:
            new_dims.append(d)
        else:
            new_dims.extend(target_dims)
    return ts.FieldType(dims=new_dims, dtype=field_type.dtype)


UNDEFINED_ARG = types.new_class("UNDEFINED_ARG")


@functools.singledispatch
def canonicalize_arguments(
    func_type: ts.CallableType,
    args: Sequence,
    kwargs: dict,
    *,
    ignore_errors: bool = False,
    use_signature_ordering: bool = False,
) -> tuple[list, dict]:
    raise NotImplementedError(f"Not implemented for type '{type(func_type).__name__}'.")


@canonicalize_arguments.register
def canonicalize_function_arguments(
    func_type: ts.FunctionType,
    args: Sequence,
    kwargs: dict,
    *,
    ignore_errors: bool = False,
    use_signature_ordering: bool = False,
) -> tuple[list, dict]:
    num_pos_params = len(func_type.pos_only_args) + len(func_type.pos_or_kw_args)
    cargs = [UNDEFINED_ARG] * max(num_pos_params, len(args))
    ckwargs = {**kwargs}
    for i, arg in enumerate(args):
        cargs[i] = arg
    for name in kwargs.keys():
        if name in func_type.pos_or_kw_args:
            args_idx = len(func_type.pos_only_args) + list(func_type.pos_or_kw_args.keys()).index(
                name
            )
            if cargs[args_idx] is UNDEFINED_ARG:
                cargs[args_idx] = ckwargs.pop(name)
            elif not ignore_errors:
                raise AssertionError(
                    f"Error canonicalizing function arguments. Got multiple values for argument '{name}'."
                )

    a, b = set(func_type.kw_only_args.keys()), set(ckwargs.keys())
    invalid_kw_args = (a - b) | (b - a)
    if invalid_kw_args and (not ignore_errors or use_signature_ordering):
        # this error can not be ignored as otherwise the invariant that no arguments are dropped
        # is invalidated.
        raise AssertionError(f"Invalid keyword arguments '{', '.join(invalid_kw_args)}'.")

    if use_signature_ordering:
        ckwargs = {k: ckwargs[k] for k in func_type.kw_only_args.keys() if k in ckwargs}

    return list(cargs), ckwargs


def structural_function_signature_incompatibilities(
    func_type: ts.FunctionType, args: Sequence, kwargs: dict[str, Any]
) -> Iterator[str]:
    """
    Return structural incompatibilities for a call to ``func_type`` with given arguments.

    This function requires `args` and `kwargs` to be canonicalized using
    `canonicalize_function_arguments`.

    Contrary to `function_signature_incompatibilities` the arguments don't need to be types, but
    can be anything. This function merely checks whether their structure matches (e.g. for every
    parameter in the signature there is a corresponding argument value).
    """
    kwargs = {**kwargs}

    # check positional arguments
    for name in {**kwargs}.keys():
        if name in func_type.pos_or_kw_args:
            args_idx = len(func_type.pos_only_args) + list(func_type.pos_or_kw_args.keys()).index(
                name
            )
            if args_idx < len(args):
                # remove the argument here such that later errors stay comprehensible
                kwargs.pop(name)
                yield f"Got multiple values for argument '{name}'."

    num_pos_params = len(func_type.pos_only_args) + len(func_type.pos_or_kw_args)
    num_pos_args = len(args) - args.count(UNDEFINED_ARG)
    if num_pos_params != num_pos_args:
        if len(kwargs) > 0:
            kwargs_msg = f"positional argument{'s' if num_pos_params != 1 else ''} (and {len(kwargs)} keyword-only argument{'s' if len(kwargs) != 1 else ''}) "
        else:
            kwargs_msg = ""
        yield f"Function takes {num_pos_params} positional argument{'s' if num_pos_params != 1 else ''}, but {num_pos_args} {kwargs_msg}were given."

    missing_positional_args = []
    for i, arg_type in zip(
        range(len(func_type.pos_only_args), num_pos_params),
        func_type.pos_or_kw_args.keys(),
    ):
        if args[i] is UNDEFINED_ARG:
            missing_positional_args.append(f"'{arg_type}'")
    if missing_positional_args:
        yield f"Missing {len(missing_positional_args)} required positional argument{'s' if len(missing_positional_args) != 1 else ''}: {', '.join(missing_positional_args)}"

    # check for missing or extra keyword arguments
    kw_a_m_b = set(func_type.kw_only_args.keys()) - set(kwargs.keys())
    if len(kw_a_m_b) > 0:
        yield f"Missing required keyword argument{'s' if len(kw_a_m_b) != 1 else ''} '{', '.join(kw_a_m_b)}'."
    kw_b_m_a = set(kwargs.keys()) - set(func_type.kw_only_args.keys())
    if len(kw_b_m_a) > 0:
        yield f"Got unexpected keyword argument{'s' if len(kw_b_m_a) != 1 else ''} '{', '.join(kw_b_m_a)}'."


@functools.singledispatch
def function_signature_incompatibilities(
    func_type: ts.CallableType, args: Sequence[ts.TypeSpec], kwargs: dict[str, ts.TypeSpec]
) -> Iterator[str]:
    """
    Return incompatibilities for a call to ``func_type`` with given arguments.

    Note that all types must be concrete/complete.
    """
    raise NotImplementedError(f"Not implemented for type '{type(func_type).__name__}'.")


@function_signature_incompatibilities.register
def function_signature_incompatibilities_func(
    func_type: ts.FunctionType,
    args: Sequence[ts.TypeSpec],
    kwargs: dict[str, ts.TypeSpec],
    *,
    skip_canonicalization: bool = False,
    skip_structural_checks: bool = False,
) -> Iterator[str]:
    if not skip_canonicalization:
        args, kwargs = canonicalize_arguments(func_type, args, kwargs, ignore_errors=True)

    # check for structural errors (e.g. wrong number of arguments)
    if not skip_structural_checks:
        error_list = list(structural_function_signature_incompatibilities(func_type, args, kwargs))
        if len(error_list) > 0:
            yield from error_list
            return

    num_pos_params = len(func_type.pos_only_args) + len(func_type.pos_or_kw_args)
    assert len(args) >= num_pos_params
    for i, (a_arg, b_arg) in enumerate(
        zip(list(func_type.pos_only_args) + list(func_type.pos_or_kw_args.values()), args)
    ):
        if (
            b_arg is not UNDEFINED_ARG
            and a_arg != b_arg
            and not is_concretizable(a_arg, to_type=b_arg)
        ):
            if i < len(func_type.pos_only_args):
                arg_repr = f"{_number_to_ordinal_number(i + 1)} argument"
            else:
                arg_repr = f"argument '{list(func_type.pos_or_kw_args.keys())[i - len(func_type.pos_only_args)]}'"
            yield f"Expected {arg_repr} to be of type '{a_arg}', got '{b_arg}'."

    for kwarg in set(func_type.kw_only_args.keys()) & set(kwargs.keys()):
        if (a_kwarg := func_type.kw_only_args[kwarg]) != (
            b_kwarg := kwargs[kwarg]
        ) and not is_concretizable(a_kwarg, to_type=b_kwarg):
            yield f"Expected keyword argument '{kwarg}' to be of type '{func_type.kw_only_args[kwarg]}', got '{kwargs[kwarg]}'."


@function_signature_incompatibilities.register
def function_signature_incompatibilities_field(
    field_type: ts.FieldType,
    args: list[ts.TypeSpec],
    kwargs: dict[str, ts.TypeSpec],
) -> Iterator[str]:
    if len(args) < 1:
        yield f"Function takes at least 1 argument, but {len(args)} were given."
        return
    for arg in args:
        if not isinstance(arg, ts.OffsetType):
            yield f"Expected arguments to be of type '{ts.OffsetType}', got '{arg}'."
            return
        if len(args) > 1 and len(arg.target) > 1:
            yield f"Function takes only 1 argument in unstructured case, but {len(args)} were given."
            return

    if kwargs:
        yield f"Got unexpected keyword argument(s) '{', '.join(kwargs.keys())}'."
        return

    source_dim = args[0].source  # type: ignore[attr-defined] # ensured by loop above
    target_dims = args[0].target  # type: ignore[attr-defined] # ensured by loop above
    # TODO: This code does not handle ellipses for dimensions. Fix it.
    assert field_type.dims is not ...
    if field_type.dims and source_dim not in field_type.dims:
        yield (
            f"Incompatible offset can not shift field defined on "
            f"{', '.join([dim.value for dim in field_type.dims])} from "
            f"{source_dim.value} to target dim(s): "
            f"{', '.join([dim.value for dim in target_dims])}"
        )


def accepts_args(
    callable_type: ts.CallableType,
    *,
    with_args: Sequence[ts.TypeSpec],
    with_kwargs: dict[str, ts.TypeSpec],
    raise_exception: bool = False,
) -> bool:
    """
    Check if a function can be called for given arguments.

    If ``raise_exception`` is given a :class:`ValueError` is raised with a
    detailed description of why the function is not callable.

    Note that all types must be concrete/complete.

    Examples:
        >>> bool_type = ts.ScalarType(kind=ts.ScalarKind.BOOL)
        >>> func_type = ts.FunctionType(
        ...     pos_only_args=[bool_type],
        ...     pos_or_kw_args={"foo": bool_type},
        ...     kw_only_args={},
        ...     returns=ts.VoidType(),
        ... )
        >>> accepts_args(func_type, with_args=[bool_type], with_kwargs={"foo": bool_type})
        True
        >>> accepts_args(func_type, with_args=[], with_kwargs={})
        False
    """
    if not isinstance(callable_type, ts.CallableType):
        if raise_exception:
            raise ValueError(f"Expected a callable type, got '{callable_type}'.")
        return False

    errors = function_signature_incompatibilities(callable_type, with_args, with_kwargs)
    if raise_exception:
        error_list = list(errors)
        if len(error_list) > 0:
            raise ValueError(
                f"Invalid call to function of type '{callable_type}':\n"
                + ("\n".join([f"  - {error}" for error in error_list]))
            )
        return True

    return next(errors, None) is None
