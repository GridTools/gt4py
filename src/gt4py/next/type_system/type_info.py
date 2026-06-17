# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import types
from collections.abc import Callable, Collection, Iterable, Iterator
from typing import Any, Final, Literal, Sequence, Type, TypeGuard, TypeVar, cast, overload

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing as xtyping, utils
from gt4py.next import common, utils as next_utils
from gt4py.next.iterator.type_system import type_specifications as it_ts
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


def _type_params(symbol_type: ts.TypeSpec) -> tuple[ts.TypeSpec, ...]:
    """Return the immediate type-parameter sub-types of ``symbol_type``.

    These are its dtype, element type, tuple elements, or function argument / return types.
    """
    match symbol_type:
        case ts.FieldType(dtype=dtype):
            return (dtype,)
        case ts.ListType(element_type=element_type):
            return (element_type,)
        case ts.TupleType(types=types) | ts.NamedCollectionType(types=types):
            return tuple(types)
        case ts.FunctionType():
            return (
                *symbol_type.pos_only_args,
                *symbol_type.pos_or_kw_args.values(),
                *symbol_type.kw_only_args.values(),
                symbol_type.returns,
            )
    # callable type wrappers (e.g. the field operator types in `ffront`) carry their
    # signature in a `definition` attribute
    if isinstance(definition := getattr(symbol_type, "definition", None), ts.TypeSpec):
        return (definition,)
    return ()


def is_generic(symbol_type: ts.TypeSpec) -> bool:
    """
    Figure out if a type contains parts that are only known when concrete arguments are given.

    Recurses into composite types, reporting ``True`` if any nested part is a `DeferredType` or
    `TypeVarType`. Unlike :func:`is_concrete` (a shallow top-level check), this is deep, so a
    tuple with a nested `DeferredType` is both concrete and generic.

    Note: this returns ``True`` for a bare ``astype`` constructor type, whose ``definition``
    carries a ``DeferredType`` by design; callers that only care about *data* arguments must
    filter for ``ts.DataType`` themselves.

    Examples:
        >>> is_generic(ts.DeferredType(constraint=None))
        True

        >>> bool_type = ts.ScalarType(kind=ts.ScalarKind.BOOL)
        >>> is_generic(bool_type)
        False

        >>> is_generic(ts.TupleType(types=[bool_type, ts.DeferredType(constraint=None)]))
        True
    """
    if isinstance(symbol_type, (ts.DeferredType, ts.TypeVarType)):
        return True
    return any(is_generic(p) for p in _type_params(symbol_type))


def type_class(symbol_type: ts.TypeSpec) -> Type[ts.TypeSpec]:
    """
    Determine which class should be used to create a compatible concrete type.

    Examples:
        >>> type_class(ts.DeferredType(constraint=ts.ScalarType)).__name__
        'ScalarType'

        >>> type_class(ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.BOOL))).__name__
        'FieldType'

        >>> type_class(ts.TupleType(types=[])).__name__
        'TupleType'
    """
    if isinstance(symbol_type, ts.DeferredType):
        constraint = symbol_type.constraint
        if constraint is None:
            raise ValueError(f"No type information available for '{symbol_type}'.")
        elif isinstance(constraint, tuple):
            raise ValueError(f"Not sufficient type information available for '{symbol_type}'.")
        return constraint
    if isinstance(symbol_type, ts.TypeSpec):
        return symbol_type.__class__
    raise ValueError(
        f"Invalid type for TypeInfo: requires '{ts.TypeSpec}', got '{type(symbol_type)}'."
    )


@overload
def primitive_constituents(
    symbol_type: ts.TypeSpec, with_path_arg: Literal[False] = False
) -> utils.XIterable[ts.TypeSpec]: ...


@overload
def primitive_constituents(
    symbol_type: ts.TypeSpec, with_path_arg: Literal[True]
) -> utils.XIterable[tuple[ts.TypeSpec, tuple[int, ...]]]: ...


def primitive_constituents(
    symbol_type: ts.TypeSpec, with_path_arg: bool = False
) -> utils.XIterable[ts.TypeSpec] | utils.XIterable[tuple[ts.TypeSpec, tuple[int, ...]]]:
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
        if isinstance(symbol_type, ts.COLLECTION_TYPE_SPECS):
            symbol_type = cast(
                ts.CollectionTypeSpec, symbol_type
            )  # This shouldn't be needed after the previous isinstance() check
            for i, el_type in enumerate(symbol_type.types):
                yield from constituents_yielder(el_type, (*path, i))
        else:
            if with_path_arg:
                yield (symbol_type, path)
            else:
                yield symbol_type

    return utils.xiter(constituents_yielder(symbol_type, ()))  # type: ignore[return-value] # why resolved to XIterable[object]?


_T = TypeVar("_T")
_C = TypeVar("_C")


def tree_map_type_constructor(
    value: ts.CollectionTypeSpecT,
    elems: Iterable[ts.DataType | ts.DimensionType | ts.DeferredType],
) -> ts.CollectionTypeSpecT:
    return (
        ts.NamedCollectionType(
            keys=value.keys, original_python_type=value.original_python_type, types=list(elems)
        )
        if isinstance(value, ts.NamedCollectionType)
        else ts.TupleType(types=list(elems))  # type: ignore[return-value]
    )


@overload
def tree_map_type(
    fun: Callable[..., _T], *, with_path_arg: bool = ..., unpack: bool = ...
) -> Callable[..., _T | ts.CollectionTypeSpec]: ...


@overload
def tree_map_type(
    fun: Callable[..., _T],
    *,
    result_collection_constructor: Callable[..., _C],
    with_path_arg: bool = ...,
    unpack: bool = ...,
) -> Callable[..., _T | _C]: ...


def tree_map_type(
    fun: Callable[..., _T],
    *,
    result_collection_constructor: Callable[..., Any] = tree_map_type_constructor,
    with_path_arg: bool = False,
    unpack: bool = False,
) -> Callable[..., Any]:
    return next_utils.tree_map(
        fun,
        collection_type=ts.COLLECTION_TYPE_SPECS,
        result_collection_constructor=result_collection_constructor,
        with_path_arg=with_path_arg,
        unpack=unpack,
    )


def extract_dtype(symbol_type: ts.TypeSpec) -> ts.ScalarType | ts.ListType | ts.TypeVarType:
    """
    Extract the data type from ``symbol_type`` if it is `FieldType`, `ScalarType` or `TypeVarType`.

    Raise an error if no dtype can be found or the result would be ambiguous.

    Examples:
        >>> print(extract_dtype(ts.ScalarType(kind=ts.ScalarKind.FLOAT64)))
        float64

        >>> print(
        ...     extract_dtype(ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.BOOL)))
        ... )
        bool
    """
    match symbol_type:
        case ts.FieldType(dtype=dtype):
            return dtype
        case ts.ScalarType() as dtype:
            return dtype
        case ts.TypeVarType() as dtype:
            return dtype
    raise ValueError(f"Can not unambiguosly extract data type from '{symbol_type}'.")


def _scalar_kinds(scalar_types: tuple[type, ...]) -> frozenset[ts.ScalarKind]:
    # Derived from the canonical scalar-type tuples in `gt4py._core.definitions` so the two
    # stay in sync; the `int`/`float` builtins collapse onto their fixed-width kind.
    return frozenset(ts.ScalarKind[np.dtype(t).name.upper()] for t in scalar_types)


_FLOATING_POINT_KINDS: Final[frozenset[ts.ScalarKind]] = _scalar_kinds(core_defs.FLOAT_TYPES)
_INTEGRAL_KINDS: Final[frozenset[ts.ScalarKind]] = _scalar_kinds(core_defs.INTEGRAL_TYPES)


def _is_field_or_scalar_of_kind(symbol_type: ts.TypeSpec, kinds: Collection[ts.ScalarKind]) -> bool:
    """Check if ``symbol_type`` is a scalar or a field whose dtype kind is in ``kinds``.

    A type variable has the property iff all of its constraints have it.
    """
    if isinstance(symbol_type, ts.TypeVarType):
        return all(_is_field_or_scalar_of_kind(c, kinds) for c in symbol_type.constraints)
    if not isinstance(symbol_type, (ts.ScalarType, ts.FieldType)):
        return False
    dtype = extract_dtype(symbol_type)
    if isinstance(dtype, ts.TypeVarType):
        return all(_is_field_or_scalar_of_kind(c, kinds) for c in dtype.constraints)
    return isinstance(dtype, ts.ScalarType) and dtype.kind in kinds


def is_floating_point(symbol_type: ts.TypeSpec) -> bool:
    """
    Check if the dtype of ``symbol_type`` is a floating point type.

    Examples:
        >>> is_floating_point(ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
        True
        >>> is_floating_point(ts.ScalarType(kind=ts.ScalarKind.FLOAT32))
        True
        >>> is_floating_point(ts.ScalarType(kind=ts.ScalarKind.INT32))
        False
        >>> is_floating_point(
        ...     ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32))
        ... )
        True
    """
    return _is_field_or_scalar_of_kind(symbol_type, _FLOATING_POINT_KINDS)


def is_integral_scalar(symbol_type: ts.TypeSpec) -> bool:
    """
    Check if ``symbol_type`` is an integral scalar (rejecting fields).

    Examples:
        >>> is_integral_scalar(ts.ScalarType(kind=ts.ScalarKind.INT32))
        True
        >>> is_integral_scalar(ts.ScalarType(kind=ts.ScalarKind.UINT64))
        True
        >>> is_integral_scalar(ts.ScalarType(kind=ts.ScalarKind.FLOAT32))
        False
        >>> is_integral_scalar(ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32)))
        False
    """
    return isinstance(symbol_type, ts.ScalarType) and symbol_type.kind in _INTEGRAL_KINDS


def is_integral(symbol_type: ts.TypeSpec) -> bool:
    """
    Check if the dtype of ``symbol_type`` is an integral type.

    Examples:
        >>> is_integral(ts.ScalarType(kind=ts.ScalarKind.INT32))
        True
        >>> is_integral(ts.ScalarType(kind=ts.ScalarKind.FLOAT32))
        False
        >>> is_integral(ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32)))
        True
    """
    return _is_field_or_scalar_of_kind(symbol_type, _INTEGRAL_KINDS)


def is_arithmetic_scalar(symbol_type: ts.TypeSpec) -> bool:
    """
    Check if ``symbol_type`` is an arithmetic scalar (rejecting fields).

    Examples:
        >>> is_arithmetic_scalar(ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
        True
        >>> is_arithmetic_scalar(ts.ScalarType(kind=ts.ScalarKind.INT32))
        True
        >>> is_arithmetic_scalar(ts.ScalarType(kind=ts.ScalarKind.BOOL))
        False
        >>> is_arithmetic_scalar(
        ...     ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.INT64))
        ... )
        False
    """
    if not isinstance(symbol_type, (ts.ScalarType, ts.TypeVarType)):
        return False
    return is_arithmetic(symbol_type)


def is_logical(symbol_type: ts.TypeSpec) -> bool:
    """
    Check if the dtype of ``symbol_type`` is a boolean type.

    Examples:
        >>> is_logical(ts.ScalarType(kind=ts.ScalarKind.BOOL))
        True
        >>> is_logical(ts.ScalarType(kind=ts.ScalarKind.INT32))
        False
        >>> is_logical(ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.BOOL)))
        True
    """
    return _is_field_or_scalar_of_kind(symbol_type, (ts.ScalarKind.BOOL,))


def is_arithmetic(symbol_type: ts.TypeSpec) -> bool:
    """
    Check if ``symbol_type`` is compatible with arithmetic operations.

    Examples:
        >>> is_arithmetic(ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
        True
        >>> is_arithmetic(ts.ScalarType(kind=ts.ScalarKind.BOOL))
        False
        >>> is_arithmetic(ts.ScalarType(kind=ts.ScalarKind.STRING))
        False
        >>> is_arithmetic(ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32)))
        True
    """
    # `is_arithmetic` cannot reuse `_is_field_or_scalar_of_kind`'s "all constraints
    #  share the kind" rule: a type variable is arithmetic if every constraint is
    #  arithmetic, even when the constraints mix floating point and integral kinds.
    if isinstance(symbol_type, ts.TypeVarType):
        return all(is_arithmetic(c) for c in symbol_type.constraints)
    if isinstance(symbol_type, (ts.ScalarType, ts.FieldType)) and isinstance(
        dtype := extract_dtype(symbol_type), ts.TypeVarType
    ):
        return is_arithmetic(dtype)
    return is_floating_point(symbol_type) or is_integral(symbol_type)


def arithmetic_bounds(arithmetic_type: ts.ScalarType) -> tuple[np.number, np.number]:
    assert is_arithmetic(arithmetic_type)
    return {  # type: ignore[return-value] # why resolved to `tuple[object, object]`?
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
    }[arithmetic_type.kind]


def is_type_or_tuple_of_type(type_: ts.TypeSpec, expected_type: type | tuple) -> bool:
    """
    Return True if ``type_`` matches any of the expected or is a tuple of them.

    Examples:
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
        >>> extract_dims(ts.ScalarType(kind=ts.ScalarKind.INT64, shape=[3, 4]))
        []
        >>> I = common.Dimension(value="I")
        >>> J = common.Dimension(value="J")
        >>> extract_dims(ts.FieldType(dims=[I, J], dtype=ts.ScalarType(kind=ts.ScalarKind.INT64)))
        [Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>), Dimension(value='J', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)]
    """
    if isinstance(symbol_type, (ts.ScalarType, ts.TypeVarType)):
        return []
    if isinstance(symbol_type, ts.FieldType):
        return symbol_type.dims
    raise ValueError(f"Can not extract dimensions from '{symbol_type}'.")


def is_local_field(type_: ts.FieldType) -> bool:
    """
    Return if `type_` is a field defined on a local dimension.

    Examples:
        >>> V = common.Dimension(value="V")
        >>> V2E = common.Dimension(value="V2E", kind=common.DimensionKind.LOCAL)
        >>> is_local_field(
        ...     ts.FieldType(dims=[V, V2E], dtype=ts.ScalarType(kind=ts.ScalarKind.INT64))
        ... )
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


# TODO(tehrengruber): This function has specializations on Iterator types, which are not part of
#  the general / shared type system. This functionality should be moved to the iterator-only
#  type system, but we need some sort of multiple dispatch for that.
# TODO(tehrengruber): Should this have a direction like is_concretizable?
def is_compatible_type(type_a: ts.TypeSpec, type_b: ts.TypeSpec) -> bool:
    """
    Predicate to determine if two types are compatible.

    This function gracefully handles:
     - iterators with unknown positions which are considered compatible to any other positions
       of another iterator.
     - iterators which are defined everywhere, i.e. empty defined dimensions
    Beside that this function simply checks for equality of types.

    >>> bool_type = ts.ScalarType(kind=ts.ScalarKind.BOOL)
    >>> IDim = common.Dimension(value="IDim")
    >>> type_on_i_of_i_it = it_ts.IteratorType(
    ...     position_dims=[IDim], defined_dims=[IDim], element_type=bool_type
    ... )
    >>> type_on_undefined_of_i_it = it_ts.IteratorType(
    ...     position_dims="unknown", defined_dims=[IDim], element_type=bool_type
    ... )
    >>> is_compatible_type(type_on_i_of_i_it, type_on_undefined_of_i_it)
    True

    >>> JDim = common.Dimension(value="JDim")
    >>> type_on_j_of_j_it = it_ts.IteratorType(
    ...     position_dims=[JDim], defined_dims=[JDim], element_type=bool_type
    ... )
    >>> is_compatible_type(type_on_i_of_i_it, type_on_j_of_j_it)
    False
    """
    is_compatible = True

    if isinstance(type_a, it_ts.IteratorType) and isinstance(type_b, it_ts.IteratorType):
        if not any(el_type.position_dims == "unknown" for el_type in [type_a, type_b]):
            is_compatible &= type_a.position_dims == type_b.position_dims
        if type_a.defined_dims and type_b.defined_dims:
            is_compatible &= type_a.defined_dims == type_b.defined_dims
        is_compatible &= type_a.element_type == type_b.element_type
    elif isinstance(type_a, ts.TupleType) and isinstance(type_b, ts.TupleType):
        if len(type_a.types) != len(type_b.types):
            return False
        for el_type_a, el_type_b in zip(type_a.types, type_b.types, strict=True):
            is_compatible &= is_compatible_type(el_type_a, el_type_b)
    elif isinstance(type_a, ts.NamedCollectionType) and isinstance(type_b, ts.NamedCollectionType):
        if type_a.keys != type_b.keys:
            return False
        if (
            not any(
                python_type is ts.ANY_PYTHON_TYPE_NAME
                for python_type in [type_a.original_python_type, type_b.original_python_type]
            )
            and type_a.original_python_type != type_b.original_python_type
        ):
            return False
        for el_type_a, el_type_b in zip(type_a.types, type_b.types, strict=True):
            is_compatible &= is_compatible_type(el_type_a, el_type_b)
    elif isinstance(type_a, ts.FunctionType) and isinstance(type_b, ts.FunctionType):
        for arg_a, arg_b in zip(type_a.pos_only_args, type_b.pos_only_args, strict=True):
            is_compatible &= is_compatible_type(arg_a, arg_b)
        for arg_a, arg_b in zip(
            type_a.pos_or_kw_args.values(), type_b.pos_or_kw_args.values(), strict=True
        ):
            is_compatible &= is_compatible_type(arg_a, arg_b)
        for arg_a, arg_b in zip(
            type_a.kw_only_args.values(), type_b.kw_only_args.values(), strict=True
        ):
            is_compatible &= is_compatible_type(arg_a, arg_b)
        is_compatible &= is_compatible_type(type_a.returns, type_b.returns)
    else:
        is_compatible &= is_concretizable(type_a, type_b)

    return is_compatible


def is_concretizable(symbol_type: ts.TypeSpec, to_type: ts.TypeSpec) -> bool:
    """
    Check if ``symbol_type`` can be concretized to ``to_type``.

    Examples:
        >>> is_concretizable(
        ...     ts.ScalarType(kind=ts.ScalarKind.INT64),
        ...     to_type=ts.ScalarType(kind=ts.ScalarKind.INT64),
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


def _bind_var(var: ts.TypeVarType, dtype: ts.TypeSpec) -> dict[str, ts.ScalarType]:
    if not isinstance(dtype, ts.ScalarType):
        # not a concrete scalar to bind to -- e.g. a `TypeVarType` (operator-from-operator
        # call), a `DeferredType` (scan), or a `ListType` (local field). Leave it unbound;
        # the caller is responsible for checking that no type variable remained unbound.
        return {}
    if dtype not in var.constraints:
        raise ValueError(f"'{dtype}' does not satisfy the constraints of type variable '{var}'.")
    return {var.name: dtype}


def _merge_bindings(parts: Iterable[dict[str, ts.ScalarType]]) -> dict[str, ts.ScalarType]:
    binding: dict[str, ts.ScalarType] = {}
    for part in parts:
        for name, dtype in part.items():
            if (previous := binding.get(name)) is not None and previous != dtype:
                raise ValueError(
                    f"Type variable '{name}' is bound inconsistently:"
                    f" '{previous}' and '{dtype}' (all arguments using '{name}'"
                    " must have the same dtype)."
                )
            binding[name] = dtype
    return binding


def _bind(param: ts.TypeSpec, arg: ts.TypeSpec) -> dict[str, ts.ScalarType]:
    match param:
        case ts.TypeVarType() as var:
            return _bind_var(var, arg)
        case ts.FieldType(dtype=ts.TypeVarType() as var):
            # scalar arguments are promoted to zero-dimensional fields
            return _bind_var(var, arg.dtype if isinstance(arg, ts.FieldType) else arg)
        case ts.ListType(element_type=element_type) if isinstance(arg, ts.ListType):
            return _bind(element_type, arg.element_type)
        case ts.TupleType() | ts.NamedCollectionType() if isinstance(
            arg, (ts.TupleType, ts.NamedCollectionType)
        ):
            # tolerant by design: a structural mismatch (e.g. tuple vs scalar) binds nothing
            # here and is reported by the regular signature checks instead.
            return _merge_bindings(_bind(p, a) for p, a in zip(param.types, arg.types))
    return {}


def bind_type_vars(
    params: Sequence[ts.TypeSpec], args: Sequence[ts.TypeSpec]
) -> dict[str, ts.ScalarType]:
    """
    Compute a binding of all type variables in ``params`` by structurally matching ``args``.

    Concrete (non-generic) parts of the parameters are ignored; a type variable position binds
    only if the corresponding argument provides a concrete scalar dtype. The caller is
    responsible for checking that no type variable remained unbound.

    Raises:
        ValueError: If a type variable would be bound inconsistently or to a dtype that is
            not one of its constraints.

    Examples:
        >>> var = ts.TypeVarType(name="T", constraints=(ts.ScalarType(kind=ts.ScalarKind.FLOAT64),))
        >>> I = common.Dimension(value="I")
        >>> binding = bind_type_vars(
        ...     [ts.FieldType(dims=[I], dtype=var)],
        ...     [ts.FieldType(dims=[I], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))],
        ... )
        >>> print(binding["T"])
        float64
    """
    return _merge_bindings(_bind(param, arg) for param, arg in zip(params, args))


def tree_map_type_params(
    fun: Callable[[ts.TypeSpec], ts.TypeSpec], symbol_type: ts.TypeSpec
) -> ts.TypeSpec:
    """Rebuild ``symbol_type`` applying ``fun`` to each immediate type-parameter sub-type.

    Leaf types are returned unchanged.
    """
    match symbol_type:
        case ts.FieldType(dims=dims, dtype=dtype):
            new_dtype = fun(dtype)
            assert isinstance(new_dtype, (ts.ScalarType, ts.ListType, ts.TypeVarType))
            return ts.FieldType(dims=dims, dtype=new_dtype)
        case ts.ListType(element_type=element_type, offset_type=offset_type):
            new_element_type = fun(element_type)
            assert isinstance(new_element_type, ts.DataType)
            return ts.ListType(element_type=new_element_type, offset_type=offset_type)
        case ts.TupleType(types=types):
            return ts.TupleType(types=[fun(t) for t in types])
        case ts.NamedCollectionType(types=types):
            return ts.NamedCollectionType(
                types=[fun(t) for t in types],
                keys=symbol_type.keys,
                original_python_type=symbol_type.original_python_type,
            )
        case ts.FunctionType():
            return ts.FunctionType(
                pos_only_args=[fun(t) for t in symbol_type.pos_only_args],
                pos_or_kw_args={name: fun(t) for name, t in symbol_type.pos_or_kw_args.items()},
                kw_only_args={name: fun(t) for name, t in symbol_type.kw_only_args.items()},
                returns=fun(symbol_type.returns),
            )
    return symbol_type


def substitute_type_vars(
    type_: ts.TypeSpec, binding: xtyping.Mapping[str, ts.ScalarType]
) -> ts.TypeSpec:
    """
    Replace all type variables in ``type_`` that are bound in ``binding``.

    Unbound type variables and all other generic parts (e.g. `DeferredType`) are kept as-is.

    Examples:
        >>> var = ts.TypeVarType(name="T", constraints=(ts.ScalarType(kind=ts.ScalarKind.FLOAT64),))
        >>> I = common.Dimension(value="I")
        >>> print(
        ...     substitute_type_vars(
        ...         ts.FieldType(dims=[I], dtype=var),
        ...         {"T": ts.ScalarType(kind=ts.ScalarKind.FLOAT64)},
        ...     )
        ... )
        Field[[I], float64]
    """
    if not binding or not is_generic(type_):
        return type_
    if isinstance(type_, ts.TypeVarType):
        return binding.get(type_.name, type_)
    return tree_map_type_params(lambda t: substitute_type_vars(t, binding), type_)


def promote(
    *types: ts.FieldType | ts.ScalarType | ts.TypeVarType, always_field: bool = False
) -> ts.FieldType | ts.ScalarType | ts.TypeVarType:
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

    >>> promoted: ts.FieldType = promote(
    ...     ts.FieldType(dims=[I, J], dtype=dtype), ts.FieldType(dims=[K], dtype=dtype)
    ... )
    >>> promoted.dims == [I, J, K] and promoted.dtype == dtype
    True
    """
    if not always_field and all(
        isinstance(type_, (ts.ScalarType, ts.TypeVarType)) for type_ in types
    ):
        if not all(type_ == types[0] for type_ in types):
            if any(isinstance(type_, ts.TypeVarType) for type_ in types):
                distinct_types = "', '".join(str(t) for t in dict.fromkeys(types))
                raise ValueError(
                    f"Could not promote '{distinct_types}': a generic dtype (type variable)"
                    " can only be combined with values of the same type variable,"
                    " not with other dtypes."
                )
            raise ValueError("Could not promote scalars of different dtype (not implemented).")
        if not all(type_.shape is None for type_ in types if isinstance(type_, ts.ScalarType)):
            raise NotImplementedError("Shape promotion not implemented.")
        return types[0]
    elif all(isinstance(type_, (ts.ScalarType, ts.FieldType, ts.TypeVarType)) for type_ in types):
        dims = common.promote_dims(*(extract_dims(type_) for type_ in types))
        extracted_dtypes = [extract_dtype(type_) for type_ in types]
        assert all(isinstance(dtype, (ts.ScalarType, ts.TypeVarType)) for dtype in extracted_dtypes)
        dtype = cast(  # type variables promote like scalars (only with themselves)
            ts.ScalarType | ts.TypeVarType,
            promote(*extracted_dtypes),  # type: ignore[arg-type] # checked above
        )

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
        f"Return type deduction of type '{type(callable_type).__name__}' not implemented."
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


@return_type.register
def return_type_constructor(
    constructor_type: ts.ConstructorType,
    *,
    with_args: Sequence[ts.TypeSpec],
    with_kwargs: dict[str, ts.TypeSpec],
) -> ts.TypeSpec:
    return constructor_type.definition.returns


UNDEFINED_ARG = types.new_class("UNDEFINED_ARG")


# TODO(egparedes): replace by `inspect.Signature.bind()`
@functools.singledispatch
def canonicalize_arguments(
    func_type: ts.CallableType,
    args: Sequence,
    kwargs: dict,
    *,
    ignore_errors: bool = False,
) -> tuple[tuple, dict]:
    raise NotImplementedError(f"Not implemented for type '{type(func_type).__name__}'.")


@canonicalize_arguments.register
def canonicalize_function_arguments(
    func_type: ts.FunctionType,
    args: Sequence,
    kwargs: dict,
    *,
    ignore_errors: bool = False,
) -> tuple[tuple, dict]:
    num_pos_only_args = len(func_type.pos_only_args)
    num_pos_or_kw_args = len(func_type.pos_or_kw_args)
    canonical_args = [*args] + (
        [UNDEFINED_ARG] * max(num_pos_only_args + num_pos_or_kw_args - len(args), 0)
    )
    remaining_kwargs = {**kwargs}

    pos_or_kw_args_names = [*func_type.pos_or_kw_args]
    for name in kwargs:
        if name in func_type.pos_or_kw_args:
            args_idx = num_pos_only_args + pos_or_kw_args_names.index(name)
            if canonical_args[args_idx] is UNDEFINED_ARG:
                canonical_args[args_idx] = remaining_kwargs.pop(name)
            elif not ignore_errors:
                raise ValueError(
                    f"Error canonicalizing function arguments. Got multiple values for argument '{name}'."
                )

    missing_kw_args = func_type.kw_only_args.keys() - remaining_kwargs.keys()
    invalid_kw_args = remaining_kwargs.keys() - func_type.kw_only_args.keys()

    if not ignore_errors:
        if missing_kw_args:
            raise ValueError(f"Missing required keyword arguments: {[*missing_kw_args]}.")
        if invalid_kw_args:
            raise ValueError(f"Invalid keyword arguments: {[*invalid_kw_args]}.")

    # Sort remaining keyword arguments in the signature ordering,
    # keeping the invalid ones at the end if 'ignore_errors' is True.
    canonical_kwargs = {
        k: remaining_kwargs[k] for k in func_type.kw_only_args if k in remaining_kwargs
    } | {k: remaining_kwargs[k] for k in invalid_kw_args}

    return tuple(canonical_args), canonical_kwargs


@canonicalize_arguments.register(ts.ConstructorType)
def canonicalize_constructor_arguments(
    constructor_type: ts.ConstructorType,
    args: tuple | list,
    kwargs: dict,
    *,
    ignore_errors: bool = False,
) -> tuple[tuple, dict]:
    return canonicalize_arguments(
        constructor_type.definition,
        args,
        kwargs,
        ignore_errors=ignore_errors,
    )


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
    for name in [*kwargs]:
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

    missing_positional_args = [
        f"'{arg_type}'"
        for arg, arg_type in zip(
            args[len(func_type.pos_only_args) : num_pos_params],
            func_type.pos_or_kw_args.keys(),
        )
        if arg is UNDEFINED_ARG
    ]

    if missing_positional_args:
        yield f"Missing {len(missing_positional_args)} required positional argument{'s' if len(missing_positional_args) != 1 else ''}: {', '.join(missing_positional_args)}"

    # check for missing or extra keyword arguments
    kw_a_m_b = func_type.kw_only_args.keys() - kwargs.keys()
    if len(kw_a_m_b) > 0:
        yield f"Missing required keyword argument{'s' if len(kw_a_m_b) != 1 else ''} '{', '.join(kw_a_m_b)}'."
    kw_b_m_a = kwargs.keys() - func_type.kw_only_args.keys()
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

    num_pos_only_args = len(func_type.pos_only_args)
    num_pos_or_kw_args = len(func_type.pos_or_kw_args)
    num_pos_params = num_pos_only_args + num_pos_or_kw_args
    assert len(args) >= num_pos_params
    for i, (a_arg, b_arg) in enumerate(
        zip(list(func_type.pos_only_args) + list(func_type.pos_or_kw_args.values()), args)
    ):
        if b_arg is not UNDEFINED_ARG and a_arg != b_arg and not is_compatible_type(a_arg, b_arg):
            if i < num_pos_only_args:
                arg_repr = f"{_number_to_ordinal_number(i + 1)} argument"
            else:
                arg_repr = (
                    f"argument '{list(func_type.pos_or_kw_args.keys())[i - num_pos_only_args]}'"
                )
            yield f"Expected {arg_repr} to be of type '{a_arg}', got '{b_arg}'."

    for kwarg in func_type.kw_only_args.keys() & kwargs.keys():
        if (a_kwarg := func_type.kw_only_args[kwarg]) != (
            b_kwarg := kwargs[kwarg]
        ) and not is_compatible_type(a_kwarg, b_kwarg):
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
    assert field_type.dims is not ...
    if field_type.dims and source_dim not in field_type.dims:
        yield (
            f"Incompatible offset can not shift field defined on "
            f"{', '.join([dim.value for dim in field_type.dims])} from "
            f"{source_dim.value} to target dim(s): "
            f"{', '.join([dim.value for dim in target_dims])}"
        )


@function_signature_incompatibilities.register
def function_signature_incompatibilities_constructor(
    constructor_type: ts.ConstructorType,
    args: Sequence[ts.TypeSpec],
    kwargs: dict[str, ts.TypeSpec],
) -> Iterator[str]:
    yield from function_signature_incompatibilities_func(constructor_type.definition, args, kwargs)


# TODO(havogt): Consider inlining the usage of this function in the call sites
# to get rid of the `raise_exception` case and because the error message here is possibly too specific.
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
        if len(error_list := [*errors]) > 0:
            raise ValueError(
                f"Invalid call to function of type '{callable_type}':\n"
                + ("\n".join([f"  - {error}" for error in error_list]))
            )
        return True

    return next(errors, None) is None


def needs_value_extraction(
    type_spec: ts.TypeSpec,
) -> xtyping.TypeIs[ts.NamedCollectionType | ts.TupleType]:
    return isinstance(type_spec, ts.NamedCollectionType) or (
        isinstance(type_spec, ts.TupleType)
        and any(needs_value_extraction(t) for t in type_spec.types)
    )
