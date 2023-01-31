# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

import functools
from typing import Callable, Iterator, Type, TypeGuard, cast

from gt4py.eve.utils import XIterable, xiter
from gt4py.next.common import Dimension, GTTypeError
from gt4py.next.type_system import type_specifications as ts


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
                raise GTTypeError(f"No type information available for {symbol_type}!")
            elif isinstance(constraint, tuple):
                raise GTTypeError(f"Not sufficient type information available for {symbol_type}!")
            return constraint
        case ts.TypeSpec() as concrete_type:
            return concrete_type.__class__
    raise GTTypeError(
        f"Invalid type for TypeInfo: requires {ts.TypeSpec}, got {type(symbol_type)}!"
    )


def primitive_constituents(
    symbol_type: ts.TypeSpec,
) -> XIterable[ts.TypeSpec]:
    """
    Return the primitive types contained in a composite type.

    >>> from gt4py.next.common import Dimension
    >>> I = Dimension(value="I")
    >>> int_type = ts.ScalarType(kind=ts.ScalarKind.INT)
    >>> field_type = ts.FieldType(dims=[I], dtype=int_type)

    >>> tuple_type = ts.TupleType(types=[int_type, field_type])
    >>> primitive_constituents(tuple_type).to_list()  # doctest: +ELLIPSIS
    [ScalarType(...), FieldType(...)]

    >>> nested_tuple = ts.TupleType(types=[field_type, tuple_type])
    >>> primitive_constituents(nested_tuple).to_list()  # doctest: +ELLIPSIS
    [FieldType(...), ScalarType(...), FieldType(...)]
    """

    def constituents_yielder(symbol_type: ts.TypeSpec):
        if isinstance(symbol_type, ts.TupleType):
            for el_type in symbol_type.types:
                yield from constituents_yielder(el_type)
        else:
            yield symbol_type

    return xiter(constituents_yielder(symbol_type))


def apply_to_primitive_constituents(
    symbol_type: ts.TypeSpec,
    fun: Callable[[ts.TypeSpec], ts.TypeSpec]
    | Callable[[ts.TypeSpec, tuple[int, ...]], ts.TypeSpec],
    with_path_arg=False,
    _path=(),
):
    """
    Apply function to all primitive constituents of a type.

    >>> int_type = ts.ScalarType(kind=ts.ScalarKind.INT)
    >>> tuple_type = ts.TupleType(types=[int_type, int_type])
    >>> print(apply_to_primitive_constituents(tuple_type, lambda primitive_type: ts.FieldType(dims=[], dtype=primitive_type)))
    tuple[Field[[], int64], Field[[], int64]]
    """
    if isinstance(symbol_type, ts.TupleType):
        return ts.TupleType(
            types=[
                apply_to_primitive_constituents(
                    el, fun, _path=(*_path, i), with_path_arg=with_path_arg
                )
                for i, el in enumerate(symbol_type.types)
            ]
        )
    if with_path_arg:
        return fun(symbol_type, _path)  # type: ignore[call-arg] # mypy not aware of `with_path_arg`
    else:
        return fun(symbol_type)  # type: ignore[call-arg] # mypy not aware of `with_path_arg`


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
    raise GTTypeError(f"Can not unambiguosly extract data type from {symbol_type}!")


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
        ts.ScalarKind.FLOAT32,
        ts.ScalarKind.FLOAT64,
    ]


def is_integral(symbol_type: ts.TypeSpec) -> bool:
    """
    Check if the dtype of ``symbol_type`` is an integral type.

    Examples:
    ---------
    >>> is_integral(ts.ScalarType(kind=ts.ScalarKind.INT))
    True
    >>> is_integral(ts.ScalarType(kind=ts.ScalarKind.INT32))
    True
    >>> is_integral(ts.ScalarType(kind=ts.ScalarKind.FLOAT32))
    False
    >>> is_integral(ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.INT)))
    True
    """
    return extract_dtype(symbol_type).kind in [
        ts.ScalarKind.INT,
        ts.ScalarKind.INT32,
        ts.ScalarKind.INT64,
    ]


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
    >>> is_number(ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.INT)))
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


def is_type_or_tuple_of_type(type_: ts.TypeSpec, expected_type: type | tuple) -> bool:
    """
    Return True if ``type_`` matches any of the expected.

    Examples:
    ---------
    >>> scalar_type = ts.ScalarType(kind=ts.ScalarKind.INT)
    >>> field_type = ts.FieldType(dims=[], dtype=scalar_type)
    >>> is_type_or_tuple_of_type(field_type, ts.FieldType)
    True
    >>> is_type_or_tuple_of_type(ts.TupleType(types=[scalar_type, field_type]), (ts.ScalarType, ts.FieldType))
    True
    >>> is_type_or_tuple_of_type(scalar_type, ts.FieldType)
    False
    >>> is_type_or_tuple_of_type(ts.TupleType(types=[scalar_type, field_type]), ts.FieldType)
    False
    """
    return all(isinstance(t, expected_type) for t in primitive_constituents(type_))


def extract_dims(symbol_type: ts.TypeSpec) -> list[Dimension]:
    """
    Try to extract field dimensions if possible.

    Scalars are treated as zero-dimensional

    Examples:
    ---------
    >>> extract_dims(ts.ScalarType(kind=ts.ScalarKind.INT64, shape=[3, 4]))
    []
    >>> I = Dimension(value="I")
    >>> J = Dimension(value="J")
    >>> extract_dims(ts.FieldType(dims=[I, J], dtype=ts.ScalarType(kind=ts.ScalarKind.INT64)))
    [Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>), Dimension(value='J', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)]
    """
    match symbol_type:
        case ts.ScalarType():
            return []
        case ts.FieldType(dims):
            return dims
    raise GTTypeError(f"Can not extract dimensions from {symbol_type}!")


def is_concretizable(symbol_type: ts.TypeSpec, to_type: ts.TypeSpec) -> bool:
    """
    Check if ``symbol_type`` can be concretized to ``to_type``.

    Examples:
    ---------
    >>> is_concretizable(
    ...     ts.ScalarType(kind=ts.ScalarKind.INT64),
    ...     to_type=ts.ScalarType(kind=ts.ScalarKind.INT64)
    ... )
    True

    >>> is_concretizable(
    ...     ts.ScalarType(kind=ts.ScalarKind.INT64),
    ...     to_type=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    ... )
    False

    >>> is_concretizable(
    ...     ts.DeferredType(constraint=None),
    ...     to_type=ts.FieldType(dtype=ts.ScalarType(kind=ts.ScalarKind.BOOL), dims=[])
    ... )
    True

    >>> is_concretizable(
    ...     ts.DeferredType(constraint=ts.DataType),
    ...     to_type=ts.FieldType(dtype=ts.ScalarType(kind=ts.ScalarKind.BOOL), dims=[])
    ... )
    True

    >>> is_concretizable(
    ...     ts.DeferredType(constraint=ts.OffsetType),
    ...     to_type=ts.FieldType(dtype=ts.ScalarType(kind=ts.ScalarKind.BOOL), dims=[])
    ... )
    False

    >>> is_concretizable(
    ...     ts.DeferredType(constraint=ts.TypeSpec),
    ...     to_type=ts.DeferredType(constraint=ts.ScalarType)
    ... )
    True

    """
    if isinstance(symbol_type, ts.DeferredType) and (
        symbol_type.constraint is None or issubclass(type_class(to_type), symbol_type.constraint)
    ):
        return True
    elif is_concrete(symbol_type):
        return symbol_type == to_type
    return False


def promote(*types: ts.FieldType | ts.ScalarType) -> ts.FieldType | ts.ScalarType:
    """
    Promote a set of field or scalar types to a common type.

    The resulting type is defined on all dimensions of the arguments, respecting
    the individual order of the dimensions of each argument (see
    :func:`promote_dims` for more details).

    >>> dtype = ts.ScalarType(kind=ts.ScalarKind.INT64)
    >>> I, J, K = (Dimension(value=dim) for dim in ["I", "J", "K"])
    >>> promoted: ts.FieldType = promote(
    ...     ts.FieldType(dims=[I, J], dtype=dtype),
    ...     ts.FieldType(dims=[I, J, K], dtype=dtype),
    ...     dtype
    ... )
    >>> promoted.dims == [I, J, K] and promoted.dtype == dtype
    True

    >>> promote(
    ...     ts.FieldType(dims=[I, J], dtype=dtype),
    ...     ts.FieldType(dims=[K], dtype=dtype)
    ... ) # doctest: +ELLIPSIS
    Traceback (most recent call last):
     ...
    gt4py.next.common.GTTypeError: Dimensions can not be promoted. Could not determine order of the following dimensions: J, K.
    """
    if all(isinstance(type_, ts.ScalarType) for type_ in types):
        if not all(type_ == types[0] for type_ in types):
            raise GTTypeError("Could not promote scalars of different dtype (not implemented).")
        if not all(type_.shape is None for type_ in types):  # type: ignore[union-attr]
            raise NotImplementedError("Shape promotion not implemented.")
        return types[0]
    elif all(isinstance(type_, (ts.ScalarType, ts.FieldType)) for type_ in types):
        dims = promote_dims(*(extract_dims(type_) for type_ in types))
        dtype = cast(ts.ScalarType, promote(*(extract_dtype(type_) for type_ in types)))

        return ts.FieldType(dims=dims, dtype=dtype)
    raise TypeError("Expected a FieldType or ScalarType.")


def promote_dims(*dims_list: list[Dimension]) -> list[Dimension]:
    """
    Find a unique ordering of multiple (individually ordered) lists of dimensions.

    The resulting list of dimensions contains all dimensions of the arguments
    in the order they originally appear. If no unique order exists or a
    contradicting order is found an exception is raised.

    A modified version (ensuring uniqueness of the order) of
    `Kahn's algorithm <https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm>`_
    is used to topologically sort the arguments.

    >>> I, J, K = (Dimension(value=dim) for dim in ["I", "J", "K"])
    >>> promote_dims([I, J], [I, J, K]) == [I, J, K]
    True
    >>> promote_dims([I, J], [K]) # doctest: +ELLIPSIS
    Traceback (most recent call last):
     ...
    gt4py.next.common.GTTypeError: Dimensions can not be promoted. Could not determine order of the following dimensions: J, K.
    >>> promote_dims([I, J], [J, I]) # doctest: +ELLIPSIS
    Traceback (most recent call last):
     ...
    gt4py.next.common.GTTypeError: Dimensions can not be promoted. The following dimensions appear in contradicting order: I, J.
    """
    # build a graph with the vertices being dimensions and edges representing
    #  the order between two dimensions. The graph is encoded as a dictionary
    #  mapping dimensions to their predecessors, i.e. a dictionary containing
    #  adjacency lists. Since graphlib.TopologicalSorter uses predecessors
    #  (contrary to successors) we also use this directionality here.
    graph: dict[Dimension, set[Dimension]] = {}
    for dims in dims_list:
        if len(dims) == 0:
            continue
        # create a vertex for each dimension
        for dim in dims:
            graph.setdefault(dim, set())
        # add edges
        predecessor = dims[0]
        for dim in dims[1:]:
            graph[dim].add(predecessor)
            predecessor = dim

    # modified version of Kahn's algorithm
    topologically_sorted_list: list[Dimension] = []

    # compute in-degree for each vertex
    in_degree = {v: 0 for v in graph.keys()}
    for v1 in graph:
        for v2 in graph[v1]:
            in_degree[v2] += 1

    # process vertices with in-degree == 0
    # TODO(tehrengruber): avoid recomputation of zero_in_degree_vertex_list
    while zero_in_degree_vertex_list := [v for v, d in in_degree.items() if d == 0]:
        if len(zero_in_degree_vertex_list) != 1:
            raise GTTypeError(
                f"Dimensions can not be promoted. Could not determine "
                f"order of the following dimensions: "
                f"{', '.join((dim.value for dim in zero_in_degree_vertex_list))}."
            )
        v = zero_in_degree_vertex_list[0]
        del in_degree[v]
        topologically_sorted_list.insert(0, v)
        # update in-degree
        for predecessor in graph[v]:
            in_degree[predecessor] -= 1

    if len(in_degree.items()) > 0:
        raise GTTypeError(
            f"Dimensions can not be promoted. The following dimensions "
            f"appear in contradicting order: {', '.join((dim.value for dim in in_degree.keys()))}."
        )

    return topologically_sorted_list


@functools.singledispatch
def return_type(
    callable_type: ts.CallableType,
    *,
    with_args: list[ts.TypeSpec],
    with_kwargs: dict[str, ts.TypeSpec],
):
    raise NotImplementedError(
        f"Return type deduction of type " f"{type(callable_type).__name__} not implemented."
    )


@return_type.register
def return_type_func(
    func_type: ts.FunctionType,
    *,
    with_args: list[ts.TypeSpec],
    with_kwargs: dict[str, ts.TypeSpec],
):
    return func_type.returns


@return_type.register
def return_type_field(
    field_type: ts.FieldType,
    *,
    with_args: list[ts.TypeSpec],
    with_kwargs: dict[str, ts.TypeSpec],
):
    try:
        accepts_args(field_type, with_args=with_args, with_kwargs=with_kwargs, raise_exception=True)
    except GTTypeError as ex:
        raise GTTypeError("Could not deduce return type of invalid remap operation.") from ex

    if not isinstance(with_args[0], ts.OffsetType):
        raise GTTypeError(f"First argument must be of type {ts.OffsetType}, got {with_args[0]}.")

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


@functools.singledispatch
def function_signature_incompatibilities(
    func_type: ts.CallableType, args: list[ts.TypeSpec], kwargs: dict[str, ts.TypeSpec]
) -> Iterator[str]:
    """
    Return incompatibilities for a call to ``func_type`` with given arguments.

    Note that all types must be concrete/complete.
    """
    raise NotImplementedError(f"Not implemented for type {type(func_type).__name__}.")


@function_signature_incompatibilities.register
def function_signature_incompatibilities_func(
    func_type: ts.FunctionType, args: list[ts.TypeSpec], kwargs: dict[str, ts.TypeSpec]
) -> Iterator[str]:

    # check positional arguments
    if len(func_type.args) != len(args):
        yield f"Function takes {len(func_type.args)} argument(s), but {len(args)} were given."
    for i, (a_arg, b_arg) in enumerate(zip(func_type.args, args)):
        if a_arg != b_arg and not is_concretizable(a_arg, to_type=b_arg):
            yield f"Expected {i}-th argument to be of type {a_arg}, but got {b_arg}."

    # check for missing or extra keyword arguments
    kw_a_m_b = set(func_type.kwargs.keys()) - set(kwargs.keys())
    if len(kw_a_m_b) > 0:
        yield f"Missing required keyword argument(s) `{'`, `'.join(kw_a_m_b)}`."
    kw_b_m_a = set(kwargs.keys()) - set(func_type.kwargs.keys())
    if len(kw_b_m_a) > 0:
        yield f"Got unexpected keyword argument(s) `{'`, `'.join(kw_b_m_a)}`."

    for kwarg in set(func_type.kwargs.keys()) & set(kwargs.keys()):
        if (a_kwarg := func_type.kwargs[kwarg]) != (
            b_kwarg := kwargs[kwarg]
        ) and not is_concretizable(a_kwarg, to_type=b_kwarg):
            yield f"Expected keyword argument {kwarg} to be of type {func_type.kwargs[kwarg]}, but got {kwargs[kwarg]}."


@function_signature_incompatibilities.register
def function_signature_incompatibilities_field(
    field_type: ts.FieldType,
    args: list[ts.TypeSpec],
    kwargs: dict[str, ts.TypeSpec],
) -> Iterator[str]:
    if len(args) != 1:
        yield f"Function takes 1 argument(s), but {len(args)} were given."
        return

    if not isinstance(args[0], ts.OffsetType):
        yield f"Expected 0-th argument to be of type {ts.OffsetType}, but got {args[0]}."
        return

    if kwargs:
        yield f"Got unexpected keyword argument(s) `{'`, `'.join(kwargs.keys())}`."
        return

    source_dim = args[0].source
    target_dims = args[0].target
    # TODO: This code does not handle ellipses for dimensions. Fix it.
    assert field_type.dims is not ...
    if field_type.dims and source_dim not in field_type.dims:
        yield f"Incompatible offset can not shift field defined on " f"{', '.join([dim.value for dim in field_type.dims])} from " f"{source_dim.value} to target dim(s): " f"{', '.join([dim.value for dim in target_dims])}"


def accepts_args(
    callable_type: ts.CallableType,
    *,
    with_args: list[ts.TypeSpec],
    with_kwargs: dict[str, ts.TypeSpec],
    raise_exception: bool = False,
) -> bool:
    """
    Check if a function can be called for given arguments.

    If ``raise_exception`` is given a :class:`GTTypeError` is raised with a
    detailed description of why the function is not callable.

    Note that all types must be concrete/complete.

    Examples:
        >>> bool_type = ts.ScalarType(kind=ts.ScalarKind.BOOL)
        >>> func_type = ts.FunctionType(
        ...     args=[bool_type],
        ...     kwargs={"foo": bool_type},
        ...     returns=ts.VoidType()
        ... )
        >>> accepts_args(func_type, with_args=[bool_type], with_kwargs={"foo": bool_type})
        True
        >>> accepts_args(func_type, with_args=[], with_kwargs={})
        False
    """
    if not isinstance(callable_type, ts.CallableType):
        if raise_exception:
            raise GTTypeError(f"Expected a callable type, but got `{callable_type}`.")
        return False

    errors = function_signature_incompatibilities(callable_type, with_args, with_kwargs)
    if raise_exception:
        error_list = list(errors)
        if len(error_list) > 0:
            raise GTTypeError(
                f"Invalid call to function of type `{callable_type}`:\n"
                + ("\n".join([f"  - {error}" for error in error_list]))
            )
        return True

    return next(errors, None) is None
