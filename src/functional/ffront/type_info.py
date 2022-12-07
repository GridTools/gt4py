import functools
from functools import reduce
from types import EllipsisType
from typing import Callable, Iterator, Type, TypeGuard, cast

from eve.utils import XIterable, xiter
from functional.common import Dimension, GTTypeError
from functional.ffront import common_types as ct


def is_concrete(symbol_type: ct.SymbolType) -> TypeGuard[ct.SymbolType]:
    """Figure out if the foast type is completely deduced."""
    if isinstance(symbol_type, ct.DeferredSymbolType):
        return False
    elif isinstance(symbol_type, ct.SymbolType):
        return True
    return False


def type_class(symbol_type: ct.SymbolType) -> Type[ct.SymbolType]:
    """
    Determine which class should be used to create a compatible concrete type.

    Examples:
    ---------
    >>> type_class(ct.DeferredSymbolType(constraint=ct.ScalarType)).__name__
    'ScalarType'

    >>> type_class(ct.FieldType(dims=[], dtype=ct.ScalarType(kind=ct.ScalarKind.BOOL))).__name__
    'FieldType'

    >>> type_class(ct.TupleType(types=[])).__name__
    'TupleType'
    """
    match symbol_type:
        case ct.DeferredSymbolType(constraint):
            if constraint is None:
                raise GTTypeError(f"No type information available for {symbol_type}!")
            elif isinstance(constraint, tuple):
                raise GTTypeError(f"Not sufficient type information available for {symbol_type}!")
            return constraint
        case ct.SymbolType() as concrete_type:
            return concrete_type.__class__
    raise GTTypeError(
        f"Invalid type for TypeInfo: requires {ct.SymbolType}, got {type(symbol_type)}!"
    )


def primitive_constituents(
    symbol_type: ct.SymbolType,
) -> XIterable[ct.SymbolType]:
    """
    Return the primitive types contained in a composite type.

    >>> from functional.common import Dimension
    >>> I = Dimension(value="I")
    >>> int_type = ct.ScalarType(kind=ct.ScalarKind.INT)
    >>> field_type = ct.FieldType(dims=[I], dtype=int_type)

    >>> tuple_type = ct.TupleType(types=[int_type, field_type])
    >>> primitive_constituents(tuple_type).to_list()  # doctest: +ELLIPSIS
    [ScalarType(...), FieldType(...)]

    >>> nested_tuple = ct.TupleType(types=[field_type, tuple_type])
    >>> primitive_constituents(nested_tuple).to_list()  # doctest: +ELLIPSIS
    [FieldType(...), ScalarType(...), FieldType(...)]
    """

    def constituents_yielder(symbol_type: ct.SymbolType):
        if isinstance(symbol_type, ct.TupleType):
            for el_type in symbol_type.types:
                yield from constituents_yielder(el_type)
        else:
            yield symbol_type

    return xiter(constituents_yielder(symbol_type))


def apply_to_primitive_constituents(
    symbol_type: ct.SymbolType,
    fun: Callable[[ct.SymbolType], ct.SymbolType]
    | Callable[[ct.SymbolType, tuple[int, ...]], ct.SymbolType],
    with_path_arg=False,
    _path=(),
):
    """
    Apply function to all primitive constituents of a type.

    >>> int_type = ct.ScalarType(kind=ct.ScalarKind.INT)
    >>> tuple_type = ct.TupleType(types=[int_type, int_type])
    >>> print(apply_to_primitive_constituents(tuple_type, lambda primitive_type: ct.FieldType(dims=[], dtype=primitive_type)))
    tuple[Field[[], int64], Field[[], int64]]
    """
    if isinstance(symbol_type, ct.TupleType):
        return ct.TupleType(
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


def extract_dtype(symbol_type: ct.SymbolType) -> ct.ScalarType:
    """
    Extract the data type from ``symbol_type`` if it is either `FieldType` or `ScalarType`.

    Raise an error if no dtype can be found or the result would be ambiguous.

    Examples:
    ---------
    >>> print(extract_dtype(ct.ScalarType(kind=ct.ScalarKind.FLOAT64)))
    float64

    >>> print(extract_dtype(ct.FieldType(dims=[], dtype=ct.ScalarType(kind=ct.ScalarKind.BOOL))))
    bool
    """
    match symbol_type:
        case ct.FieldType(dtype=dtype):
            return dtype
        case ct.ScalarType() as dtype:
            return dtype
    raise GTTypeError(f"Can not unambiguosly extract data type from {symbol_type}!")


def is_floating_point(symbol_type: ct.SymbolType) -> bool:
    """
    Check if the dtype of ``symbol_type`` is a floating point type.

    Examples:
    ---------
    >>> is_floating_point(ct.ScalarType(kind=ct.ScalarKind.FLOAT64))
    True
    >>> is_floating_point(ct.ScalarType(kind=ct.ScalarKind.FLOAT32))
    True
    >>> is_floating_point(ct.ScalarType(kind=ct.ScalarKind.INT32))
    False
    >>> is_floating_point(ct.FieldType(dims=[Dimension(value="I")], dtype=ct.ScalarType(kind=ct.ScalarKind.FLOAT64)))
    True
    """
    return extract_dtype(symbol_type).kind in [
        ct.ScalarKind.FLOAT32,
        ct.ScalarKind.FLOAT64,
    ]


def is_integral(symbol_type: ct.SymbolType) -> bool:
    """
    Check if the dtype of ``symbol_type`` is an integral type.

    Examples:
    ---------
    >>> is_integral(ct.ScalarType(kind=ct.ScalarKind.INT))
    True
    >>> is_integral(ct.ScalarType(kind=ct.ScalarKind.INT32))
    True
    >>> is_integral(ct.ScalarType(kind=ct.ScalarKind.FLOAT32))
    False
    >>> is_integral(ct.FieldType(dims=[Dimension(value="I")], dtype=ct.ScalarType(kind=ct.ScalarKind.INT)))
    True
    """
    return extract_dtype(symbol_type).kind in [
        ct.ScalarKind.INT,
        ct.ScalarKind.INT32,
        ct.ScalarKind.INT64,
    ]


def is_number(symbol_type: ct.SymbolType) -> bool:
    """
    Check if ``symbol_type`` is either intergral or float.

    Examples:
    ---------
    >>> is_number(ct.ScalarType(kind=ct.ScalarKind.FLOAT64))
    True
    >>> is_number(ct.ScalarType(kind=ct.ScalarKind.INT32))
    True
    >>> is_number(ct.ScalarType(kind=ct.ScalarKind.BOOL))
    False
    >>> is_number(ct.FieldType(dims=[], dtype=ct.ScalarType(kind=ct.ScalarKind.INT)))
    False
    """
    if not isinstance(symbol_type, ct.ScalarType):
        return False
    # TODO(nfarabullini): re-factor is_arithmetic such that it only checks for scalars
    #  and the emtpy field pass in an another function
    return is_arithmetic(symbol_type)


def is_logical(symbol_type: ct.SymbolType) -> bool:
    return extract_dtype(symbol_type).kind is ct.ScalarKind.BOOL


def is_arithmetic(symbol_type: ct.SymbolType) -> bool:
    """
    Check if ``symbol_type`` is compatible with arithmetic operations.

    Examples:
    ---------
    >>> is_arithmetic(ct.ScalarType(kind=ct.ScalarKind.FLOAT64))
    True
    >>> is_arithmetic(ct.ScalarType(kind=ct.ScalarKind.BOOL))
    False
    >>> is_arithmetic(ct.ScalarType(kind=ct.ScalarKind.STRING))
    False
    >>> is_arithmetic(ct.FieldType(dims=[], dtype=ct.ScalarType(kind=ct.ScalarKind.INT32)))
    True
    """
    return is_floating_point(symbol_type) or is_integral(symbol_type)


def is_field_type_or_tuple_of_field_type(type_: ct.SymbolType) -> bool:
    """
     Return True if ``type_`` is FieldType or FieldType nested in TupleType.

     Examples:
     ---------
    >>> scalar_type = ct.ScalarType(kind=ct.ScalarKind.INT)
    >>> field_type = ct.FieldType(dims=[], dtype=scalar_type)
     >>> is_field_type_or_tuple_of_field_type(field_type)
     True
     >>> is_field_type_or_tuple_of_field_type(ct.TupleType(types=[field_type, field_type]))
     True
     >>> is_field_type_or_tuple_of_field_type(ct.TupleType(types=[field_type, scalar_type]))
     False
    """
    return all(isinstance(t, ct.FieldType) for t in primitive_constituents(type_))


def extract_dims(symbol_type: ct.SymbolType) -> list[Dimension]:
    """
    Try to extract field dimensions if possible.

    Scalars are treated as zero-dimensional

    Examples:
    ---------
    >>> extract_dims(ct.ScalarType(kind=ct.ScalarKind.INT64, shape=[3, 4]))
    []
    >>> I = Dimension(value="I")
    >>> J = Dimension(value="J")
    >>> extract_dims(ct.FieldType(dims=[I, J], dtype=ct.ScalarType(kind=ct.ScalarKind.INT64)))
    [Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>), Dimension(value='J', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)]
    """
    match symbol_type:
        case ct.ScalarType():
            return []
        case ct.FieldType(dims):
            return dims
    raise GTTypeError(f"Can not extract dimensions from {symbol_type}!")


def is_concretizable(symbol_type: ct.SymbolType, to_type: ct.SymbolType) -> bool:
    """
    Check if ``symbol_type`` can be concretized to ``to_type``.

    Examples:
    ---------
    >>> is_concretizable(
    ...     ct.ScalarType(kind=ct.ScalarKind.INT64),
    ...     to_type=ct.ScalarType(kind=ct.ScalarKind.INT64)
    ... )
    True

    >>> is_concretizable(
    ...     ct.ScalarType(kind=ct.ScalarKind.INT64),
    ...     to_type=ct.ScalarType(kind=ct.ScalarKind.FLOAT64)
    ... )
    False

    >>> is_concretizable(
    ...     ct.DeferredSymbolType(constraint=None),
    ...     to_type=ct.FieldType(dtype=ct.ScalarType(kind=ct.ScalarKind.BOOL), dims=[])
    ... )
    True

    >>> is_concretizable(
    ...     ct.DeferredSymbolType(constraint=ct.DataType),
    ...     to_type=ct.FieldType(dtype=ct.ScalarType(kind=ct.ScalarKind.BOOL), dims=[])
    ... )
    True

    >>> is_concretizable(
    ...     ct.DeferredSymbolType(constraint=ct.OffsetType),
    ...     to_type=ct.FieldType(dtype=ct.ScalarType(kind=ct.ScalarKind.BOOL), dims=[])
    ... )
    False

    >>> is_concretizable(
    ...     ct.DeferredSymbolType(constraint=ct.SymbolType),
    ...     to_type=ct.DeferredSymbolType(constraint=ct.ScalarType)
    ... )
    True

    """
    if isinstance(symbol_type, ct.DeferredSymbolType) and (
        symbol_type.constraint is None or issubclass(type_class(to_type), symbol_type.constraint)
    ):
        return True
    elif is_concrete(symbol_type):
        return symbol_type == to_type
    return False


def _is_zero_dim_field(field: ct.SymbolType) -> bool:
    return isinstance(field, ct.FieldType) and field.dims != Ellipsis and len(field.dims) == 0


def promote_zero_dims(
    args: list[ct.SymbolType], function_type: ct.FieldOperatorType | ct.ProgramType
) -> list:
    """Promote arg types to zero dimensional fields if compatible and required by function signature."""
    new_args = args.copy()
    for arg_i, arg in enumerate(args):
        def_type = function_type.definition.args[arg_i]
        if _is_zero_dim_field(def_type) and is_number(arg):
            if extract_dtype(def_type) == extract_dtype(arg):
                new_args[arg_i] = def_type
            else:
                raise GTTypeError(f"{arg} is not compatible with {def_type}.")
    return new_args


def promote(*types: ct.FieldType | ct.ScalarType) -> ct.FieldType | ct.ScalarType:
    """
    Promote a set of field or scalar types to a common type.

    The resulting type is defined on all dimensions of the arguments, respecting
    the individual order of the dimensions of each argument (see
    :func:`promote_dims` for more details).

    >>> dtype = ct.ScalarType(kind=ct.ScalarKind.INT64)
    >>> I, J, K = (Dimension(value=dim) for dim in ["I", "J", "K"])
    >>> promoted: ct.FieldType = promote(
    ...     ct.FieldType(dims=[I, J], dtype=dtype),
    ...     ct.FieldType(dims=[I, J, K], dtype=dtype),
    ...     dtype
    ... )
    >>> promoted.dims == [I, J, K] and promoted.dtype == dtype
    True

    >>> promote(
    ...     ct.FieldType(dims=[I, J], dtype=dtype),
    ...     ct.FieldType(dims=[K], dtype=dtype)
    ... ) # doctest: +ELLIPSIS
    Traceback (most recent call last):
     ...
    functional.common.GTTypeError: Dimensions can not be promoted. Could not determine order of the following dimensions: J, K.
    """
    if all(isinstance(type_, ct.ScalarType) for type_ in types):
        if not all(type_ == types[0] for type_ in types):
            raise GTTypeError("Could not promote scalars of different dtype (not implemented).")
        if not all(type_.shape is None for type_ in types):  # type: ignore[union-attr]
            raise NotImplementedError("Shape promotion not implemented.")
        return types[0]
    elif all(isinstance(type_, (ct.ScalarType, ct.FieldType)) for type_ in types):
        dims = promote_dims(*(extract_dims(type_) for type_ in types))
        dtype = cast(ct.ScalarType, promote(*(extract_dtype(type_) for type_ in types)))

        return ct.FieldType(dims=dims, dtype=dtype)
    raise TypeError("Expected a FieldType or ScalarType.")


def promote_dims(
    *dims_list: list[Dimension] | EllipsisType,
) -> list[Dimension] | EllipsisType:
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
    functional.common.GTTypeError: Dimensions can not be promoted. Could not determine order of the following dimensions: J, K.
    >>> promote_dims([I, J], [J, I]) # doctest: +ELLIPSIS
    Traceback (most recent call last):
     ...
    functional.common.GTTypeError: Dimensions can not be promoted. The following dimensions appear in contradicting order: I, J.
    """
    # build a graph with the vertices being dimensions and edges representing
    #  the order between two dimensions. The graph is encoded as a dictionary
    #  mapping dimensions to their predecessors, i.e. a dictionary containing
    #  adjacency lists. Since graphlib.TopologicalSorter uses predecessors
    #  (contrary to successors) we also use this directionality here.
    graph: dict[Dimension, set[Dimension]] = {}
    for dims in dims_list:
        if dims == Ellipsis:
            return Ellipsis
        dims = cast(list[Dimension], dims)
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
    callable_type: ct.CallableType,
    *,
    with_args: list[ct.SymbolType],
    with_kwargs: dict[str, ct.SymbolType],
):
    raise NotImplementedError(
        f"Return type deduction of type " f"{type(callable_type).__name__} not implemented."
    )


@return_type.register
def return_type_func(
    func_type: ct.FunctionType,
    *,
    with_args: list[ct.SymbolType],
    with_kwargs: dict[str, ct.SymbolType],
):
    return func_type.returns


@return_type.register
def return_type_fieldop(
    fieldop_type: ct.FieldOperatorType,
    *,
    with_args: list[ct.SymbolType],
    with_kwargs: dict[str, ct.SymbolType],
):
    ret_type = return_type(fieldop_type.definition, with_args=with_args, with_kwargs=with_kwargs)
    return ret_type


@return_type.register
def return_type_scanop(
    callable_type: ct.ScanOperatorType,
    *,
    with_args: list[ct.SymbolType],
    with_kwargs: dict[str, ct.SymbolType],
):
    carry_dtype = callable_type.definition.returns
    promoted_dims = promote_dims(
        *(extract_dims(el) for arg in with_args for el in primitive_constituents(arg)),
        # the vertical axis is always added to the dimension of the returned
        #  field
        [callable_type.axis],
    )
    return apply_to_primitive_constituents(
        carry_dtype, lambda arg: ct.FieldType(dims=promoted_dims, dtype=cast(ct.ScalarType, arg))
    )


@return_type.register
def return_type_field(
    field_type: ct.FieldType,
    *,
    with_args: list[ct.SymbolType],
    with_kwargs: dict[str, ct.SymbolType],
):
    try:
        accepts_args(field_type, with_args=with_args, with_kwargs=with_kwargs, raise_exception=True)
    except GTTypeError as ex:
        raise GTTypeError("Could not deduce return type of invalid remap operation.") from ex

    if not isinstance(with_args[0], ct.OffsetType):
        raise GTTypeError(f"First argument must be of type {ct.OffsetType}, got {with_args[0]}.")

    source_dim = with_args[0].source
    target_dims = with_args[0].target
    new_dims = []
    for d in field_type.dims:
        if d != source_dim:
            new_dims.append(d)
        else:
            new_dims.extend(target_dims)
    return ct.FieldType(dims=new_dims, dtype=field_type.dtype)


@functools.singledispatch
def function_signature_incompatibilities(
    func_type: ct.CallableType, args: list[ct.SymbolType], kwargs: dict[str, ct.SymbolType]
) -> Iterator[str]:
    """
    Return incompatibilities for a call to ``func_type`` with given arguments.

    Note that all types must be concrete/complete.
    """
    raise NotImplementedError(f"Not implemented for type {type(func_type).__name__}.")


@function_signature_incompatibilities.register
def function_signature_incompatibilities_func(
    func_type: ct.FunctionType,
    args: list[ct.SymbolType],
    kwargs: dict[str, ct.SymbolType],
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
def function_signature_incompatibilities_fieldop(
    fieldop_type: ct.FieldOperatorType, args: list[ct.SymbolType], kwargs: dict[str, ct.SymbolType]
) -> Iterator[str]:
    new_args = promote_zero_dims(args, fieldop_type)
    yield from function_signature_incompatibilities_func(fieldop_type.definition, new_args, kwargs)


@function_signature_incompatibilities.register
def function_signature_incompatibilities_scanop(
    scanop_type: ct.ScanOperatorType, args: list[ct.SymbolType], kwargs: dict[str, ct.SymbolType]
) -> Iterator[str]:
    if not all(
        is_field_type_or_tuple_of_field_type(arg) or isinstance(arg, ct.ScalarType) for arg in args
    ):
        yield "Arguments to scan operator must be fields, scalars or tuples thereof."
        return

    new_args = [
        ct.FieldType(dims=[], dtype=extract_dtype(arg)) if is_number(arg) else arg for arg in args
    ]

    arg_dims = [extract_dims(el) for arg in new_args for el in primitive_constituents(arg)]
    try:
        promote_dims(*arg_dims)
    except GTTypeError as e:
        yield e.args[0]

    if len(new_args) != len(scanop_type.definition.args) - 1:
        yield f"Scan operator takes {len(scanop_type.definition.args)-1} arguments, but {len(new_args)} were given."
        return

    promoted_args = []
    for i, scan_pass_arg in enumerate(scanop_type.definition.args[1:]):
        # Helper function that given a scalar type in the signature of the scan
        # pass return a field type with that dtype and the dimensions of the
        # corresponding field type in the requested `args` type. Defined here
        # as we capture `i`.
        def _as_field(dtype: ct.ScalarType, path: tuple[int, ...]) -> ct.FieldType:
            try:
                el_type = reduce(lambda type_, idx: type_.types[idx], path, new_args[i])  # type: ignore[attr-defined] # noqa: B023
                return ct.FieldType(dims=extract_dims(el_type), dtype=dtype)
            except (IndexError, AttributeError):
                # The structure of the scan passes argument and the requested
                # argument type differ. As such we can not extract the dimensions
                # and just return a generic field shown in the error later on.
                return ct.FieldType(dims=..., dtype=dtype)

        promoted_args.append(
            apply_to_primitive_constituents(scan_pass_arg, _as_field, with_path_arg=True)  # type: ignore[arg-type]
        )

    # build a function type to leverage the already existing signature checking
    #  capabilities
    function_type = ct.FunctionType(
        args=promoted_args,
        kwargs={},
        returns=ct.DeferredSymbolType(constraint=None),
    )

    yield from function_signature_incompatibilities(function_type, new_args, kwargs)


@function_signature_incompatibilities.register
def function_signature_incompatibilities_program(
    program_type: ct.ProgramType, args: list[ct.SymbolType], kwargs: dict[str, ct.SymbolType]
) -> Iterator[str]:
    new_args = promote_zero_dims(args, program_type)
    yield from function_signature_incompatibilities_func(program_type.definition, new_args, kwargs)


@function_signature_incompatibilities.register
def function_signature_incompatibilities_field(
    field_type: ct.FieldType,
    args: list[ct.SymbolType],
    kwargs: dict[str, ct.SymbolType],
) -> Iterator[str]:
    if len(args) != 1:
        yield f"Function takes 1 argument(s), but {len(args)} were given."
        return

    if not isinstance(args[0], ct.OffsetType):
        yield f"Expected 0-th argument to be of type {ct.OffsetType}, but got {args[0]}."
        return

    if kwargs:
        yield f"Got unexpected keyword argument(s) `{'`, `'.join(kwargs.keys())}`."
        return

    source_dim = args[0].source
    target_dims = args[0].target
    if field_type.dims and source_dim not in field_type.dims:
        yield f"Incompatible offset can not shift field defined on " f"{', '.join([dim.value for dim in field_type.dims])} from " f"{source_dim.value} to target dim(s): " f"{', '.join([dim.value for dim in target_dims])}"


def accepts_args(
    callable_type: ct.CallableType,
    *,
    with_args: list[ct.SymbolType],
    with_kwargs: dict[str, ct.SymbolType],
    raise_exception: bool = False,
) -> bool:
    """
    Check if a function can be called for given arguments.

    If ``raise_exception`` is given a :class:`GTTypeError` is raised with a
    detailed description of why the function is not callable.

    Note that all types must be concrete/complete.

    Examples:
        >>> bool_type = ct.ScalarType(kind=ct.ScalarKind.BOOL)
        >>> func_type = ct.FunctionType(
        ...     args=[bool_type],
        ...     kwargs={"foo": bool_type},
        ...     returns=ct.VoidType()
        ... )
        >>> accepts_args(func_type, with_args=[bool_type], with_kwargs={"foo": bool_type})
        True
        >>> accepts_args(func_type, with_args=[], with_kwargs={})
        False
    """
    if not isinstance(callable_type, ct.CallableType):
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
