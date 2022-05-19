from typing import Iterator, Type, TypeGuard

from functional.common import Dimension, GTTypeError
from functional.ffront import common_types as ct


def is_concrete(symbol_type: ct.SymbolType) -> TypeGuard[ct.SymbolType]:
    """Figure out if the foast type is completely deduced."""
    match symbol_type:
        case ct.DeferredSymbolType():
            return False
        case ct.SymbolType():
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
            return constraint
        case ct.SymbolType() as concrete_type:
            return concrete_type.__class__
    raise GTTypeError(
        f"Invalid type for TypeInfo: requires {ct.SymbolType}, got {type(symbol_type)}!"
    )


def extract_dtype(symbol_type: ct.SymbolType) -> ct.ScalarType:
    """
    Extract the data type from ``symbol_type`` if it is one of FieldType or ScalarType.

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


def is_arithmetic(symbol_type: ct.SymbolType) -> bool:
    """
    Check if ``symbol_type`` is compatible with arithmetic operations.

    Examples:
    ---------
    >>> is_arithmetic(ct.ScalarType(kind=ct.ScalarKind.FLOAT64))
    True
    >>> is_arithmetic(ct.ScalarType(kind=ct.ScalarKind.BOOL))
    False
    >>> is_arithmetic(ct.FieldType(dims=[], dtype=ct.ScalarType(kind=ct.ScalarKind.INT32)))
    True
    """
    if extract_dtype(symbol_type).kind in [
        ct.ScalarKind.INT32,
        ct.ScalarKind.INT64,
        ct.ScalarKind.FLOAT32,
        ct.ScalarKind.FLOAT64,
    ]:
        return True
    return False


def is_logical(symbol_type: ct.SymbolType) -> bool:
    return extract_dtype(symbol_type).kind is ct.ScalarKind.BOOL


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
    [Dimension(value='I', local=False), Dimension(value='J', local=False)]
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


def _is_sublist(small_list, big_list):
    if len(small_list) > len(big_list):
        return False
    elif all(i in big_list for i in small_list):
        start = big_list.index(small_list[0])
        end = big_list.index(small_list[-1]) + 1
        return small_list == big_list[start:end]
    return False


def is_dimensionally_promotable(symbol_type: ct.SymbolType, to_type: ct.SymbolType) -> bool:
    """
    Check if `symbol_type` has no fixed dimensionality and can be dimensionally promoted.

    This is not to be mistaken for broadcasting, a more general concept, which allows interactions
    between differing sets of dimensions.
    """
    # scalars can always be broadcasted
    if type_class(symbol_type) is ct.ScalarType:
        return True
    # symbol_type must be either zero- or any dimensional or have same dimensionality as to_type
    elif type_class(to_type) is ct.FieldType:
        dims = extract_dims(symbol_type)
        return dims in [[], Ellipsis] or _is_sublist(dims, extract_dims(to_type))
    return False


def function_signature_incompatibilities(
    func_type: ct.FunctionType, args: list[ct.SymbolType], kwargs: dict[str, ct.SymbolType]
) -> Iterator[str]:
    """
    Return incompatibilities for a call to ``func_type`` with given arguments.

    Note that all types must be concrete/complete.
    """
    # check positional arguments
    if len(func_type.args) != len(args):
        yield f"Function takes {len(func_type.args)} arguments, but {len(args)} were given."
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


def is_callable(
    function_type: ct.FunctionType,
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
        >>> is_callable(func_type, with_args=[bool_type], with_kwargs={"foo": bool_type})
        True
        >>> is_callable(func_type, with_args=[], with_kwargs={})
        False
    """
    if not isinstance(function_type, ct.FunctionType):
        if raise_exception:
            raise GTTypeError(f"Expected a function type, but got `{function_type}`.")
        return False

    errors = function_signature_incompatibilities(function_type, with_args, with_kwargs)
    if raise_exception:
        error_list = list(errors)
        if len(error_list) > 0:
            raise GTTypeError(
                f"Invalid call to function of type `{function_type}`:\n"
                + ("\n".join([f"  - {error}" for error in error_list]))
            )
        return True

    return next(errors, None) is None
