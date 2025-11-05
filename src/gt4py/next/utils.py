# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import inspect
import itertools
from collections.abc import Mapping
from typing import (
    Any,
    Callable,
    ClassVar,
    NamedTuple,
    Optional,
    ParamSpec,
    Protocol,
    Sequence,
    TypeGuard,
    TypeVar,
    cast,
    overload,
)


class RecursionGuard:
    """
    Context manager to guard against inifinite recursion.

    >>> def foo(i):
    ...     with RecursionGuard(i):
    ...         if i % 2 == 0:
    ...             foo(i)
    ...     return i
    >>> foo(3)
    3
    >>> foo(2)  # doctest:+ELLIPSIS
    Traceback (most recent call last):
        ...
    gt4py.next.utils.RecursionGuard.RecursionDetected
    """

    guarded_objects: ClassVar[set[int]] = set()

    obj: Any

    class RecursionDetected(Exception):
        pass

    def __init__(self, obj: Any):
        self.obj = obj

    def __enter__(self) -> None:
        if id(self.obj) in self.guarded_objects:
            raise self.RecursionDetected()
        self.guarded_objects.add(id(self.obj))

    def __exit__(self, *exc: Any) -> None:
        self.guarded_objects.remove(id(self.obj))


_T = TypeVar("_T")
_P = ParamSpec("_P")
_R = TypeVar("_R")


def is_tuple_of(v: Any, t: type[_T]) -> TypeGuard[tuple[_T, ...]]:
    return isinstance(v, tuple) and all(isinstance(e, t) for e in v)


# TODO(havogt): remove flatten duplications in the whole codebase
def flatten_nested_tuple(
    value: tuple[
        _T | tuple, ...
    ],  # `_T` omitted on purpose as type of `value`, to properly deduce `_T` on the user-side
) -> tuple[_T, ...]:
    if isinstance(value, tuple):
        return sum((flatten_nested_tuple(v) for v in value), start=())  # type: ignore[arg-type] # cannot properly express nesting
    else:
        return (value,)


@overload
def tree_map(
    fun: Callable[_P, _R],
    *,
    collection_type: type | tuple[type, ...] = tuple,
    result_collection_constructor: Optional[Callable] = None,
    unpack: bool = False,
    with_path_arg: bool = False,
) -> Callable[..., _R | tuple[_R | tuple, ...]]: ...


@overload
def tree_map(
    *,
    collection_type: type | tuple[type, ...] = tuple,
    result_collection_constructor: Optional[Callable] = None,
    unpack: bool = False,
    with_path_arg: bool = False,
) -> Callable[
    [Callable[_P, _R]], Callable[..., Any]
]: ...  # TODO(havogt): typing of `result_collection_constructor` is too weak here


def tree_map(
    fun: Optional[Callable[_P, _R]] = None,
    *,
    collection_type: type | tuple[type, ...] = tuple,
    result_collection_constructor: Optional[Callable] = None,
    unpack: bool = False,
    with_path_arg: bool = False,
) -> Callable[..., _R | tuple[_R | tuple, ...]] | Callable[[Callable[_P, _R]], Callable[..., Any]]:
    """
    Apply `fun` to each entry of (possibly nested) collections (by default `tuple`s).

    Args:
        fun: Function to apply to each entry of the collection.
        collection_type: Type of the collection to be traversed. Can be a single type or a tuple of types.
        result_collection_constructor: Type of the collection to be returned. If `None` the same type as `collection_type` is used.
        unpack: Replicate tuple structure returned from `fun` to the mapped result, i.e. return
          tuple of result collections instead of result collections of tuples.
        with_path_arg: Pass the path to access the current element to `fun`.
    Examples:
        >>> tree_map(lambda x: x + 1)(((1, 2), 3))
        ((2, 3), 4)

        >>> tree_map(lambda x, y: x + y)(((1, 2), 3), ((4, 5), 6))
        ((5, 7), 9)

        >>> tree_map(collection_type=list)(lambda x: x + 1)([[1, 2], 3])
        [[2, 3], 4]

        >>> tree_map(
        ...     collection_type=(list, tuple),
        ...     result_collection_constructor=lambda value, elts: tuple(elts)
        ...     if isinstance(value, list)
        ...     else list(elts),
        ... )(lambda x: x + 1)([(1, 2), 3])
        ([2, 3], 4)

        >>> @tree_map
        ... def impl(x):
        ...     return x + 1
        >>> impl(((1, 2), 3))
        ((2, 3), 4)

        >>> @tree_map(with_path_arg=True)
        ... def impl(x, path: tuple[int, ...]):
        ...     path_str = "".join(f"[{i}]" for i in path)
        ...     return f"t{path_str} = {x}"
        >>> t = impl(((1, 2), 3))
        >>> t[0][0]
        't[0][0] = 1'
        >>> t[0][1]
        't[0][1] = 2'
        >>> t[1]
        't[1] = 3'

        >>> @tree_map(unpack=True)
        ... def impl(x):
        ...     return (x, x**2)
        >>> identity, squared = impl(((2, 3), 4))
        >>> identity
        ((2, 3), 4)
        >>> squared
        ((4, 9), 16)
    """

    if result_collection_constructor is None:
        if isinstance(collection_type, tuple):
            # Note: that doesn't mean `collection_type=tuple`, but e.g. `collection_type=(list, tuple)`
            raise TypeError(
                "tree_map() requires `result_collection_constructor` when `collection_type` is a tuple of types."
            )
        result_collection_constructor = lambda _, elts: collection_type(elts)  # noqa: E731 # because a lambda is clearer

    if fun:

        @functools.wraps(fun)
        def impl(*args: Any | tuple[Any | tuple, ...]) -> _R | tuple[_R | tuple, ...]:
            if isinstance(args[0], collection_type):
                non_path_args: Sequence[Any]
                if with_path_arg:
                    *non_path_args, path = args
                    args = (*non_path_args, tuple((*path, i) for i in range(len(args[0]))))
                else:
                    non_path_args = args

                assert all(
                    isinstance(arg, collection_type) and len(args[0]) == len(arg)
                    for arg in non_path_args
                )
                assert result_collection_constructor is not None
                ctor = functools.partial(result_collection_constructor, args[0])

                mapped = [impl(*arg) for arg in zip(*args)]
                if unpack:
                    return tuple(map(ctor, zip(*mapped)))
                else:
                    return ctor(mapped)

            return fun(  # type: ignore[call-arg]
                *cast(_P.args, args),  # type: ignore[valid-type]
            )  # mypy doesn't understand that `args` at this point is of type `_P.args`

        if with_path_arg:
            return lambda *args: impl(*args, ())
        else:
            return impl
    else:
        return functools.partial(
            tree_map,
            collection_type=collection_type,
            result_collection_constructor=result_collection_constructor,
            unpack=unpack,
            with_path_arg=with_path_arg,
        )


_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


def equalize_tuple_structure(
    d1: _T1, d2: _T2, *, fill_value: Any, bidirectional: bool = True
) -> tuple[_T1, _T2]:
    """
    Given two values or tuples thereof equalize their structure.

    If one of the arguments is a tuple the argument will be promoted to a tuple of the same
    structure.

    >>> equalize_tuple_structure((1,), (2, 3), fill_value=None) == (
    ...     (1, None),
    ...     (2, 3),
    ... )
    True

    >>> equalize_tuple_structure(1, (2, 3), fill_value=None) == (
    ...     (1, 1),
    ...     (2, 3),
    ... )
    True

    If `bidirectional` is `False` only the second argument is equalized and an error is raised if
    the first argument would require promotion.
    """
    d1_is_tuple = isinstance(d1, tuple)
    d2_is_tuple = isinstance(d2, tuple)
    if not d1_is_tuple and d2_is_tuple:
        if not bidirectional:
            raise ValueError(f"Expected `{d2!s}` to have same structure as `{d1!s}`.")
        return equalize_tuple_structure(
            (d1,) * len(d2),  # type: ignore[arg-type] # ensured by condition above
            d2,
            fill_value=fill_value,
            bidirectional=bidirectional,
        )
    if d1_is_tuple and not d2_is_tuple:
        return equalize_tuple_structure(
            d1,
            (d2,) * len(d1),  # type: ignore[arg-type] # ensured by condition above
            fill_value=fill_value,
            bidirectional=bidirectional,
        )
    if d1_is_tuple and d2_is_tuple:
        if not bidirectional and len(d1) < len(d2):  # type: ignore[arg-type] # ensured by condition above
            raise ValueError(f"Expected `{d2!s}` to be longer than or equal to `{d1!s}`.")
        return tuple(  # type: ignore[return-value] # mypy not smart enough
            zip(
                *(
                    equalize_tuple_structure(
                        el1, el2, fill_value=fill_value, bidirectional=bidirectional
                    )
                    for el1, el2 in itertools.zip_longest(d1, d2, fillvalue=fill_value)  # type: ignore[call-overload] # d1, d2 are tuples
                )
            )
        )
    return d1, d2


class CanonicalizationOptions(NamedTuple):
    function_name: str
    allow_kwargs_mutation: bool
    sort_kwargs: bool


class CallArgsCanonicalizer(Protocol):
    def __call__(self, args: tuple, kwargs: dict[str, Any]) -> tuple[tuple, dict[str, Any]]: ...

    @property
    def options(self) -> CanonicalizationOptions: ...

    def cache_info(self) -> functools._CacheInfo: ...
    def cache_clear(self) -> None: ...


class CustomCallArgsCanonicalizerFactory(Protocol):
    def __call__(
        self, passed_pos_args_count: int, passed_kwargs_keys: tuple[str, ...]
    ) -> CallArgsCanonicalizer: ...


def make_args_canonicalizer_factory(
    signature: inspect.Signature,
    *,
    name: str = "unknown",
    allow_kwargs_mutation: bool = True,
    sort_kwargs: bool = False,
) -> CustomCallArgsCanonicalizerFactory:
    """
    Create a factory for functions that canonicalize call arguments for a given signature.

    For a full description of what canonicalization means, see `make_args_canonicalizer`.

    Returns:
        A factory that creates canonicalizers for a given number of positional arguments
        and a given set of keyword argument names.

    Note:
        `inspect.Signature.bind()` is not used here because it introduces too much overhead
        on the hot path. Instead, this function generates specialized canonicalizer functions
        for each combination of positional argument count and keyword argument names.
    """
    params: Mapping[str, inspect.Parameter] = signature.parameters
    if inspect.Parameter.VAR_POSITIONAL in (p.kind for p in params.values()):
        raise ValueError("Cannot create canonicalizer for functions with variadic parameters.")
    if inspect.Parameter.VAR_KEYWORD in (p.kind for p in params.values()):
        raise ValueError(
            "Cannot create canonicalizer for functions with variadic keyword parameters."
        )

    pos_name_to_index = {
        key: pos
        for pos, key in enumerate(
            key
            for key, param in params.items()
            if param.kind
            in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        )
    }
    pos_args_names = {*pos_name_to_index.keys()}
    pos_args_count = len(pos_name_to_index)
    kwonly_names = [
        key for key, param in params.items() if param.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    all_keywords = {
        key for key, param in params.items() if param.kind != inspect.Parameter.POSITIONAL_ONLY
    }
    all_args_count = len(params)

    if allow_kwargs_mutation and not sort_kwargs:
        # If we can mutate kwargs and don't need to sort them,
        # we can use the passed kwargs dict, which should be
        # in the correct order after popping the positional args.
        canonical_kwargs_expr = "kwargs"
    else:
        kwargs_items_exprs = [f"{name!r}: kwargs[{name!r}]" for name in kwonly_names]
        canonical_kwargs_expr = f"{{ {str.join(', ', kwargs_items_exprs)} }}"

    @functools.cache
    def canonicalizer_factory(
        passed_pos_args_count: int, passed_kwargs_keys: tuple[str, ...]
    ) -> CallArgsCanonicalizer:
        # This function and the generated canonicalizer are optimized for performance
        if passed_pos_args_count > pos_args_count:
            raise ValueError(
                f"Too many positional arguments (expected {pos_args_count}, got {passed_pos_args_count})."
            )
        passed_kwargs_set = {*passed_kwargs_keys}
        if unexpected_kwargs := (passed_kwargs_set - all_keywords):
            raise ValueError(f"Got unexpected keyword arguments: {unexpected_kwargs}.")
        if missing_kwargs := ({*kwonly_names} - passed_kwargs_set):
            raise ValueError(f"Missing keyword arguments: {missing_kwargs}.")
        if (total_arg_count := (passed_pos_args_count + len(passed_kwargs_keys))) > all_args_count:
            raise ValueError(
                f"Too many total arguments for the passed signature (expected {all_args_count}, got {total_arg_count})."
            )

        if pos_args_names & passed_kwargs_set:
            unpack_args_stmt = (
                (f"{str.join(', ', (f'a{i}' for i in range(passed_pos_args_count)))}, = args")
                if passed_pos_args_count
                else "# No args to unpack"
            )

            passed_args_indices = iter(range(passed_pos_args_count))
            if allow_kwargs_mutation:
                canonical_args_comprehension = (
                    f"kwargs.pop({key!r})"
                    if key in passed_kwargs_keys
                    else f"a{next(passed_args_indices)}"
                    for key in pos_name_to_index
                )
            else:
                canonical_args_comprehension = (
                    f"kwargs[{key!r}]"
                    if key in passed_kwargs_keys
                    else f"a{next(passed_args_indices)}"
                    for key in pos_name_to_index
                )
            canonical_args_expr = f"{str.join(', ', canonical_args_comprehension)},"

        else:
            unpack_args_stmt = "# No unpacking needed"
            canonical_args_expr = "args"
            if canonical_kwargs_expr == "kwargs":
                # Args are already canonical, return identity
                return cast(CallArgsCanonicalizer, lambda args, kwargs: (args, kwargs))

        shortened_passed_kwargs_keys = str.join(
            "_", (f"{key[:7]}{len(key)}" if len(key) > 8 else key for key in passed_kwargs_keys)
        )
        canonicalizer_func_name = (
            f"canonicalizer_for_{name}_{passed_pos_args_count}__{shortened_passed_kwargs_keys}"
        )
        canonicalizer_src = f"""
from __future__ import annotations

def {canonicalizer_func_name}(
    args: tuple, kwargs: dict[str, Any]
) -> tuple[tuple, dict[str, Any]]:
    try:
        {unpack_args_stmt}
        canonical_args = {canonical_args_expr}
        canonical_kwargs = {canonical_kwargs_expr}
        return canonical_args, canonical_kwargs
    except ValueError as error:
        raise ValueError("Error in arguments canonicalization.") from error
"""
        ns: dict[str, Any] = {}
        exec(canonicalizer_src, ns)
        return cast(CallArgsCanonicalizer, ns[canonicalizer_func_name])

    return canonicalizer_factory


def make_args_canonicalizer(
    signature: inspect.Signature,
    *,
    name: str = "unknown",
    allow_kwargs_mutation: bool = True,
    sort_kwargs: bool = False,
) -> CallArgsCanonicalizer:
    """
    Create a call arguments canonicalizer function from a given signature.

    The canonicalization means that the returned arguments are as all positional
    arguments were passed positionally, and only keyword-only arguments appear
    in the dictionary.

    Args:
        signature: The signature for which to create the canonicalizer.

    Keyword Args:
        name: Name of the function for which the canonicalizer is created.
        allow_kwargs_mutation: If `True`, the `kwargs` dictionary passed to the canonicalizer
            may be mutated. If `False`, the passed `kwargs` dictionary is copied first, which
            may introduce extra overhead in the canonicalizer.
        sort_kwargs: Select if the canonicalizer functions should order the keys of the output
            `kwargs` dictionary by the order of set in the function signature.

    Note:
        This function does not support variadic parameters (i.e., `*args` or `**kwargs`)
        nor parameters with default values in the signature. The implementation uses
        `make_args_canonicalizer_factory` internally.
    """
    canonicalizer_factory: CustomCallArgsCanonicalizerFactory = make_args_canonicalizer_factory(
        signature,
        name=name,
        allow_kwargs_mutation=allow_kwargs_mutation,
        sort_kwargs=sort_kwargs,
    )

    def canonicalizer(args: tuple, kwargs: dict[str, Any]) -> tuple[tuple, dict[str, Any]]:
        return canonicalizer_factory(
            passed_pos_args_count=len(args), passed_kwargs_keys=tuple(sorted(kwargs.keys()))
        )(args, kwargs)

    canonicalizer.options = CanonicalizationOptions(  # type: ignore[attr-defined] # adding new attribute
        function_name=name,
        allow_kwargs_mutation=allow_kwargs_mutation,
        sort_kwargs=sort_kwargs,
    )

    # canonicalizer_factory() is a conventional functools.cache instance, but it is never
    # exposed to the user. Here we expose its cache-related methods on the canonicalizer.
    canonicalizer.cache_info = canonicalizer_factory.cache_info  # type: ignore[attr-defined] # adding new attribute
    canonicalizer.cache_clear = canonicalizer_factory.cache_clear  # type: ignore[attr-defined] # adding new attribute

    return cast(CallArgsCanonicalizer, canonicalizer)
