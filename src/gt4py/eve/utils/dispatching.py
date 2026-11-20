# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Function dispatching utilities."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, cast

from .. import xtyping


T = TypeVar("T")
P = ParamSpec("P")


def singledispatcher(
    default: Callable[P, T] | None = None, *, implementations: dict[type, Callable[[Any], Any]]
) -> xtyping.SingleDispatchCallable[P, T]:
    """
    Create a single-dispatch callable from a default and a registry of implementations.

    This is a thin wrapper around :func:`functools.singledispatch` that allows
    constructing a dispatcher from an existing mapping rather than decorating
    individual functions.

    Args:
        default: The default implementation used when no type-specific handler
            is found. If ``None``, the implementation registered for ``object``
            in `implementations` is used as the default.
        implementations: A mapping from types to their registered handler
            functions. If `default` is ``None``, ``object`` must be present.

    Returns:
        A single-dispatch callable with the registered implementations.

    Examples:
        >>> def default_impl(x: Any) -> str:
        ...     return f"default: {x}"
        >>> def int_impl(x: int) -> str:
        ...     return f"int: {x}"
        >>> dispatch = singledispatcher(default_impl, implementations={int: int_impl})
        >>> dispatch(3.14)
        'default: 3.14'
        >>> dispatch(42)
        'int: 42'

    """
    if default is None:
        if object not in implementations:
            raise ValueError("A default implementation for 'object' must be provided.")
        default = cast(Callable[P, T], implementations[object])
    else:
        if not callable(default):
            raise ValueError(f"Default implementation must be callable, got '{default}'.")
        if object in implementations:
            raise ValueError(
                "Default implementation for 'object' is already provided in 'implementations'."
            )

    assert callable(default)  # for mypy

    if xtyping.is_single_dispatch_callable(default):
        # `default` is itself a single-dispatch callable (e.g. a dispatcher used
        # as the fallback to chain dispatchers). `functools.singledispatch`
        # copies the wrapped callable's ``__dict__`` -- which, for a dispatcher,
        # holds ``register``/``registry`` -- onto the new dispatcher via
        # ``update_wrapper``, aliasing the two. Registering the implementations
        # below would then silently mutate ``default`` itself. Forward through a
        # plain function (empty ``__dict__``) so the new dispatcher keeps its own
        # independent registry.
        inner_default = default

        def default(*args: Any, **kwargs: Any) -> T:  # deliberately shadowing the parameter
            return inner_default(*args, **kwargs)

    result = functools.singledispatch(default)
    for cls, func in implementations.items():
        result.register(cls)(func)
    return cast(xtyping.SingleDispatchCallable[P, T], result)


def merge_dispatchers(
    *dispatchers: xtyping.SingleDispatchCallable[P, T], default: Callable[P, T] | None = None
) -> xtyping.SingleDispatchCallable[P, T]:
    """
    Merge multiple single-dispatch callables into one.

    The resulting dispatcher will have the union of the registered
    implementations of the input dispatchers. If `default` is provided
    it will be used as the default implementation for the merged
    dispatcher, otherwise the default implementation of the last
    dispatcher will be used.
    """
    if not dispatchers:
        raise ValueError("At least one dispatcher must be provided.")

    merged_registry: dict[Any, Any] = {}
    for d in dispatchers:
        if not xtyping.is_single_dispatch_callable(d):
            raise TypeError(
                f"Expected only single-dispatch callables, got '{d}' of type '{type(d)}'"
            )
        merged_registry.update(d.registry)

    if default is not None:
        merged_registry.pop(object, None)  # remove default implementation from registry if present

    return singledispatcher(default, implementations=merged_registry)
