# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ast
import collections.abc
import contextlib
import dataclasses
import inspect
import textwrap
import types
import typing
import warnings
from collections.abc import Callable
from typing import Generic, ParamSpec, TypeVar


P = ParamSpec("P")
T = TypeVar("T")


def _get_unique_name(func: Callable) -> str:
    """Generate a unique name for a callable object."""
    return (
        f"{func.__module__}.{getattr(func, '__qualname__', func.__class__.__qualname__)}#{id(func)}"
    )


def _is_empty_function(func: Callable) -> bool:
    """Check if a callable object is empty (i.e., contains no statements)."""
    try:
        assert callable(func)
        callable_src = (
            inspect.getsource(func)
            if isinstance(func, types.FunctionType)
            else inspect.getsource(func.__call__)  # type: ignore[operator]  # asserted above
        )
        callable_ast = ast.parse(textwrap.dedent(callable_src))
        return all(
            isinstance(st, ast.Pass)
            or (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant))
            for st in typing.cast(ast.FunctionDef, callable_ast.body[0]).body
        )
    except Exception:
        return False


@dataclasses.dataclass(slots=True)
class _BaseHook(Generic[T, P]):
    """Base class to define callback registration functionality for all hook types."""

    definition: Callable[P, T]
    registry: dict[str, Callable[P, T]] = dataclasses.field(default_factory=dict, kw_only=True)
    callbacks: tuple[Callable[P, T], ...] = dataclasses.field(default=(), init=False)

    def __post_init__(self) -> None:
        # As an optimization to avoid an empty function call if no callbacks are
        # registered, we only add the original definitions to the list of callables
        # if it contains a non-empty definition.
        if not _is_empty_function(self.definition):
            self.callbacks = (self.definition,)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def register(
        self, callback: Callable[P, T], *, name: str | None = None, index: int | None = None
    ) -> None:
        """
        Register a callback to the hook.

        Args:
            callback: The callable to register.
            name: An optional name for the callback. If not provided, a unique name will be generated.
            index: An optional index at which to insert the callback (not counting the original
               definition). If not provided, the callback will be appended to the end of the list.
        """

        callable_signature = inspect.signature(callback)
        hook_signature = inspect.signature(self.definition)

        signature_mismatch = len(callable_signature.parameters) != len(
            hook_signature.parameters
        ) or any(
            # Remove the annotation before comparison to avoid false mismatches
            actual_param.replace(annotation="") != expected_param.replace(annotation="")
            for actual_param, expected_param in zip(
                callable_signature.parameters.values(), hook_signature.parameters.values()
            )
        )
        if signature_mismatch:
            raise ValueError(
                f"Callback signature {callable_signature} does not match hook signature {hook_signature}"
            )
        try:
            callable_typing = typing.get_type_hints(callback)
            hook_typing = typing.get_type_hints(self.definition)
            if not all(
                callable_typing[arg_key] == arg_typing
                for arg_key, arg_typing in hook_typing.items()
            ):
                warnings.warn(
                    f"Callback annotations {callable_typing} does not match expected hook annotations {hook_typing}",
                    stacklevel=2,
                )
        except Exception:
            pass
        name = name or _get_unique_name(callback)

        if index is None:
            self.callbacks += (callback,)
        else:
            if self.callbacks and self.callbacks[0] is self.definition:
                index += 1  # The original definition should always go first
            self.callbacks = (*self.callbacks[:index], callback, *self.callbacks[index:])

        self.registry[name] = callback

    def remove(self, callback: str | Callable[P, T]) -> None:
        """
        Remove a registered callback from the hook.

        Args:
            callback: The callable object to remove or its registered name.
        """
        if isinstance(callback, str):
            name = callback
            if name not in self.registry:
                raise KeyError(f"No callback registered under the name '{name}'")
        else:
            name = _get_unique_name(callback)
            if name not in self.registry:
                raise KeyError(f"Callback object {callback} not found in registry")

        callback = self.registry.pop(name)
        assert callback in self.callbacks
        self.callbacks = tuple(cb for cb in self.callbacks if cb is not callback)


@dataclasses.dataclass(slots=True)
class EventHook(_BaseHook[None, P]):
    """Event hook specification."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        for func in self.callbacks:
            func(*args, **kwargs)


@dataclasses.dataclass(slots=True)
class ContextHook(
    contextlib.AbstractContextManager, _BaseHook[contextlib.AbstractContextManager, P]
):
    """
    Context hook specification.

    This hook type is used to define context managers that can be stacked together.
    """

    ctx_managers: collections.abc.Sequence[contextlib.AbstractContextManager] = dataclasses.field(
        default=(), init=False
    )

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> contextlib.AbstractContextManager:
        self.ctx_managers = [func(*args, **kwargs) for func in self.callbacks]
        return self

    def __enter__(self) -> None:
        for ctx_manager in self.ctx_managers:
            ctx_manager.__enter__()

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        for ctx_manager in reversed(self.ctx_managers):
            ctx_manager.__exit__(type_, value, traceback)
        self.ctx_managers = ()
