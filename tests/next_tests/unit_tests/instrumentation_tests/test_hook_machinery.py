# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import dataclasses

import pytest

from gt4py.next.instrumentation.hook_machinery import (
    EventHook,
    ContextHook,
    _get_unique_name,
    _is_empty_function,
)


def test_get_unique_name():
    def func1():
        pass

    def func2():
        pass

    assert _get_unique_name(func1) != _get_unique_name(func2)

    class A:
        def __call__(self): ...

    assert _get_unique_name(A) == _get_unique_name(A)

    a1, a2 = A(), A()

    assert (a1_name := _get_unique_name(a1)) != (a2_name := _get_unique_name(a2))
    assert _get_unique_name(a1) == a1_name
    assert _get_unique_name(a2) == a2_name


def test_empty_function():
    def empty():
        pass

    assert _is_empty_function(empty) is True

    def non_empty():
        return 1

    assert _is_empty_function(non_empty) is False

    def with_docstring():
        """This is a docstring."""

    assert _is_empty_function(with_docstring) is True

    def with_ellipsis(): ...

    assert _is_empty_function(with_ellipsis) is True

    class A:
        def __call__(self): ...

    assert _is_empty_function(A()) is True


class TestEventHook:
    def test_event_hook_call_with_no_callbacks(self):
        @EventHook
        def hook(x: int) -> None:
            pass

        hook(42)  # Should not raise

    def test_event_hook_call_with_callbacks(self):
        results = []

        @EventHook
        def hook(x: int) -> None:
            pass

        def callback1(x: int) -> None:
            results.append(x)

        def callback2(x: int) -> None:
            results.append(x * 2)

        hook.register(callback1)
        hook.register(callback2)
        hook(5)

        assert results == [5, 10]

    def test_event_hook_register_with_signature_mismatch(self):
        @EventHook
        def hook(x: int) -> None:
            pass

        def bad_callback(x: int, y: int) -> None:
            pass

        with pytest.raises(ValueError, match="Callback signature"):
            hook.register(bad_callback)

    def test_event_hook_register_with_annotation_mismatch(self):
        @EventHook
        def hook(x: int) -> None:
            pass

        def weird_callback(x: str) -> None:
            pass

        with pytest.warns(UserWarning, match="Callback annotations"):
            hook.register(weird_callback)

    def test_event_hook_register_with_name(self):
        @EventHook
        def hook(x: int) -> None:
            pass

        def callback(x: int) -> None:
            pass

        hook.register(callback, name="my_callback")

        assert "my_callback" in hook.registry

    def test_event_hook_register_with_index(self):
        results = []

        @EventHook
        def hook(x: int) -> None:
            pass

        def callback1(x: int) -> None:
            results.append(1)

        def callback2(x: int) -> None:
            results.append(2)

        hook.register(callback1)
        hook.register(callback2, index=0)
        hook(0)

        assert results == [2, 1]

    def test_event_hook_remove_by_name(self):
        results = []

        @EventHook
        def hook(x: int) -> None:
            pass

        def callback(x: int) -> None:
            results.append(x)

        hook.register(callback, name="test_cb")
        hook(42)
        assert results == [42]

        hook.remove("test_cb")
        results = []
        hook(42)

        assert results == []

    def test_event_hook_remove_by_callback(self):
        results = []

        @EventHook
        def hook(x: int) -> None:
            pass

        def callback(x: int) -> None:
            results.append(x)

        hook.register(callback)
        hook(42)
        assert results == [42]

        hook.remove(callback)
        results = []
        hook(42)

        assert results == []

    def test_event_hook_remove_nonexistent_raises(self):
        @EventHook
        def hook(x: int) -> None:
            pass

        with pytest.raises(KeyError):
            hook.remove("nonexistent")


class TestContextHook:
    def test_context_hook_basic(self):
        enter_called = []
        exit_called = []

        @ContextHook
        def hook() -> contextlib.AbstractContextManager:
            pass

        @contextlib.contextmanager
        def callback():
            enter_called.append(True)
            yield
            exit_called.append(True)

        hook.register(callback)

        with hook():
            assert len(enter_called) == 1

        assert len(exit_called) == 1

    def test_context_hook_multiple_callbacks(self):
        order = []

        @ContextHook
        def hook() -> contextlib.AbstractContextManager:
            pass

        @contextlib.contextmanager
        def callback1():
            order.append("enter1")
            yield
            order.append("exit1")

        @contextlib.contextmanager
        def callback2():
            order.append("enter2")
            yield
            order.append("exit2")

        hook.register(callback1)
        hook.register(callback2)

        with hook():
            pass

        # Entry in order, but exit in reverse
        assert order == ["enter1", "enter2", "exit2", "exit1"]

    def test_context_hook_with_arguments(self):
        results = []

        @ContextHook
        def hook(x: int) -> contextlib.AbstractContextManager:
            pass

        @contextlib.contextmanager
        def callback(x: int):
            results.append(x)
            yield

        hook.register(callback)

        with hook(42):
            pass

        assert results == [42]
