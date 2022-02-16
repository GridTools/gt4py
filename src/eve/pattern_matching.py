# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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
import inspect
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Iterator


@dataclass(frozen=True)
class ObjectPattern:
    cls: type
    attrs: dict[str, Any]

    def matches(self, other: Any, raise_: bool = False) -> bool:
        """Return if object pattern matches `other` using :func:`get_differences`.

        If `raise_` is specified raises an exception with all differences found.
        """
        diffs = list(get_differences(self, other))
        if raise_ and len(diffs) != 0:
            diffs_str = "\n  ".join([f"  {self.cls.__name__}{path}: {msg}" for path, msg in diffs])
            raise ValueError(f"Object and pattern don't match:\n  {diffs_str}")
        return len(diffs) == 0

    def __str__(self) -> str:
        attrs_str = ", ".join([f"{str(k)}={str(v)}" for k, v in self.attrs.items()])
        return f"{self.cls.__name__}({attrs_str})"


@dataclass(frozen=True)
class ObjectPatternConstructor:
    cls: type

    def __call__(self, **kwargs: Any) -> ObjectPattern:
        return ObjectPattern(self.cls, kwargs)


def _get_differences_object_pattern(
    a: ObjectPattern, b: Any, path: str = ""
) -> Iterator[tuple[str, str]]:
    if not isinstance(b, a.cls):
        yield (
            path,
            f"Expected an instance of class {a.cls.__name__}, but got {type(b).__name__}",
        )
    else:
        for k in a.attrs.keys():
            if not hasattr(b, k):
                yield (path, f"Value has no attribute {k}.")
            else:
                for diff in get_differences(a.attrs[k], getattr(b, k), path=f"{path}.{k}"):
                    yield diff


def _get_differences_list(a: list, b: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if not isinstance(b, list):
        yield (path, f"Expected list, but got {type(b).__name__}")
    elif len(a) != len(b):
        yield (path, f"Expected list of length {len(a)}, but got length {len(b)}")
    else:
        for i, (el_a, el_b) in enumerate(zip(a, b)):
            for diff in get_differences(el_a, el_b, path=f"{path}[{i}]"):
                yield diff


def _get_differences_dict(a: dict, b: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if not isinstance(b, dict):
        yield (path, f"Expected dict, but got {type(b).__name__}")
    elif set(a.keys()) != set(b.keys()):
        a_min_b = set(a.keys()).difference(b.keys())
        b_min_a = set(b.keys()).difference(a.keys())
        if a_min_b:
            missing_keys_str = "`" + "`, `".join(map(str, a_min_b)) + "`"
            yield (
                path,
                f"Expected dictionary with keys `{'`, `'.join(map(str, a.keys()))}`, but the following keys are missing: {missing_keys_str}",
            )
        if b_min_a:
            extra_keys_str = "`" + "`, `".join(map(str, b_min_a)) + "`"
            yield (
                path,
                f"Expected dictionary with keys `{'`, `'.join(map(str, a.keys()))}`, but the following keys are extra: {extra_keys_str}",
            )
    else:
        for k, v_a, v_b in zip(a.keys(), a.values(), b.values()):
            yield from get_differences(v_a, v_b, path=f'{path}["{k}"]')


def get_differences(a: Any, b: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Compare two objects and return a list of differences.

    If the arguments are lists or dictionaries comparison is recursively per item. Objects are compared
    using equality operator or if the left-hand-side is an `ObjectPattern` its type and attributes
    are compared to the right-hand-side object. Only the attributes of the `ObjectPattern` are used
    for comparison, disregarding potential additional attributes of the right-hand-side.
    """
    if isinstance(a, ObjectPattern):
        yield from _get_differences_object_pattern(a, b, path)
    elif isinstance(a, list):
        yield from _get_differences_list(a, b, path)
    elif isinstance(a, dict):
        yield from _get_differences_dict(a, b, path)
    else:
        if type(a) != type(b):
            yield (path, f"Expected a value of type {type(a).__name__}, but got {type(b).__name__}")
        elif a != b:
            yield (path, f"Values are not equal. `{a}` != `{b}`")


@dataclass(frozen=True)
class ModuleWrapper:
    """
    Small wrapper to conveniently create `ObjectPattern`s for classes of a module.

    Example:
    >>> import foo_ir  # doctest: +SKIP
    >>> foo_ir_ = ModuleWrapper(foo_ir)  # doctest: +SKIP
    >>> assert foo_ir_.Foo(bar="baz").matches(foo_ir.Foo(bar="baz", foo="bar"))  # doctest: +SKIP
    >>> assert not foo_ir_.Foo(bar="bar").matches(foo_ir.Foo(bar="baz", foo="bar"))  # doctest: +SKIP
    """

    module: ModuleType

    def __getattr__(self, item: str) -> ObjectPatternConstructor:
        val = getattr(self.module, item)
        if not inspect.isclass(val):
            raise ValueError("Only classes allowed.")
        return ObjectPatternConstructor(val)
