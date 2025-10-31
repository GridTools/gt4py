# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Basic utilities for Python programming."""

from __future__ import annotations

import collections.abc
import functools
import hashlib
import importlib.util
import itertools
import json
import os
import string
import sys
import time
import types
import warnings
from typing import Generic, TypeGuard, TypeVar

from gt4py.cartesian import config as gt_config


NOTHING = object()


def slugify(value: str, *, replace_spaces=True, valid_symbols="-_.()", invalid_marker=""):
    valid_chars = valid_symbols + string.ascii_letters + string.digits
    slug = "".join(c if c in valid_chars else invalid_marker for c in value)
    if replace_spaces:
        slug = slug.replace(" ", "_")
    return slug


def jsonify(value, indent=2):
    return json.dumps(value, indent=indent, default=lambda obj: str(obj))


def is_identifier_name(value: object, namespaced: bool = True) -> TypeGuard[str]:
    if isinstance(value, str):
        if namespaced:
            return all(name.isidentifier() for name in value.split("."))

        return value.isidentifier()

    return False


def listify(value):
    return value if isinstance(value, collections.abc.Sequence) else [value]


def flatten(nested_iterables, filter_none=False, *, skip_types=(str, bytes)):
    return list(flatten_iter(nested_iterables, filter_none, skip_types=skip_types))


def flatten_iter(nested_iterables, filter_none=False, *, skip_types=(str, bytes)):
    for item in nested_iterables:
        if isinstance(item, collections.abc.Iterable) and not isinstance(item, skip_types):
            yield from flatten(item, filter_none)
        else:
            if item is not None or not filter_none:
                yield item


def get_member(instance, item_name):
    try:
        if isinstance(instance, collections.abc.Mapping) or (
            isinstance(instance, collections.abc.Sequence) and isinstance(item_name, int)
        ):
            return instance[item_name]

        return getattr(instance, item_name)
    except Exception:
        return NOTHING


def compose(*functions_or_iterable):
    """Return a function that chains the input functions.

    Derived from: https://mathieularose.com/function-composition-in-python/
    """
    if len(functions_or_iterable) == 1 and isinstance(
        functions_or_iterable[0], collections.abc.Iterable
    ):
        functions = functions_or_iterable[0]
    else:
        functions = functions_or_iterable

    return functools.reduce(
        lambda this_func, previous_func: lambda x: this_func(previous_func(x)),
        functions,
        lambda x: x,
    )


def is_collection(value, *, optional=False, accept_mapping=True):
    if value is None and optional:
        return True
    if (
        value is not None
        and isinstance(value, collections.abc.Iterable)
        and not isinstance(value, (str, bytes))
    ):
        return accept_mapping or not isinstance(value, collections.abc.Mapping)

    return False


def is_iterable_of(
    value,
    item_class,
    iterable_class=collections.abc.Iterable,
    *,
    optional=False,
    accept_mapping=True,
):
    try:
        if value is None and optional:
            return True
        if value is not None and (
            isinstance(value, iterable_class) and all([isinstance(i, item_class) for i in value])
        ):
            return accept_mapping or not isinstance(value, collections.abc.Mapping)

    except Exception:
        pass

    return False


def is_mapping_of(
    value, key_class, value_class=None, mapping_class=collections.abc.Mapping, *, optional=False
):
    try:
        if value is None and optional:
            return True
        if (
            value is not None
            and (isinstance(value, mapping_class))
            and all([isinstance(k, key_class) for k in value.keys()])
            and (value_class is None or all([isinstance(v, value_class) for v in value.values()]))
        ):
            return True

    except Exception:
        pass

    return False


def normalize_mapping(mapping, key_types=(object,), *, filter_none=False):
    assert isinstance(mapping, collections.abc.Mapping)
    if isinstance(key_types, collections.abc.Iterable):
        assert all(isinstance(k, type) for k in key_types)
        key_types = tuple(key_types)
    result = {}
    for key, value in mapping.items():
        if not filter_none or value is not None:
            if is_iterable_of(key, key_types, iterable_class=(tuple, list)):
                for k in key:
                    result[k] = value
            else:
                result[key] = value

    return result


def shash(*args, hash_algorithm: hashlib._Hash | None = None, length: int | None = None) -> str:
    """Hash the given arguments.

    Args:
        hash_algorithm: Specify the hashlib algorithm used. Defaults to sha 256.
        length: Trim to the first `length` digits of the hash. Returns the full hash by default.
    """
    if hash_algorithm is None:
        hash_algorithm = hashlib.sha256()

    for item in args:
        if not isinstance(item, bytes):
            if isinstance(item, collections.abc.Mapping):
                item = list(flatten(sorted(item.items())))
            elif not isinstance(item, str) and isinstance(item, collections.abc.Iterable):
                item = list(flatten(item))
            item = str.encode(repr(item))
        hash_algorithm.update(item)

    digest = hash_algorithm.hexdigest()
    if length is not None and length > len(digest):
        warnings.warn(
            f"Requested hash of length {length}, but the full hash's length is {len(digest)}. Returning the full hash.",
            stacklevel=2,
        )
        length = None
    return digest[:length] if length is not None else digest


def shashed_id(*args, length: int = 10) -> str:
    """Hash the given arguments and trim to length."""
    return shash(*args, length=length)


def classmethod_to_function(class_method, instance=None, owner=None, remove_cls_arg=False):
    if remove_cls_arg:
        return functools.partial(class_method.__get__(instance, owner), None)
    return class_method.__get__(instance, owner)


def namespace_from_nested_dict(nested_dict: dict) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        **{
            key: namespace_from_nested_dict(value) if isinstance(value, dict) else value
            for key, value in nested_dict.items()
        }
    )


def make_local_dir(dir_name, base_dir=None, *, mode=0o777, is_package=False, is_cache=False):
    if base_dir is None:
        base_dir = os.getcwd()
    dir_name = os.path.join(base_dir, dir_name)
    return make_dir(dir_name)


def make_dir(dir_name, *, mode=0o777, is_package=False, is_cache=False):
    dir_name = os.path.abspath(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

    if is_package:
        with open(os.path.join(dir_name, "__init__.py"), "a"):
            pass

    if is_cache:
        with open(os.path.join(dir_name, "CACHEDIR.TAG"), "w") as f:
            f.write(
                """Signature: 8a477f597d28d172789f06886806bc55
# This file is a cache directory tag created by GT4Py.
# For information about cache directory tags, see:
#	http://www.brynosaurus.com/cachedir/
"""
            )

    return dir_name


def make_module_from_file(qualified_name, file_path, *, public_import=False):
    """Import module from file.

    References:
      https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
      https://stackoverflow.com/a/43602645
    """

    def load():
        spec = importlib.util.spec_from_file_location(qualified_name, file_path)
        if not spec:
            raise ModuleNotFoundError(f"No module named '{qualified_name}'")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if public_import:
            sys.modules[qualified_name] = module
            package_name = getattr(module, "__package__", "")
            if not package_name:
                package_name = ".".join(qualified_name.split(".")[:-1])
                module.__package__ = package_name
            components = package_name.split(".")
            module_name = qualified_name.split(".")[-1]

            if components[0] in sys.modules:
                parent = sys.modules[components[0]]
                for current in components[1:]:
                    parent = getattr(parent, current, None)
                    if not parent:
                        break
                else:
                    setattr(parent, module_name, module)
        return module

    for _i in range(max(gt_config.cache_settings["load_retries"], 0)):
        try:
            return load()
        except ModuleNotFoundError:
            time.sleep(max(gt_config.cache_settings["load_retry_delay"], 0) / 1000)

    return load()


def patch_module(module, member, new_value, *, recursive=True):
    """Monkey patch a module replacing a member with a new value."""
    if not isinstance(module, types.ModuleType):
        raise ValueError("Invalid 'module' argument")

    pending = [module]
    visited = set()
    originals = {}

    while pending:
        patched = {}
        current, pending = pending[0], pending[1:]
        visited.add(current)

        for name, value in current.__dict__.items():
            if value is member:
                patched[name] = value
                current.__dict__[name] = new_value
            elif isinstance(value, types.ModuleType) and value not in visited:
                pending.append(value)

        if patched:
            originals[current] = patched

    return dict(
        module=module,
        original_value=member,
        patched_value=new_value,
        recursive=recursive,
        originals=originals,
    )


def restore_module(patch, *, verify=True):
    """Restore a module patched with the `patch_module()` function."""
    if not isinstance(patch, dict) or not {
        "module",
        "original_value",
        "patched_value",
        "recursive",
        "originals",
    } <= set(patch.keys()):
        raise ValueError("Invalid 'patch' definition")

    patched_value = patch["patched_value"]
    original_value = patch["original_value"]

    for current, changes in patch["originals"].items():
        for name, value in changes.items():
            assert (
                value is original_value and current.__dict__[name] is patched_value
            ) or not verify
            current.__dict__[name] = original_value


T = TypeVar("T")


class Registry(Generic[T], dict[str, T]):
    @property
    def names(self) -> list[str]:
        return list(self.keys())

    def register(self, name: str, item: T) -> T:
        if name in self.keys():
            raise ValueError(f"Name {name} already exists in registry.")

        def _wrapper(obj):
            self[name] = obj
            return obj

        return _wrapper(item)


class ClassProperty:
    """Like a :class:`property`, but the wrapped function is a class method."""

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        if doc is None and fget is not None:
            doc = fget.__doc__
        self.__doc__ = doc

    def __get__(self, instance, cls=None):
        if instance is None:
            if cls is None:
                return self
        elif cls is None:
            cls = type(instance)
        if self.fget is None:
            raise AttributeError("Invalid class property getter")
        return self.fget(cls)

    def __set__(self, instance, value):
        if self.fset is None:
            raise AttributeError("Invalid class property setter")
        self.fset(type(instance), value)

    def __delete__(self, instance):
        if self.fdel is None:
            raise AttributeError("Invalid class property deleter")
        self.fdel(type(instance))

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)


classproperty = ClassProperty


class BaseFrozen:
    def __setattr__(self, key, value):
        raise AttributeError("Attempting a modification of an attribute in a frozen class")

    def __delattr__(self, item):
        raise AttributeError("Attempting a deletion of an attribute in a frozen class")


class BaseSingleton:
    def __new__(cls, *args, **kwargs):
        if getattr(cls, "_instance", None) is None:
            cls._instance = object.__new__(cls, *args, **kwargs)

        return cls._instance


class UniqueIdGenerator:
    def __init__(self, start=1):
        self.counter = itertools.count(start=start)
        self._current = None

    @property
    def new(self):
        self._current = next(self.counter)
        return self._current

    @property
    def current(self):
        return self._current


def warn_experimental_feature(*, feature: str, ADR: str) -> None:
    """Warning message for experimental features.

    Experimental features are required to print a (one-time) warning message such that users
    know what to expect. We facilitate consistent warnings by providing this convenience
    function.

    Args:
        feature (str): Name of the experimental feature.
        ADR (str): Path to the associated ADR, relative to `docs/development/ADRs/cartesian/`

    Raises:
        ValueError: In case the the `ADR` can't be found.
    """

    # be nice and remove a potential `/` prefixed
    ADR = ADR.removeprefix("/")

    warnings.warn(
        f"{feature} is an experimental feature. Please read "
        f"<https://github.com/GridTools/gt4py/blob/main/docs/development/ADRs/cartesian/{ADR}> "
        "to understand the consequences.",
        category=UserWarning,
        stacklevel=2,
    )
