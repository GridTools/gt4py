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

"""Basic utilities for Python programming."""

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

from gt4py.cartesian import config as gt_config


NOTHING = object()


def slugify(value: str, *, replace_spaces=True, valid_symbols="-_.()", invalid_marker=""):
    valid_chars = valid_symbols + string.ascii_letters + string.digits
    slug = "".join(c if c in valid_chars else invalid_marker for c in value)
    if replace_spaces:
        slug = slug.replace(" ", "_")
    return slug


# def stringify(value):
#     pass


def jsonify(value, indent=2):
    return json.dumps(value, indent=indent, default=lambda obj: str(obj))


def is_identifier_name(value, namespaced=True):
    if isinstance(value, str):
        if namespaced:
            return all(name.isidentifier() for name in value.split("."))
        else:
            return value.isidentifier()
    else:
        return False


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
        if (
            isinstance(instance, collections.abc.Mapping)
            or isinstance(instance, collections.abc.Sequence)
            and isinstance(item_name, int)
        ):
            return instance[item_name]
        else:
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


def shash(*args, hash_algorithm=None):
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

    return hash_algorithm.hexdigest()


def shashed_id(*args, length=10, hash_algorithm=None):
    return shash(*args, hash_algorithm=hash_algorithm)[:length]


def classmethod_to_function(class_method, instance=None, owner=type(None), remove_cls_arg=False):
    if remove_cls_arg:
        return functools.partial(class_method.__get__(instance, owner), None)
    else:
        return class_method.__get__(instance, owner)


def namespace_from_nested_dict(nested_dict):
    assert isinstance(nested_dict, dict)
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
                """Signature: 8
a477f597d28d172789f06886806bc55
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

    patch = dict(
        module=module,
        original_value=member,
        patched_value=new_value,
        recursive=recursive,
        originals=originals,
    )

    return patch


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


class Registry(dict):
    @property
    def names(self):
        return list(self.keys())

    def register(self, name, item=NOTHING):
        if name in self.keys():
            raise ValueError("Name already exists in registry")

        def _wrapper(obj):
            self[name] = obj
            return obj

        return _wrapper if item is NOTHING else _wrapper(item)


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
