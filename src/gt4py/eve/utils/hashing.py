# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Hashing and diffing utilities."""

from __future__ import annotations

import functools
import io
import pickle
import pprint
from collections.abc import Callable
from typing import Any, Generic, Optional, TypeVar, overload

import deepdiff
import xxhash

from .. import xtyping
from ..type_definitions import NOTHING, NothingType


_T = TypeVar("_T")


class HashableBy(Generic[_T]):
    __slots__ = ("hashed_value", "value")

    def __init__(self, hash_func: Callable[[_T], int], value: _T) -> None:
        self.value = value
        self.hashed_value = hash_func(value)

    def __hash__(self) -> int:
        return self.hashed_value

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, HashableBy)
        return self.value is other.value or self.value == other.value

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(value={self.value!r}, hashed_value={self.hashed_value!r})"
        )


@overload
def hashable_by(func: Callable[[_T], int], value: _T) -> HashableBy[_T]: ...


@overload
def hashable_by(
    func: Callable[[_T], int], value: NothingType = NOTHING
) -> functools.partial[HashableBy[_T]]: ...


def hashable_by(
    func: Callable[[_T], int], value: _T | NothingType = NOTHING
) -> HashableBy[_T] | functools.partial[HashableBy[_T]]:
    """
    Creates a wrapper that uses `func` to hash the value passed to it.
    """
    return (
        HashableBy(func, value)  # type: ignore[arg-type] # checked in condition
        if value is not NOTHING
        else functools.partial(hashable_by, func)
    )


hashable_by_id = hashable_by(id)


cached_hash = hashable_by(hash)


class EqualityBy(HashableBy):
    """Use a hash function as the definition of equality for the wrapped object."""

    __hash__ = HashableBy.__hash__

    def __eq__(self, other: Any) -> bool:
        return self is other or hash(self) == hash(other)


#: Pickle protocol pinned for `content_hash` so the serialized byte stream (and
#: therefore the resulting hash) is reproducible across Python versions,
#: regardless of changes to `pickle.DEFAULT_PROTOCOL`.
_CONTENT_HASH_PICKLE_PROTOCOL: int = 5


def content_hash(
    *args: Any,
    hash_algorithm: xtyping.HashlibAlgorithm | None = None,
    pickler_type: type[pickle.Pickler] = pickle.Pickler,
) -> str:
    """Stable content-based hash function using instance serialization data.

    It provides a customizable hash function for any kind of data.
    Unlike the builtin `hash` function, it is stable (same hash value across
    interpreter reboots and Python versions) and it does not use hash
    customizations on user classes (it uses `pickle` internally to get a byte
    stream). The pickle protocol is pinned (:data:`CONTENT_HASH_PICKLE_PROTOCOL`)
    so the byte stream does not depend on the running Python's default protocol.

    Arguments:
        hash_algorithm: object implementing the `hash algorithm` interface
            from :mod:`hashlib`. Defaults to :class:`xxhash.xxh64`.

    """
    if hash_algorithm is None:
        hash_algorithm = xxhash.xxh64()  # type: ignore[assignment]
    assert hash_algorithm is not None

    buf = io.BytesIO()
    pickler_type(buf, protocol=_CONTENT_HASH_PICKLE_PROTOCOL).dump(args)

    hash_algorithm.update(buf.getvalue())
    result = hash_algorithm.hexdigest()
    assert isinstance(result, str)

    return result


ddiff = deepdiff.diff.DeepDiff
"""Shortcut for deepdiff.diff.DeepDiff.

Check https://zepworks.com/deepdiff/current/diff.html for more info.
"""


def dhash(obj: Any, **kwargs: Any) -> str:
    """Shortcut for deepdiff.deephash.DeepHash.

    Check https://zepworks.com/deepdiff/current/deephash.html for more info.
    """
    return deepdiff.deephash.DeepHash(obj)[obj]


def pprint_ddiff(
    old: Any, new: Any, *, pprint_opts: Optional[dict[str, Any]] = None, **kwargs: Any
) -> None:
    """Pretty printing of deepdiff.diff.DeepDiff objects.

    Keyword Arguments:
        pprint_opts: kwargs dict with options for pprint.pprint.
    """
    pprint_opts = pprint_opts or {"indent": 2}
    pprint.pprint(deepdiff.diff.DeepDiff(old, new, **kwargs), **pprint_opts)
