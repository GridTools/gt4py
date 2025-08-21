# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:
    from gt4py.next.type_system import type_specifications as ts

# functools.singledispatch would be nice, but would have to be on the metaclass level
_registry: list[tuple[Callable[[type], bool], Callable[[type], ts.TupleType] | ts.TupleType]] = []


# for users
def register(
    predicate: Callable[[type], bool], func_or_type: Callable[[type], ts.TupleType] | ts.TupleType
) -> None:
    _registry.append((predicate, func_or_type))


# only internally
def get_constructor_type(cls: type) -> ts.TupleType:
    for predicate, func_or_type in _registry:
        if predicate(cls):
            if callable(func_or_type):
                return func_or_type(cls)
            return func_or_type
    raise KeyError(f"get_constructor_type not implemented for {cls}.")
