# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable

from gt4py.next import common


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


# TODO customize
def flatten(arg):
    return (
        arg
        if isinstance(arg, common.Field) or not dataclasses.is_dataclass(arg)
        else tuple(getattr(arg, f.name) for f in dataclasses.fields(arg))
    )



# import functools
# from collections.abc import Callable
# from typing import TypeAlias, Any
# from gt4py.eve.extended_typing import NestedTuple
# from gt4py.eve import extended_typing as xtyping
# from gt4py._core import definitions as core_defs


# NestedTupleConstructor: TypeAlias = Callable[[Any], NestedTuple[core_defs.Scalar | Field]]

# @functools.cache
# def nested_tuple_constructor(container_type: type) -> NestedTupleConstructor:

#     if issubclass(container_type, tuple):
#         return 
        
#     if issubclass(container_type, xtyping.DataClass):
#         return lambda x: {k: nested_tuple_constructor(type(v))(v) for k, v in x.items()}
    
# return  (c.field1, c.field2, ((f3:=c.field3.foo), c_field3.f4))

