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
def flatten(arg) -> tuple | common.Field:
    return (
        arg
        if isinstance(arg, common.Field) or not dataclasses.is_dataclass(arg)
        else tuple(getattr(arg, f.name) for f in dataclasses.fields(arg))
    )


import functools
from collections.abc import Callable
from typing import Any, TypeAlias

from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing as xtyping
from gt4py.eve.extended_typing import NestedTuple


NestedTupleConstructor: TypeAlias = Callable[[Any], NestedTuple[core_defs.Scalar | common.Field]]


@functools.cache
def make_container_extractor(
    type_spec: ts.TypeSpec, container_type: type[core_defs.PythonContainer]
) -> NestedTupleConstructor:
    assert isinstance(type_spec, ts.NamedTupleType)
    assert issubclass(container_type, core_defs.PythonContainer)

    def make_expr(in_arg: str, type_spec: ts.NamedTupleType, container_type: type[core_defs.PythonContainer]):
        children_hints = get_type_hints(container_type)
        nested_types = {
            key: make_expr(f"{in_arg}.{key}", spec, children_hints[key])
            for key, spec in zip(type_spec.keys, type_spec.types)
            if isinstance(spec, ts.NamedTupleType)
        }
        if nested_types or not isinstance(container_type, tuple):
            return f"({', '.join([nested_types.get(key, f'{in_arg}.{key}') for key in type_spec.keys])})"
        else:
            return f"{in_arg}"

    return eval(f"lambda x: {make_expr('x', type_spec, container_type)}")
