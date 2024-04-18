# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

from __future__ import annotations

import collections
import dataclasses
import functools
import hashlib
import types
import typing
from typing import Any, Generic, Optional, TypeVar

import xxhash

from gt4py import eve
from gt4py.next import common
from gt4py.next.ffront import field_operator_ast as foast, program_ast as past, source_utils
from gt4py.next.type_system import type_specifications as ts


if typing.TYPE_CHECKING:
    from gt4py.next.ffront import decorator


OperatorNodeT = TypeVar("OperatorNodeT", bound=foast.LocatedNode)


@dataclasses.dataclass(frozen=True)
class FieldOperatorDefinition(Generic[OperatorNodeT]):
    definition: types.FunctionType
    grid_type: Optional[common.GridType] = None
    node_class: type[OperatorNodeT] = dataclasses.field(default=foast.FieldOperator)  # type: ignore[assignment] # TODO(ricoh): understand why mypy complains
    attributes: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class FoastOperatorDefinition(Generic[OperatorNodeT]):
    foast_node: OperatorNodeT
    closure_vars: dict[str, Any]
    grid_type: Optional[common.GridType] = None
    attributes: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class FoastWithTypes(Generic[OperatorNodeT]):
    foast_op_def: FoastOperatorDefinition[OperatorNodeT]
    arg_types: tuple[ts.TypeSpec, ...]
    kwarg_types: dict[str, ts.TypeSpec]
    closure_vars: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class FoastClosure(Generic[OperatorNodeT]):
    foast_op_def: FoastOperatorDefinition[OperatorNodeT]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    closure_vars: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class ProgramDefinition:
    definition: types.FunctionType
    grid_type: Optional[common.GridType] = None


@dataclasses.dataclass(frozen=True)
class PastProgramDefinition:
    past_node: past.Program
    closure_vars: dict[str, Any]
    grid_type: Optional[common.GridType] = None


@dataclasses.dataclass(frozen=True)
class PastClosure:
    closure_vars: dict[str, Any]
    past_node: past.Program
    grid_type: Optional[common.GridType]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


Hasher_T: typing.TypeAlias = eve.extended_typing.HashlibAlgorithm | xxhash.xxh64 | hashlib._Hash


def cache_key(obj: Any, algorithm: Optional[str | Hasher_T] = None) -> str:
    hasher: Hasher_T
    if not algorithm:
        hasher = xxhash.xxh64()
    elif isinstance(algorithm, str):
        hasher = hashlib.new(algorithm)
    else:
        hasher = algorithm

    update_cache_key(obj, hasher)
    return hasher.hexdigest()


@functools.singledispatch
def update_cache_key(obj: Any, hasher: Hasher_T) -> None:
    if dataclasses.is_dataclass(obj):
        update_cache_key(obj.__class__, hasher)
        for field in dataclasses.fields(obj):
            update_cache_key(getattr(obj, field.name), hasher)
    # the following is to avoid circular dependencies
    elif hasattr(obj, "backend"):  # assume it is a decorator wrapper
        update_cache_key_fielop(obj, hasher)
    else:
        hasher.update(str(obj).encode())


@update_cache_key.register
def update_cache_key_str(obj: str, hasher: Hasher_T) -> None:
    hasher.update(str(obj).encode())


@update_cache_key.register
def update_cache_key_builtins(
    obj: str | None | bool | int | float,
    hasher: Hasher_T,
) -> None:
    hasher.update(str(obj).encode())


@update_cache_key.register
def update_cache_key_func(obj: types.FunctionType, hasher: Hasher_T) -> None:
    sourcedef = source_utils.SourceDefinition.from_function(obj)
    for item in sourcedef:
        update_cache_key(item, hasher)


@update_cache_key.register
def update_cache_key_dict(obj: dict, hasher: Hasher_T) -> None:
    for key, value in obj.items():
        update_cache_key(key, hasher)
        update_cache_key(value, hasher)


@update_cache_key.register
def update_cache_key_type(obj: type, hasher: Hasher_T) -> None:
    hasher.update(obj.__name__.encode())


@update_cache_key.register
def update_cache_key_sequence(
    obj: tuple | list | collections.abc.Iterable, hasher: Hasher_T
) -> None:
    for item in obj:
        update_cache_key(item, hasher)


@update_cache_key.register
def update_cache_key_foast(obj: foast.LocatedNode, hasher: Hasher_T) -> None:
    update_cache_key(obj.location, hasher)
    update_cache_key(str(obj), hasher)


# not registered to avoid circular dependencies
def update_cache_key_fielop(
    obj: decorator.FieldOperator | decorator.Program,
    hasher: Hasher_T,
) -> None:
    if hasattr(obj, "definition_stage"):
        update_cache_key(obj.definition_stage, hasher)
    elif hasattr(obj, "foast_stage"):
        update_cache_key(obj.foast_stage, hasher)
    elif hasattr(obj, "past_stage"):
        update_cache_key(obj.past_stage, hasher)
    update_cache_key(obj.backend, hasher)
