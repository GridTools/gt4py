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


# TODO(ricoh): This type seems to not really catch the relevant types
#   which leads to the ignores below
Hasher_T: typing.TypeAlias = eve.extended_typing.HashlibAlgorithm


def fingerprint_stage(obj: Any, algorithm: Optional[str | Hasher_T] = None) -> str:
    hasher: Hasher_T
    if not algorithm:
        hasher = xxhash.xxh64()  # type: ignore[assignment] # see todo above
    elif isinstance(algorithm, str):
        hasher = hashlib.new(algorithm)  # type: ignore[assignment] # see todo above
    else:
        hasher = algorithm

    add_content_to_fingerprint(obj, hasher)
    return hasher.hexdigest()


@functools.singledispatch
def add_content_to_fingerprint(obj: Any, hasher: Hasher_T) -> None:
    # the following is to avoid circular dependencies
    if hasattr(obj, "backend"):  # assume it is a decorator wrapper
        add_content_to_fingerprint_fielop(obj, hasher)
    else:
        hasher.update(str(obj).encode())


@add_content_to_fingerprint.register(FieldOperatorDefinition)
@add_content_to_fingerprint.register(FoastOperatorDefinition)
@add_content_to_fingerprint.register(FoastWithTypes)
@add_content_to_fingerprint.register(FoastClosure)
@add_content_to_fingerprint.register(ProgramDefinition)
@add_content_to_fingerprint.register(PastProgramDefinition)
@add_content_to_fingerprint.register(PastClosure)
def add_content_to_fingerprint_stages(obj: Any, hasher: Hasher_T) -> None:
    add_content_to_fingerprint(obj.__class__, hasher)
    for field in dataclasses.fields(obj):
        add_content_to_fingerprint(getattr(obj, field.name), hasher)


@add_content_to_fingerprint.register
def add_content_to_fingerprint_str(obj: str, hasher: Hasher_T) -> None:
    hasher.update(str(obj).encode())


@add_content_to_fingerprint.register(int)
@add_content_to_fingerprint.register(bool)
@add_content_to_fingerprint.register(float)
def add_content_to_fingerprint_builtins(
    obj: None,
    hasher: Hasher_T,
) -> None:
    hasher.update(str(obj).encode())


@add_content_to_fingerprint.register
def add_content_to_fingerprint_func(obj: types.FunctionType, hasher: Hasher_T) -> None:
    sourcedef = source_utils.SourceDefinition.from_function(obj)
    for item in sourcedef:
        add_content_to_fingerprint(item, hasher)


@add_content_to_fingerprint.register
def add_content_to_fingerprint_dict(obj: dict, hasher: Hasher_T) -> None:
    for key, value in obj.items():
        add_content_to_fingerprint(key, hasher)
        add_content_to_fingerprint(value, hasher)


@add_content_to_fingerprint.register
def add_content_to_fingerprint_type(obj: type, hasher: Hasher_T) -> None:
    hasher.update(obj.__name__.encode())


@add_content_to_fingerprint.register
def add_content_to_fingerprint_sequence(obj: collections.abc.Iterable, hasher: Hasher_T) -> None:
    for item in obj:
        add_content_to_fingerprint(item, hasher)


@add_content_to_fingerprint.register
def add_content_to_fingerprint_foast(obj: foast.LocatedNode, hasher: Hasher_T) -> None:
    add_content_to_fingerprint(obj.location, hasher)
    add_content_to_fingerprint(str(obj), hasher)


# not registered to avoid circular dependencies
def add_content_to_fingerprint_fielop(
    obj: decorator.FieldOperator | decorator.Program,
    hasher: Hasher_T,
) -> None:
    if hasattr(obj, "definition_stage"):
        add_content_to_fingerprint(obj.definition_stage, hasher)
    elif hasattr(obj, "foast_stage"):
        add_content_to_fingerprint(obj.foast_stage, hasher)
    elif hasattr(obj, "past_stage"):
        add_content_to_fingerprint(obj.past_stage, hasher)
    add_content_to_fingerprint(obj.backend, hasher)
