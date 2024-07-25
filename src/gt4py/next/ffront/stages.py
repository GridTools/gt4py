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

from gt4py.eve import extended_typing as xtyping
from gt4py.next import common
from gt4py.next.ffront import field_operator_ast as foast, program_ast as past, source_utils
from gt4py.next.otf import arguments, workflow


OperatorNodeT = TypeVar("OperatorNodeT", bound=foast.LocatedNode)


@dataclasses.dataclass(frozen=True)
class FieldOperatorDefinition(Generic[OperatorNodeT]):
    definition: types.FunctionType
    grid_type: Optional[common.GridType] = None
    node_class: type[OperatorNodeT] = dataclasses.field(default=foast.FieldOperator)  # type: ignore[assignment] # TODO(ricoh): understand why mypy complains
    attributes: dict[str, Any] = dataclasses.field(default_factory=dict)
    debug: bool = False


DSL_FOP: typing.TypeAlias = FieldOperatorDefinition
AOT_DSL_FOP: typing.TypeAlias = workflow.DataArgsPair[DSL_FOP, arguments.CompileTimeArgs]


@dataclasses.dataclass(frozen=True)
class FoastOperatorDefinition(Generic[OperatorNodeT]):
    foast_node: OperatorNodeT
    closure_vars: dict[str, Any]
    grid_type: Optional[common.GridType] = None
    attributes: dict[str, Any] = dataclasses.field(default_factory=dict)
    debug: bool = False


FOP: typing.TypeAlias = FoastOperatorDefinition
AOT_FOP: typing.TypeAlias = workflow.DataArgsPair[FOP, arguments.CompileTimeArgs]


@dataclasses.dataclass(frozen=True)
class ProgramDefinition:
    definition: types.FunctionType
    grid_type: Optional[common.GridType] = None
    debug: bool = False


DSL_PRG: typing.TypeAlias = ProgramDefinition
AOT_DSL_PRG: typing.TypeAlias = workflow.DataArgsPair[DSL_PRG, arguments.CompileTimeArgs]


@dataclasses.dataclass(frozen=True)
class PastProgramDefinition:
    past_node: past.Program
    closure_vars: dict[str, Any]
    grid_type: Optional[common.GridType] = None
    debug: bool = False


PRG: typing.TypeAlias = PastProgramDefinition
AOT_PRG: typing.TypeAlias = workflow.DataArgsPair[PRG, arguments.CompileTimeArgs]


def fingerprint_stage(obj: Any, algorithm: Optional[str | xtyping.HashlibAlgorithm] = None) -> str:
    hasher: xtyping.HashlibAlgorithm
    if not algorithm:
        hasher = xxhash.xxh64()
    elif isinstance(algorithm, str):
        hasher = hashlib.new(algorithm)
    else:
        hasher = algorithm

    add_content_to_fingerprint(obj, hasher)
    import devtools

    if hasattr(obj, "definition"):
        print(obj.definition.id if hasattr(obj.definition, "id") else obj.definition.__name__)
    print(id(obj))
    print(type(obj))
    print(id(obj))
    devtools.debug(hasher.hexdigest())
    return hasher.hexdigest()


@functools.singledispatch
def add_content_to_fingerprint(obj: Any, hasher: xtyping.HashlibAlgorithm) -> None:
    hasher.update(str(obj).encode())


for t in (str, int):
    add_content_to_fingerprint.register(t, add_content_to_fingerprint.registry[object])


@add_content_to_fingerprint.register(FieldOperatorDefinition)
@add_content_to_fingerprint.register(FoastOperatorDefinition)
@add_content_to_fingerprint.register(PastProgramDefinition)
@add_content_to_fingerprint.register(workflow.DataArgsPair)
@add_content_to_fingerprint.register(arguments.CompileTimeArgs)
def add_stage_to_fingerprint(obj: Any, hasher: xtyping.HashlibAlgorithm) -> None:
    add_content_to_fingerprint(obj.__class__, hasher)
    for field in dataclasses.fields(obj):
        add_content_to_fingerprint(getattr(obj, field.name), hasher)


def add_jit_args_id_to_fingerprint(
    obj: arguments.JITArgs, hasher: xtyping.HashlibAlgorithm
) -> None:
    add_content_to_fingerprint(str(id(obj)), hasher)


@add_content_to_fingerprint.register
def add_func_to_fingerprint(obj: types.FunctionType, hasher: xtyping.HashlibAlgorithm) -> None:
    sourcedef = source_utils.SourceDefinition.from_function(obj)
    for item in sourcedef:
        add_content_to_fingerprint(item, hasher)


@add_content_to_fingerprint.register
def add_dict_to_fingerprint(obj: dict, hasher: xtyping.HashlibAlgorithm) -> None:
    for key, value in obj.items():
        print(f"{key}: {value} added to fingerprint")
        add_content_to_fingerprint(key, hasher)
        add_content_to_fingerprint(value, hasher)


@add_content_to_fingerprint.register
def add_type_to_fingerprint(obj: type, hasher: xtyping.HashlibAlgorithm) -> None:
    hasher.update(obj.__name__.encode())


@add_content_to_fingerprint.register
def add_sequence_to_fingerprint(
    obj: collections.abc.Iterable, hasher: xtyping.HashlibAlgorithm
) -> None:
    for item in obj:
        add_content_to_fingerprint(item, hasher)


@add_content_to_fingerprint.register
def add_foast_located_node_to_fingerprint(
    obj: foast.LocatedNode, hasher: xtyping.HashlibAlgorithm
) -> None:
    add_content_to_fingerprint(obj.location, hasher)
    add_content_to_fingerprint(str(obj), hasher)
    add_content_to_fingerprint(str(obj), hasher)
