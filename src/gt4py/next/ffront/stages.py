# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
from gt4py.next.otf import arguments, toolchain


OperatorNodeT = TypeVar("OperatorNodeT", bound=foast.LocatedNode)


@dataclasses.dataclass(frozen=True)
class FieldOperatorDefinition(Generic[OperatorNodeT]):
    definition: types.FunctionType
    grid_type: Optional[common.GridType] = None
    node_class: type[OperatorNodeT] = dataclasses.field(default=foast.FieldOperator)  # type: ignore[assignment] # TODO(ricoh): understand why mypy complains
    attributes: dict[str, Any] = dataclasses.field(default_factory=dict)
    debug: bool = False


DSL_FOP: typing.TypeAlias = FieldOperatorDefinition
AOT_DSL_FOP: typing.TypeAlias = toolchain.CompilableProgram[DSL_FOP, arguments.CompileTimeArgs]


@dataclasses.dataclass(frozen=True)
class FoastOperatorDefinition(Generic[OperatorNodeT]):
    foast_node: OperatorNodeT
    closure_vars: dict[str, Any]
    grid_type: Optional[common.GridType] = None
    attributes: dict[str, Any] = dataclasses.field(default_factory=dict)
    debug: bool = False


FOP: typing.TypeAlias = FoastOperatorDefinition
AOT_FOP: typing.TypeAlias = toolchain.CompilableProgram[FOP, arguments.CompileTimeArgs]


@dataclasses.dataclass(frozen=True)
class ProgramDefinition:
    definition: types.FunctionType
    grid_type: Optional[common.GridType] = None
    debug: bool = False


DSL_PRG: typing.TypeAlias = ProgramDefinition
AOT_DSL_PRG: typing.TypeAlias = toolchain.CompilableProgram[DSL_PRG, arguments.CompileTimeArgs]


@dataclasses.dataclass(frozen=True)
class PastProgramDefinition:
    past_node: past.Program
    closure_vars: dict[str, Any]
    grid_type: Optional[common.GridType] = None
    debug: bool = False


PRG: typing.TypeAlias = PastProgramDefinition
AOT_PRG: typing.TypeAlias = toolchain.CompilableProgram[PRG, arguments.CompileTimeArgs]


def fingerprint_stage(obj: Any, algorithm: Optional[str | xtyping.HashlibAlgorithm] = None) -> str:
    hasher: xtyping.HashlibAlgorithm
    if not algorithm:
        hasher = xxhash.xxh64()  # type: ignore[assignment]  # fixing this requires https://github.com/ifduyue/python-xxhash/issues/104
    elif isinstance(algorithm, str):
        hasher = hashlib.new(algorithm)
    else:
        hasher = algorithm

    add_content_to_fingerprint(obj, hasher)
    return hasher.hexdigest()


@functools.singledispatch
def add_content_to_fingerprint(obj: Any, hasher: xtyping.HashlibAlgorithm) -> None:
    hasher.update(str(obj).encode())


for t in (str, int):
    add_content_to_fingerprint.register(t, add_content_to_fingerprint.registry[object])


@add_content_to_fingerprint.register(FieldOperatorDefinition)
@add_content_to_fingerprint.register(FoastOperatorDefinition)
@add_content_to_fingerprint.register(ProgramDefinition)
@add_content_to_fingerprint.register(PastProgramDefinition)
@add_content_to_fingerprint.register(toolchain.CompilableProgram)
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

    closure_vars = source_utils.get_closure_vars_from_function(obj)
    for item in sorted(closure_vars.items(), key=lambda x: x[0]):
        add_content_to_fingerprint(item, hasher)


@add_content_to_fingerprint.register
def add_dict_to_fingerprint(obj: dict, hasher: xtyping.HashlibAlgorithm) -> None:
    # just a small helper to additionally allow sorting types (by just using their name)
    def key_function(key: Any) -> Any:
        if isinstance(key, type):
            return key.__module__, key.__qualname__
        return key

    for key in sorted(obj.keys(), key=key_function):
        add_content_to_fingerprint(key, hasher)
        add_content_to_fingerprint(obj[key], hasher)


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
