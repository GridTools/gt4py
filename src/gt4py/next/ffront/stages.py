# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Definitions of the stages of the GT4Py frontend.

Classes in this module contain different forms of field operator and program
definitions, which are used as input or output of the different stages of
the frontend.

All classes containing a definition of a GT4Py computation in any form use the
`Def` suffix. Definitions containing actual Python functions whose source code
should be interpreted as GT4Py embedded domain-specific language have `DSL` in
their name. Definitions containing definitions as an AST of one the internal GT4Py
dialects contain `AST`.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import functools
import hashlib
import types
import typing
from typing import Any, Optional, TypeVar

import xxhash

from gt4py.eve import extended_typing as xtyping
from gt4py.next import common
from gt4py.next.ffront import field_operator_ast as foast, program_ast as past, source_utils
from gt4py.next.otf import arguments, toolchain


@dataclasses.dataclass(frozen=True)
class DSLFieldOperatorDef:
    definition: types.FunctionType
    node_class: type[foast.OperatorNode] = foast.FieldOperator
    attributes: dict[str, Any] = dataclasses.field(default_factory=dict)
    grid_type: Optional[common.GridType] = None
    debug: bool = False


ConcreteDSLFieldOperatorDef: typing.TypeAlias = toolchain.ConcreteArtifact[
    DSLFieldOperatorDef, arguments.CompileTimeArgs
]


@dataclasses.dataclass(frozen=True)
class FOASTOperatorDef:
    foast_node: foast.OperatorNode
    closure_vars: dict[str, Any]
    grid_type: Optional[common.GridType] = None
    attributes: dict[str, Any] = dataclasses.field(default_factory=dict)
    debug: bool = False


ConcreteFOASTOperatorDef: typing.TypeAlias = toolchain.ConcreteArtifact[
    FOASTOperatorDef, arguments.CompileTimeArgs
]


@dataclasses.dataclass(frozen=True)
class DSLProgramDef:
    definition: types.FunctionType
    grid_type: Optional[common.GridType] = None
    debug: bool = False


ConcreteDSLProgramDef: typing.TypeAlias = toolchain.ConcreteArtifact[
    DSLProgramDef, arguments.CompileTimeArgs
]


@dataclasses.dataclass(frozen=True)
class PASTProgramDef:
    past_node: past.Program
    closure_vars: dict[str, Any]
    grid_type: Optional[common.GridType] = None
    debug: bool = False


ConcretePASTProgramDef: typing.TypeAlias = toolchain.ConcreteArtifact[
    PASTProgramDef, arguments.CompileTimeArgs
]

DSLDefinitionT = TypeVar("DSLDefinitionT", DSLFieldOperatorDef, DSLProgramDef)


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


@add_content_to_fingerprint.register(DSLFieldOperatorDef)
@add_content_to_fingerprint.register(FOASTOperatorDef)
@add_content_to_fingerprint.register(DSLProgramDef)
@add_content_to_fingerprint.register(PASTProgramDef)
@add_content_to_fingerprint.register(toolchain.ConcreteArtifact)
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
