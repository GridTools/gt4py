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

import dataclasses
import functools
import types
import typing
from typing import Any, Optional, TypeVar

from gt4py.eve import utils as eve_utils
from gt4py.next import common, utils
from gt4py.next.ffront import field_operator_ast as foast, program_ast as past, source_utils
from gt4py.next.otf import arguments, toolchain


# Create a custom pickler for the BaseStage `fingerprinter` that handles
# `types.FunctionType` by using its source code and closure variables.
# This should be enough for the use case of GT4Py DSL definitions,
# which are expected to be pure functions without complicated closures.
fingerprint_reducer = eve_utils.singledispatcher(
    utils.fingerprint_reducer,
    {
        types.FunctionType: lambda f: (
            f.__class__,
            (),
            (
                source_utils.make_source_definition_from_function(f),
                source_utils.get_closure_vars_from_function(f),
            ),
        )
    },
)

fingerprinter = functools.partial(
    eve_utils.content_hash,
    pickler=eve_utils.custom_pickler(fingerprint_reducer, name="FFrontFingerprintPickler"),
)


@dataclasses.dataclass(frozen=True)
class BaseStage(utils.FingerprintedDataclass):
    """Base class for optimized Fingerprinted implementations in frozen dataclasses."""

    fingerprinter = staticmethod(fingerprinter)


@dataclasses.dataclass(frozen=True)
class DSLFieldOperatorDef(BaseStage):
    definition: types.FunctionType
    node_class: type[foast.OperatorNode] = foast.FieldOperator
    attributes: dict[str, Any] = dataclasses.field(default_factory=dict)
    grid_type: Optional[common.GridType] = None
    debug: bool = False


ConcreteDSLFieldOperatorDef: typing.TypeAlias = toolchain.ConcreteArtifact[
    DSLFieldOperatorDef, arguments.CompileTimeArgs
]


@dataclasses.dataclass(frozen=True)
class FOASTOperatorDef(BaseStage):
    foast_node: foast.OperatorNode
    closure_vars: dict[str, Any]
    grid_type: Optional[common.GridType] = None
    attributes: dict[str, Any] = dataclasses.field(default_factory=dict)
    debug: bool = False


ConcreteFOASTOperatorDef: typing.TypeAlias = toolchain.ConcreteArtifact[
    FOASTOperatorDef, arguments.CompileTimeArgs
]


@dataclasses.dataclass(frozen=True)
class DSLProgramDef(BaseStage):
    definition: types.FunctionType
    grid_type: Optional[common.GridType] = None
    debug: bool = False


ConcreteDSLProgramDef: typing.TypeAlias = toolchain.ConcreteArtifact[
    DSLProgramDef, arguments.CompileTimeArgs
]


@dataclasses.dataclass(frozen=True)
class PASTProgramDef(BaseStage):
    past_node: past.Program
    closure_vars: dict[str, Any]
    grid_type: Optional[common.GridType] = None
    debug: bool = False


ConcretePASTProgramDef: typing.TypeAlias = toolchain.ConcreteArtifact[
    PASTProgramDef, arguments.CompileTimeArgs
]

DSLDefinition = DSLFieldOperatorDef | DSLProgramDef
DSLDefinitionT = TypeVar("DSLDefinitionT", DSLFieldOperatorDef, DSLProgramDef)
