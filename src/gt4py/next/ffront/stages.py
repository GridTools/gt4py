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
import types
import typing
from typing import Any, Optional, TypeVar

from gt4py.next import common, utils
from gt4py.next.ffront import field_operator_ast as foast, program_ast as past, source_utils
from gt4py.next.otf import arguments, toolchain


@dataclasses.dataclass(frozen=True)
class BaseStage: ...


def _decompose_definition_function(func: types.FunctionType) -> utils.TreeNode:
    """
    Decompose a Python function into its source code and closure variables.

    This should be enough for the use case of GT4Py DSL definitions, which are
    expected to be pure functions without complicated closures.
    """
    return utils.TreeNode(
        b"definition_function",
        (
            source_utils.make_source_definition_from_function(func),
            source_utils.get_closure_vars_from_function(func),
        ),
    )


#: Fingerprinter for the frontend stages: skips source locations on AST nodes
#: and fingerprints DSL definition functions by their source code and closure
#: variables (instead of by qualified name).
semantic_fingerprinter: utils.Fingerprinter = utils.make_fingerprinter(
    {
        **foast.semantic_fingerprint_handlers,
        types.FunctionType: _decompose_definition_function,
    }
)


#: Public alias for semantic_fingerprinter.
fingerprinter = semantic_fingerprinter


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
