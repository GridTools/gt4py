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
from typing import Any, Optional, TypeVar, cast

from gt4py.eve import extended_typing as xtyping, utils as eve_utils
from gt4py.next import common, utils
from gt4py.next.ffront import field_operator_ast as foast, program_ast as past, source_utils
from gt4py.next.otf import arguments, toolchain


@dataclasses.dataclass(frozen=True)
class BaseStage(utils.MetadataBasedPicklingMixin): ...


_base_reducer_fn: xtyping.SingleDispatchCallable[Any, Any] = eve_utils.merge_dispatchers(
    utils.StableContainerPickler.reducer_override, foast.semantic_pickler.reducer_override
)


class _SemanticPickler(eve_utils.PurePickler):
    """
    Create a custom pickler for the BaseStage `fingerprinter` that handles
    `types.FunctionType` by using its source code and closure variables.
    This should be enough for the use case of GT4Py DSL definitions,
    which are expected to be pure functions without complicated closures.
    """

    # This is the final reducer override for the pickler, which merges the custom
    # function reducer above with the standard stable container reducer.
    _sorting_reducer_override_fn: xtyping.SingleDispatchCallable[Any, Any] = (
        eve_utils.merge_dispatchers(
            _base_reducer_fn,
            eve_utils.singledispatcher(
                default=lambda obj: NotImplemented,
                implementations={
                    types.FunctionType: eve_utils.pickle_reducer_factory(
                        get_state=lambda f: (
                            source_utils.make_source_definition_from_function(f),
                            source_utils.get_closure_vars_from_function(f),
                        )
                    ),
                },
            ),
        )
    )

    reducer_override = staticmethod(_sorting_reducer_override_fn)


SemanticPickler = cast(type[utils.SingleDispatchPickler], _SemanticPickler)


semantic_fingerprinter = functools.partial(eve_utils.content_hash, pickler_type=SemanticPickler)


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
