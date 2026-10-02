# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Definition stages of the toolchain and the contracts of the steps between them.

This module hosts the DSL-aware half of the on-the-fly compilation
vocabulary: which forms a program definition can take on its way from DSL
source to a compilable IR program, and the typed contracts of the steps a
compile pipeline is composed of. The DSL-agnostic artifact models these
contracts produce live in `gt4py.next.otf.artifacts`.
"""

from __future__ import annotations

from typing import Protocol, TypeAlias, TypeVar

from gt4py.next.ffront import stages as ffront_stages
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import arguments, artifacts, workflow


CodeSpecT = TypeVar("CodeSpecT", bound=artifacts.SourceCodeSpec)
TargetCodeSpecT = TypeVar("TargetCodeSpecT", bound=artifacts.SourceCodeSpec)


IRDefinitionT = TypeVar(
    "IRDefinitionT",
    ffront_stages.DSLFieldOperatorDef,
    ffront_stages.DSLProgramDef,
    ffront_stages.FOASTOperatorDef,
    ffront_stages.PASTProgramDef,
    itir.Program,
)
ArgsDefinitionT = TypeVar("ArgsDefinitionT", arguments.JITArgs, arguments.CompileTimeArgs)

ConcreteProgramDef: TypeAlias = workflow.ConcreteArtifact[IRDefinitionT, ArgsDefinitionT]
CompilableProgramDef: TypeAlias = ConcreteProgramDef[itir.Program, arguments.CompileTimeArgs]


class TranslationStep(
    workflow.ReplaceEnabledWorkflowMixin[CompilableProgramDef, artifacts.ProgramSource[CodeSpecT]],
    Protocol[CodeSpecT],
):
    """Translate a GT4Py program to source code (ProgramCall -> ProgramSource)."""

    ...


class CompilationStep(
    workflow.Workflow[
        artifacts.ExtensionSource[CodeSpecT, TargetCodeSpecT], artifacts.CompilationArtifact
    ],
    Protocol[CodeSpecT, TargetCodeSpecT],
):
    """Run the build system and produce an ``artifacts.CompilationArtifact``.

    Each backend defines its own concrete artifact dataclass (frozen,
    picklable, with a ``load`` method); they all satisfy the
    ``artifacts.CompilationArtifact`` Protocol structurally.
    """

    def __call__(
        self, source: artifacts.ExtensionSource[CodeSpecT, TargetCodeSpecT]
    ) -> artifacts.CompilationArtifact: ...
