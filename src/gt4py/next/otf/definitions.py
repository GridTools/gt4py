# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Protocol, TypeAlias, TypeVar

from gt4py.next.ffront import stages as ffront_stages
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import arguments, languages, stages, toolchain, workflow


CodeConfigT = TypeVar("CodeConfigT", bound=languages.SourceCodeConfig)
ToCodeConfigT = TypeVar("ToCodeConfigT", bound=languages.SourceCodeConfig)


IRDefinitionT = TypeVar(
    "IRDefinitionT",
    ffront_stages.DSLFieldOperatorDef,
    ffront_stages.DSLProgramDef,
    ffront_stages.FOASTOperatorDef,
    ffront_stages.PASTProgramDef,
    itir.Program,
)
ArgsDefinitionT = TypeVar("ArgsDefinitionT", arguments.JITArgs, arguments.CompileTimeArgs)

ConcreteProgramDef: TypeAlias = toolchain.ConcreteArtifact[IRDefinitionT, ArgsDefinitionT]
CompilableProgramDef: TypeAlias = ConcreteProgramDef[itir.Program, arguments.CompileTimeArgs]


class TranslationStep(
    workflow.ReplaceEnabledWorkflowMixin[CompilableProgramDef, stages.ProgramSource[CodeConfigT]],
    Protocol[CodeConfigT],
):
    """Translate a GT4Py program to source code (ProgramCall -> ProgramSource)."""

    ...


class BindingStep(Protocol[CodeConfigT, ToCodeConfigT]):
    """
    Generate Bindings for program source and package both together (ProgramSource -> CompilableSource).

    In the special cases where bindings are not required, such a step could also simply construct
    a ``CompilableSource`` from the ``ProgramSource`` with bindings set to ``None``.
    """

    def __call__(
        self, program_source: stages.ProgramSource[CodeConfigT]
    ) -> stages.CompilableProject[CodeConfigT, ToCodeConfigT]: ...


class CompilationStep(
    workflow.Workflow[
        stages.CompilableProject[CodeConfigT, ToCodeConfigT], stages.ExecutableProgram
    ],
    Protocol[CodeConfigT, ToCodeConfigT],
):
    """Compile program source code and bindings into a python callable (CompilableSource -> CompiledProgram)."""

    def __call__(
        self, source: stages.CompilableProject[CodeConfigT, ToCodeConfigT]
    ) -> stages.ExecutableProgram: ...
