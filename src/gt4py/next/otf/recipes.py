# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses

from gt4py.next.otf import definitions, stages, workflow


@dataclasses.dataclass(frozen=True)
class OTFBuildWorkflow(
    workflow.NamedStepSequence[definitions.CompilableProgramDef, stages.BuildArtifact]
):
    """Translation + bindings + build system; ends at an on-disk :class:`stages.BuildArtifact`."""

    translation: definitions.TranslationStep
    bindings: workflow.Workflow[stages.ProgramSource, stages.CompilableProject]
    compilation: workflow.Workflow[stages.CompilableProject, stages.BuildArtifact]


@dataclasses.dataclass(frozen=True)
class OTFFinalizeWorkflow(
    workflow.NamedStepSequence[stages.BuildArtifact, stages.ExecutableProgram]
):
    """Import the built module and apply decoration to get a live callable."""

    load: workflow.Workflow[stages.BuildArtifact, stages.ExecutableProgram]
    decoration: workflow.Workflow[stages.ExecutableProgram, stages.ExecutableProgram]


@dataclasses.dataclass(frozen=True)
class OTFCompileWorkflow(
    workflow.NamedStepSequence[definitions.CompilableProgramDef, stages.ExecutableProgram]
):
    """Full OTF pipeline: the ``build`` phase ends at a picklable artifact, ``finalize`` rehydrates it."""

    build: workflow.Workflow[definitions.CompilableProgramDef, stages.BuildArtifact]
    finalize: workflow.Workflow[stages.BuildArtifact, stages.ExecutableProgram]
