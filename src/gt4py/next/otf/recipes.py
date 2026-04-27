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
class OTFCompileWorkflow(
    workflow.NamedStepSequence[definitions.CompilableProgramDef, stages.CompilationArtifact]
):
    """Translation + bindings + build system; ends at a :class:`stages.CompilationArtifact`.

    Used as :attr:`gt4py.next.backend.Backend.executor`. The ``cached=True``
    backend trait wraps it in a :class:`workflow.CachedStep` keyed on
    :class:`definitions.CompilableProgramDef`.
    """

    translation: definitions.TranslationStep
    bindings: workflow.Workflow[stages.ProgramSource, stages.CompilableProject]
    compilation: workflow.Workflow[stages.CompilableProject, stages.CompilationArtifact]
