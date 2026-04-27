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
    """Translation + bindings + build system; ends at a :class:`stages.BuildArtifact`.

    The artifact's concrete type is backend-specific (e.g. ``GTFNBuildArtifact``
    or ``DaCeBuildArtifact``); both share only the
    :class:`stages.BuildArtifact` Protocol — frozen, picklable, self-
    materializing. The whole post-build phase lives on the artifact itself
    (``artifact.materialize()`` returns the directly-callable program); this
    workflow's job is just to produce the artifact.

    Used directly as :attr:`gt4py.next.backend.Backend.executor`. The
    ``cached=True`` backend trait wraps this whole workflow in a
    :class:`workflow.CachedStep` — caching keys on
    :class:`definitions.CompilableProgramDef` and values on a picklable
    artifact.
    """

    translation: definitions.TranslationStep
    bindings: workflow.Workflow[stages.ProgramSource, stages.CompilableProject]
    compilation: workflow.Workflow[stages.CompilableProject, stages.BuildArtifact]
