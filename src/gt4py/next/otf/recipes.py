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


def materialize_artifact(artifact: stages.BuildArtifact) -> stages.ExecutableProgram:
    """Default ``finalize`` step for :class:`OTFCompileWorkflow`.

    Universal across backends: dispatches into the artifact's own
    :meth:`stages.BuildArtifact.materialize` method. The dispatch happens
    through ordinary Python method resolution on the artifact's concrete
    type — no separate registry, no backend-specific finalize plumbing.
    """
    return artifact.materialize()


@dataclasses.dataclass(frozen=True)
class OTFBuildWorkflow(
    workflow.NamedStepSequence[definitions.CompilableProgramDef, stages.BuildArtifact]
):
    """Translation + bindings + build system; ends at a :class:`stages.BuildArtifact`.

    The artifact's concrete type is backend-specific (e.g. ``GTFNBuildArtifact``
    or ``DaCeBuildArtifact``); both share only the
    :class:`stages.BuildArtifact` Protocol — frozen, picklable, self-
    materializing.

    Grouped as a sub-workflow so the ``cached=True`` backend trait can wrap
    just this sub-workflow in a :class:`workflow.CachedStep` — caching keys
    on :class:`definitions.CompilableProgramDef` and values on a picklable
    artifact.
    """

    translation: definitions.TranslationStep
    bindings: workflow.Workflow[stages.ProgramSource, stages.CompilableProject]
    compilation: workflow.Workflow[stages.CompilableProject, stages.BuildArtifact]


@dataclasses.dataclass(frozen=True)
class OTFCompileWorkflow(
    workflow.NamedStepSequence[definitions.CompilableProgramDef, stages.ExecutableProgram]
):
    """Full OTF pipeline: build an artifact, then materialize a callable.

    1. ``build`` — produces a picklable :class:`stages.BuildArtifact`. Heavy,
       idempotent, parallelizable; the natural cache target.
    2. ``finalize`` — rehydrates the artifact into a directly-callable
       :class:`stages.ExecutableProgram`. Defaults to
       :func:`materialize_artifact`, which dispatches through the artifact's
       own :meth:`stages.BuildArtifact.materialize` — backend-specific code
       lives on the artifact, not in a sibling free function.

    Backends typically only configure ``build``; ``finalize`` falls through
    to the artifact's own materialization logic. Override ``finalize`` only
    to wrap the entire post-build phase (e.g. add a tracing wrapper).
    """

    build: workflow.Workflow[definitions.CompilableProgramDef, stages.BuildArtifact]
    finalize: workflow.Workflow[stages.BuildArtifact, stages.ExecutableProgram] = (
        materialize_artifact
    )
