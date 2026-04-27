# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import Any

from gt4py.next.otf import definitions, stages, workflow


@dataclasses.dataclass(frozen=True)
class OTFBuildWorkflow(
    workflow.NamedStepSequence[definitions.CompilableProgramDef, Any]
):
    """Translation + bindings + build system; ends at an on-disk artifact.

    The artifact type is backend-specific (e.g. a ``GTFNBuildArtifact`` or a
    ``DaCeBuildArtifact``); a workflow only ever pairs a backend's build with
    that same backend's finalize, so no cross-backend artifact protocol is
    needed.

    Grouped as a sub-workflow so the ``cached=True`` backend trait can wrap
    just this sub-workflow in a :class:`workflow.CachedStep` — caching keys
    on :class:`definitions.CompilableProgramDef` and values on a picklable,
    backend-specific artifact dataclass.
    """

    translation: definitions.TranslationStep
    bindings: workflow.Workflow[stages.ProgramSource, stages.CompilableProject]
    compilation: workflow.Workflow[stages.CompilableProject, Any]


@dataclasses.dataclass(frozen=True)
class OTFCompileWorkflow(
    workflow.NamedStepSequence[definitions.CompilableProgramDef, stages.ExecutableProgram]
):
    """Full OTF pipeline: two phases separated by an on-disk artifact boundary.

    1. ``build`` — produces a picklable, backend-specific build artifact.
       Heavy, idempotent, parallelizable across processes; the natural cache
       target.
    2. ``finalize`` — rehydrates the artifact into a directly-callable
       :class:`stages.ExecutableProgram`. Backend-internal; whatever
       sequence of "load the .so / wrap with gt4py calling convention /
       attach metrics" the backend needs.

    The artifact dataclass is the contract between these two phases. By
    convention, artifacts are frozen dataclasses, picklable across process
    boundaries, and self-describing (carry every property finalize needs,
    e.g. ``device_type``). Each backend defines its own; nothing about that
    contract is enforced by this module — it is per-backend convention.
    """

    build: workflow.Workflow[definitions.CompilableProgramDef, Any]
    finalize: workflow.Workflow[Any, stages.ExecutableProgram]
