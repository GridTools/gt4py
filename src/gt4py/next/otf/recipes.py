# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import dataclasses

from gt4py.next.ffront import stages as ffront_stages
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import stages, step_types, workflow


@dataclasses.dataclass(frozen=True)
class FieldopTransformWorkflow(workflow.NamedStepSequence):
    """Modular workflow for transformations with access to intermediates."""

    func_to_foast: workflow.SkippableStep[
        ffront_stages.FieldOperatorDefinition | ffront_stages.FoastOperatorDefinition,
        ffront_stages.FoastOperatorDefinition,
    ]
    foast_inject_args: workflow.Workflow[
        ffront_stages.FoastOperatorDefinition, ffront_stages.FoastClosure
    ]
    foast_to_past_closure: workflow.Workflow[ffront_stages.FoastClosure, ffront_stages.PastClosure]
    past_transform_args: workflow.Workflow[ffront_stages.PastClosure, ffront_stages.PastClosure]
    past_to_itir: workflow.Workflow[ffront_stages.PastClosure, stages.ProgramCall]
    foast_to_itir: workflow.Workflow[ffront_stages.FoastOperatorDefinition, itir.Expr]

    @property
    def step_order(self):
        return [
            "func_to_foast",
            "foast_inject_args",
            "foast_to_past_closure",
            "past_transform_args",
            "past_to_itir",
        ]


@dataclasses.dataclass(frozen=True)
class ProgramTransformWorkflow(workflow.NamedStepSequence):
    """Modular workflow for transformations with access to intermediates."""

    func_to_past: workflow.SkippableStep[
        ffront_stages.ProgramDefinition | ffront_stages.PastProgramDefinition,
        ffront_stages.PastProgramDefinition,
    ]
    past_lint: workflow.Workflow[
        ffront_stages.PastProgramDefinition, ffront_stages.PastProgramDefinition
    ]
    past_inject_args: workflow.Workflow[
        ffront_stages.PastProgramDefinition, ffront_stages.PastClosure
    ]
    past_transform_args: workflow.Workflow[ffront_stages.PastClosure, ffront_stages.PastClosure]
    past_to_itir: workflow.Workflow[ffront_stages.PastClosure, stages.ProgramCall]


@dataclasses.dataclass(frozen=True)
class OTFCompileWorkflow(workflow.NamedStepSequence):
    """The typical compiled backend steps composed into a workflow."""

    translation: step_types.TranslationStep
    bindings: workflow.Workflow[stages.ProgramSource, stages.CompilableSource]
    compilation: workflow.Workflow[stages.CompilableSource, stages.CompiledProgram]
    decoration: workflow.Workflow[stages.CompiledProgram, stages.CompiledProgram]
