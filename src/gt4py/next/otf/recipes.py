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
from typing import Any

from gt4py.next.ffront import stages as ffront_stages
from gt4py.next.otf import stages, step_types, workflow


@dataclasses.dataclass(frozen=True)
class ProgramTransformWorkflow(workflow.NamedStepSequence):
    """Modular workflow for transformations with access to intermediates."""

    func_to_past: workflow.SkippableStep[
        ffront_stages.ProgramDefinition | ffront_stages.PastProgramDefinition,
        ffront_stages.PastProgramDefinition,
    ]
    past_transform_args: workflow.Workflow[ffront_stages.PastClosure, ffront_stages.PastClosure]
    past_to_itir: workflow.Workflow[ffront_stages.PastClosure, stages.ProgramCall]

    args: tuple[Any, ...] = dataclasses.field(default_factory=tuple)
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __call__(
        self, inp: ffront_stages.ProgramDefinition | ffront_stages.PastProgramDefinition
    ) -> stages.ProgramCall:
        past_stage = self.func_to_past(inp)
        return self.past_to_itir(
            self.past_transform_args(
                ffront_stages.PastClosure(
                    past_node=past_stage.past_node,
                    closure_vars=past_stage.closure_vars,
                    grid_type=past_stage.grid_type,
                    args=self.args,
                    kwargs=self.kwargs,
                )
            )
        )


@dataclasses.dataclass(frozen=True)
class OTFCompileWorkflow(workflow.NamedStepSequence):
    """The typical compiled backend steps composed into a workflow."""

    translation: step_types.TranslationStep
    bindings: workflow.Workflow[stages.ProgramSource, stages.CompilableSource]
    compilation: workflow.Workflow[stages.CompilableSource, stages.CompiledProgram]
    decoration: workflow.Workflow[stages.CompiledProgram, stages.CompiledProgram]
