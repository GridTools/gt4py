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
import typing
from typing import Any

from gt4py.next import backend
from gt4py.next.backend import ARGS, CARG, INPUT_DATA, INPUT_PAIR
from gt4py.next.ffront import (
    foast_to_itir,
    foast_to_past,
    func_to_foast,
    func_to_past,
    past_process_args,
    past_to_itir,
    signature,
    stages as ffront_stages,
)
from gt4py.next.ffront.past_passes import linters as past_linters
from gt4py.next.ffront.stages import (
    AOT_DSL_FOP,
    AOT_DSL_PRG,
    AOT_FOP,
    AOT_PRG,
    DSL_FOP,
    DSL_PRG,
    FOP,
    PRG,
)
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import arguments, stages, workflow


DataT = typing.TypeVar("DataT")


@dataclasses.dataclass(frozen=True)
class FieldopTransformWorkflow(workflow.MultiWorkflow[INPUT_PAIR, stages.AOTProgram]):
    """Modular workflow for transformations with access to intermediates."""

    aotify_args: workflow.Workflow[
        workflow.DataArgsPair[INPUT_DATA, ARGS], workflow.DataArgsPair[INPUT_DATA, CARG]
    ] = dataclasses.field(default_factory=arguments.adapted_jit_to_aot_args_factory)

    func_to_foast: workflow.Workflow[AOT_DSL_FOP, AOT_FOP] = dataclasses.field(
        default_factory=func_to_foast.adapted_func_to_foast_factory
    )

    func_to_past: workflow.Workflow[AOT_DSL_PRG, AOT_PRG] = dataclasses.field(
        default_factory=func_to_past.adapted_func_to_past_factory
    )

    foast_to_itir: workflow.Workflow[AOT_FOP, itir.Expr] = dataclasses.field(
        default_factory=foast_to_itir.adapted_foast_to_itir_factory
    )

    field_view_op_to_prog: workflow.Workflow[AOT_FOP, AOT_PRG] = dataclasses.field(
        default_factory=foast_to_past.operator_to_program_factory
    )

    past_lint: workflow.Workflow[AOT_PRG, AOT_PRG] = dataclasses.field(
        default_factory=past_linters.adapted_linter_factory
    )

    field_view_prog_args_transform: workflow.Workflow[AOT_PRG, AOT_PRG] = dataclasses.field(
        default_factory=past_process_args.transform_program_args_factory
    )

    past_to_itir: workflow.Workflow[AOT_PRG, stages.AOTProgram] = dataclasses.field(
        default_factory=past_to_itir.past_to_itir_factory
    )

    def step_order(self, inp: INPUT_PAIR) -> list[str]:
        steps: list[str] = []
        if isinstance(inp.args, ARGS):
            steps.append("aotify_args")
        match inp.data:
            case DSL_FOP():
                steps.extend(
                    [
                        "func_to_foast",
                        "field_view_op_to_prog",
                        "past_lint",
                        "field_view_prog_args_transform",
                    ]
                )
            case FOP():
                steps.extend(
                    ["field_view_op_to_prog", "past_lint", "field_view_prog_args_transform"]
                )
            case DSL_PRG():
                steps.extend(["func_to_past", "past_lint", "field_view_prog_args_transform"])
            case PRG():
                steps.extend(["past_lint", "field_view_prog_args_transform"])
            case _:
                pass
        steps.append("past_to_itir")
        return steps


DEFAULT_TRANSFORMS: FieldopTransformWorkflow = FieldopTransformWorkflow()


@dataclasses.dataclass(frozen=True)
class ExpBackend(backend.Backend):
    def __call__(
        self,
        program: ffront_stages.ProgramDefinition | ffront_stages.FieldOperatorDefinition,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        _ = kwargs.pop("from_fieldop", None)
        # taking the offset provider out is not needed
        args, kwargs = signature.convert_to_positional(program, *args, **kwargs)
        program_info = self.transforms_fop(
            workflow.DataArgsPair(
                data=program,
                args=arguments.CompileTimeArgs.from_concrete_no_size(*args, **kwargs),
            )
        )
        # TODO(ricoh): get rid of executors altogether
        self.executor.otf_workflow(program_info)(*args, **kwargs)  # type: ignore[attr-defined]
