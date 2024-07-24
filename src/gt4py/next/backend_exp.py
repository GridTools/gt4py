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
from typing import Any, Optional

from gt4py.next import backend
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
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import arguments, stages, workflow
from gt4py.next.type_system import type_specifications as ts, type_translation


DataT = typing.TypeVar("DataT")


ARGS: typing.TypeAlias = arguments.JITArgs
CARG: typing.TypeAlias = arguments.CompileArgSpec
DSL_FOP: typing.TypeAlias = ffront_stages.FieldOperatorDefinition
FOP: typing.TypeAlias = ffront_stages.FoastOperatorDefinition
DSL_PRG: typing.TypeAlias = ffront_stages.ProgramDefinition
PRG: typing.TypeAlias = ffront_stages.PastProgramDefinition
IT_PRG: typing.TypeAlias = itir.FencilDefinition


@dataclasses.dataclass(frozen=True)
class ItirShim:
    definition: ffront_stages.FoastOperatorDefinition
    foast_to_itir: workflow.Workflow[ffront_stages.FoastOperatorDefinition, itir.Expr]

    def __gt_closure_vars__(self) -> Optional[dict[str, Any]]:
        return self.definition.closure_vars

    def __gt_type__(self) -> ts.CallableType:
        return self.definition.foast_node.type

    def __gt_itir__(self) -> itir.Expr:
        return self.foast_to_itir(self.definition)


@dataclasses.dataclass(frozen=True)
class FieldviewOpToFieldviewProg:
    foast_to_itir: workflow.Workflow[ffront_stages.FoastOperatorDefinition, itir.Expr] = (
        dataclasses.field(
            default_factory=lambda: workflow.CachedStep(
                step=foast_to_itir.foast_to_itir, hash_function=ffront_stages.fingerprint_stage
            )
        )
    )

    def __call__(
        self,
        inp: workflow.DataArgsPair[ffront_stages.FoastOperatorDefinition, arguments.CompileArgSpec],
    ) -> workflow.DataArgsPair[ffront_stages.PastProgramDefinition, arguments.CompileArgSpec]:
        fieldview_program = foast_to_past.foast_to_past(
            ffront_stages.FoastWithTypes(
                foast_op_def=inp.data,
                arg_types=tuple(type_translation.from_value(arg) for arg in inp.args.args),
                kwarg_types={
                    k: type_translation.from_value(v)
                    for k, v in inp.args.kwargs.items()
                    if k not in ["from_fieldop"]
                },
                closure_vars={inp.data.foast_node.id: ItirShim(inp.data, self.foast_to_itir)},
            )
        )
        return workflow.DataArgsPair(
            data=fieldview_program,
            args=inp.args,
        )


def transform_prog_args(
    inp: workflow.DataArgsPair[ffront_stages.PastProgramDefinition, arguments.CompileArgSpec],
) -> workflow.DataArgsPair[ffront_stages.PastProgramDefinition, arguments.CompileArgSpec]:
    transformed = past_process_args.PastProcessArgs(aot_off=True)(
        ffront_stages.PastClosure(definition=inp.data, args=inp.args.args, kwargs=inp.args.kwargs)
    )
    return workflow.DataArgsPair(
        data=transformed.definition,
        args=dataclasses.replace(inp.args, args=transformed.args, kwargs=transformed.kwargs),
    )


@dataclasses.dataclass(frozen=True)
class PastToItirAdapter:
    step: workflow.Workflow[ffront_stages.AOTFieldviewProgramAst, stages.AOTProgram] = (
        dataclasses.field(default_factory=past_to_itir.PastToItir)
    )

    def __call__(
        self,
        inp: workflow.DataArgsPair[ffront_stages.PastProgramDefinition, arguments.CompileArgSpec],
    ) -> stages.AOTProgram:
        aot_fvprog = ffront_stages.AOTFieldviewProgramAst(definition=inp.data, argspec=inp.args)
        return self.step(aot_fvprog)


def jit_to_aot_args(
    inp: arguments.JITArgs,
) -> arguments.CompileArgSpec:
    return arguments.CompileArgSpec.from_concrete_no_size(*inp.args, **inp.kwargs)


AOT_FOP: typing.TypeAlias = workflow.DataArgsPair[FOP, CARG]
AOT_PRG: typing.TypeAlias = workflow.DataArgsPair[PRG, CARG]


INPUT_DATA_T: typing.TypeAlias = DSL_FOP | FOP | DSL_PRG | PRG | IT_PRG


@dataclasses.dataclass(frozen=True)
class FieldopTransformWorkflow(workflow.MultiWorkflow):
    """Modular workflow for transformations with access to intermediates."""

    aotify_args: workflow.Workflow[
        workflow.DataArgsPair[INPUT_DATA_T, ARGS], workflow.DataArgsPair[INPUT_DATA_T, CARG]
    ] = dataclasses.field(default_factory=lambda: workflow.ArgsOnlyAdapter(jit_to_aot_args))

    func_to_foast: workflow.Workflow[workflow.DataArgsPair[DSL_FOP | FOP, CARG], AOT_FOP] = (
        dataclasses.field(
            default_factory=lambda: workflow.DataOnlyAdapter(
                func_to_foast.OptionalFuncToFoastFactory(cached=True)
            )
        )
    )

    func_to_past: workflow.Workflow[workflow.DataArgsPair[DSL_PRG | PRG, CARG], AOT_PRG] = (
        dataclasses.field(
            default_factory=lambda: workflow.DataOnlyAdapter(
                func_to_past.OptionalFuncToPastFactory(cached=True)
            )
        )
    )

    foast_to_itir: workflow.Workflow[AOT_FOP, itir.Expr] = dataclasses.field(
        default_factory=lambda: workflow.StripArgsAdapter(
            workflow.CachedStep(
                step=foast_to_itir.foast_to_itir, hash_function=ffront_stages.fingerprint_stage
            )
        )
    )

    field_view_op_to_prog: workflow.Workflow[workflow.DataArgsPair[FOP, CARG], AOT_PRG] = (
        dataclasses.field(default_factory=FieldviewOpToFieldviewProg)
    )

    past_lint: workflow.Workflow[AOT_PRG, AOT_PRG] = dataclasses.field(
        default_factory=lambda: workflow.DataOnlyAdapter(past_linters.LinterFactory())
    )

    field_view_prog_args_transform: workflow.Workflow[AOT_PRG, AOT_PRG] = dataclasses.field(
        default=transform_prog_args
    )

    past_to_itir: workflow.Workflow[AOT_PRG, stages.AOTProgram] = dataclasses.field(
        default_factory=PastToItirAdapter
    )

    def step_order(
        self, inp: workflow.DataArgsPair[DSL_FOP | FOP | DSL_PRG | PRG | IT_PRG, ARGS | CARG]
    ) -> list[str]:
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
                data=program,  # type: ignore[arg-type] # TODO(ricoh): should go away when toolchain unified everywhere
                args=arguments.CompileArgSpec.from_concrete_no_size(*args, **kwargs),
            )
        )
        # TODO(ricoh): get rid of executors altogether
        self.executor.otf_workflow(program_info)(*args, **kwargs)  # type: ignore[attr-defined]
