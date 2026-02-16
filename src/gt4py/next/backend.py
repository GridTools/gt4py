# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import Generic

from gt4py._core import definitions as core_defs
from gt4py.next import allocators as next_allocators
from gt4py.next.ffront import (
    foast_to_gtir,
    foast_to_past,
    func_to_foast,
    func_to_past,
    past_process_args,
    past_to_itir,
    stages as ffront_stages,
)
from gt4py.next.ffront.past_passes import linters as past_linters
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import arguments, definitions, stages, toolchain, workflow


def jit_to_aot_args(
    inp: arguments.JITArgs,
) -> arguments.CompileTimeArgs:
    return arguments.CompileTimeArgs.from_concrete(*inp.args, **inp.kwargs)


def adapted_jit_to_aot_args_factory() -> workflow.Workflow[
    definitions.ConcreteProgramDef[definitions.IRDefinitionT, arguments.JITArgs],
    definitions.ConcreteProgramDef[definitions.IRDefinitionT, arguments.CompileTimeArgs],
]:
    """Wrap `jit_to_aot` into a workflow adapter to fit into backend transform workflows."""
    return toolchain.ArgsOnlyAdapter(jit_to_aot_args)


@dataclasses.dataclass(frozen=True)
class Transforms(
    workflow.MultiWorkflow[
        definitions.ConcreteProgramDef[definitions.IRDefinitionT, definitions.ArgsDefinitionT],
        definitions.CompilableProgramDef,
    ]
):
    """
    Modular workflow for transformations with access to intermediates.

    The set and order of transformation steps depends on the input type.
    Thus this workflow can be applied to DSL field operator and program definitions,
    as well as their AST representations. Even to Iterator IR programs, although in that
    case it will be a no-op.

    The input to the workflow as well as each step must be a `CompilableProgram`. The arguments
    inside the `CompilableProgram` passed to the whole workflow may be concrete (`JITArgs`)
    or compile-time (`CompileTimeArgs`). The individual steps (apart from `.aotify_args`)
    require compile-time arguments. Some of the steps can work with an empty `CompileTimeArgs` instance.
    """

    aotify_args: workflow.Workflow[
        definitions.ConcreteProgramDef[definitions.IRDefinitionT, arguments.JITArgs],
        definitions.ConcreteProgramDef[definitions.IRDefinitionT, arguments.CompileTimeArgs],
    ] = dataclasses.field(default_factory=adapted_jit_to_aot_args_factory)

    func_to_foast: workflow.Workflow[
        ffront_stages.ConcreteDSLFieldOperatorDef, ffront_stages.ConcreteFOASTOperatorDef
    ] = dataclasses.field(default_factory=func_to_foast.adapted_func_to_foast_factory)

    func_to_past: workflow.Workflow[
        ffront_stages.ConcreteDSLProgramDef, ffront_stages.ConcretePASTProgramDef
    ] = dataclasses.field(default_factory=func_to_past.adapted_func_to_past_factory)

    foast_to_itir: workflow.Workflow[
        ffront_stages.ConcreteFOASTOperatorDef, itir.FunctionDefinition
    ] = dataclasses.field(default_factory=foast_to_gtir.adapted_foast_to_gtir_factory)

    field_view_op_to_prog: workflow.Workflow[
        ffront_stages.ConcreteFOASTOperatorDef, ffront_stages.ConcretePASTProgramDef
    ] = dataclasses.field(default_factory=foast_to_past.operator_to_program_factory)

    past_lint: workflow.Workflow[
        ffront_stages.ConcretePASTProgramDef, ffront_stages.ConcretePASTProgramDef
    ] = dataclasses.field(default_factory=past_linters.adapted_linter_factory)

    field_view_prog_args_transform: workflow.Workflow[
        ffront_stages.ConcretePASTProgramDef, ffront_stages.ConcretePASTProgramDef
    ] = dataclasses.field(default_factory=past_process_args.transform_program_args_factory)

    past_to_itir: workflow.Workflow[
        ffront_stages.ConcretePASTProgramDef, definitions.CompilableProgramDef
    ] = dataclasses.field(default_factory=past_to_itir.past_to_gtir_factory)

    def step_order(self, inp: definitions.ConcreteProgramDef) -> list[str]:
        steps: list[str] = []
        if isinstance(inp.args, arguments.JITArgs):
            steps.append("aotify_args")
        match inp.data:
            case ffront_stages.DSLFieldOperatorDef():
                steps.extend(
                    [
                        "func_to_foast",
                        "field_view_op_to_prog",
                        "past_lint",
                        "field_view_prog_args_transform",
                        "past_to_itir",
                    ]
                )
            case ffront_stages.FOASTOperatorDef():
                steps.extend(
                    [
                        "field_view_op_to_prog",
                        "past_lint",
                        "field_view_prog_args_transform",
                        "past_to_itir",
                    ]
                )
            case ffront_stages.DSLProgramDef():
                steps.extend(
                    [
                        "func_to_past",
                        "past_lint",
                        "field_view_prog_args_transform",
                        "past_to_itir",
                    ]
                )
            case ffront_stages.PASTProgramDef():
                steps.extend(["past_lint", "field_view_prog_args_transform", "past_to_itir"])
            case itir.Program():
                pass
            case _:
                raise ValueError("Unexpected input.")
        return steps


DEFAULT_TRANSFORMS: Transforms = Transforms()


# TODO(tehrengruber): Rename class and `executor` & `transforms` attribute. Maybe:
#  `Backend` -> `Toolchain`
#  `transforms` -> `frontend_transforms`
#  `executor` -> `backend_transforms`
@dataclasses.dataclass(frozen=True)
class Backend(Generic[core_defs.DeviceTypeT]):
    name: str
    executor: workflow.Workflow[definitions.CompilableProgramDef, stages.ExecutableProgram]
    allocator: next_allocators.FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]
    transforms: workflow.Workflow[definitions.ConcreteProgramDef, definitions.CompilableProgramDef]

    def compile(
        self, program: definitions.IRDefinitionT, compile_time_args: arguments.CompileTimeArgs
    ) -> stages.ExecutableProgram:
        return self.executor(
            self.transforms(definitions.ConcreteProgramDef(data=program, args=compile_time_args))
        )

    @property
    def __gt_allocator__(
        self,
    ) -> next_allocators.FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]:
        return self.allocator
