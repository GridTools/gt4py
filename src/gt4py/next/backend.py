# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import typing
from typing import Any, Generic

from gt4py._core import definitions as core_defs
from gt4py.next import allocators as next_allocators
from gt4py.next.ffront import (
    foast_to_gtir,
    foast_to_past,
    func_to_foast,
    func_to_past,
    past_process_args,
    past_to_itir,
    signature,
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
from gt4py.next.otf import arguments, stages, toolchain, workflow


ARGS: typing.TypeAlias = arguments.JITArgs
CARG: typing.TypeAlias = arguments.CompileTimeArgs
IT_PRG: typing.TypeAlias = itir.Program


INPUT_DATA: typing.TypeAlias = DSL_FOP | FOP | DSL_PRG | PRG | IT_PRG
INPUT_PAIR: typing.TypeAlias = toolchain.CompilableProgram[INPUT_DATA, ARGS | CARG]


@dataclasses.dataclass(frozen=True)
class Transforms(workflow.MultiWorkflow[INPUT_PAIR, stages.CompilableProgram]):
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
        toolchain.CompilableProgram[INPUT_DATA, ARGS], toolchain.CompilableProgram[INPUT_DATA, CARG]
    ] = dataclasses.field(default_factory=arguments.adapted_jit_to_aot_args_factory)

    func_to_foast: workflow.Workflow[AOT_DSL_FOP, AOT_FOP] = dataclasses.field(
        default_factory=func_to_foast.adapted_func_to_foast_factory
    )

    func_to_past: workflow.Workflow[AOT_DSL_PRG, AOT_PRG] = dataclasses.field(
        default_factory=func_to_past.adapted_func_to_past_factory
    )

    foast_to_itir: workflow.Workflow[AOT_FOP, itir.Expr] = dataclasses.field(
        default_factory=foast_to_gtir.adapted_foast_to_gtir_factory
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

    past_to_itir: workflow.Workflow[AOT_PRG, stages.CompilableProgram] = dataclasses.field(
        default_factory=past_to_itir.past_to_gtir_factory
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
                        "past_to_itir",
                    ]
                )
            case FOP():
                steps.extend(
                    [
                        "field_view_op_to_prog",
                        "past_lint",
                        "field_view_prog_args_transform",
                        "past_to_itir",
                    ]
                )
            case DSL_PRG():
                steps.extend(
                    ["func_to_past", "past_lint", "field_view_prog_args_transform", "past_to_itir"]
                )
            case PRG():
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
    executor: workflow.Workflow[stages.CompilableProgram, stages.CompiledProgram]
    allocator: next_allocators.FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]
    transforms: workflow.Workflow[INPUT_PAIR, stages.CompilableProgram]

    def __call__(
        self,
        program: INPUT_DATA,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not isinstance(program, IT_PRG):
            args, kwargs = signature.convert_to_positional(program, *args, **kwargs)
        try:
            compiled = program._compiled
        except:
            compiled = self.jit(program, *args, **kwargs)
            object.__setattr__(program, "_compiled", compiled)
        compiled(*args, **kwargs)

    def jit(self, program: INPUT_DATA, *args: Any, **kwargs: Any) -> stages.CompiledProgram:
        if not isinstance(program, IT_PRG):
            args, kwargs = signature.convert_to_positional(program, *args, **kwargs)
        aot_args = arguments.CompileTimeArgs.from_concrete_no_size(*args, **kwargs)
        return self.compile(program, aot_args)

    def compile(self, program: INPUT_DATA, compile_time_args: CARG) -> stages.CompiledProgram:
        return self.executor(
            self.transforms(toolchain.CompilableProgram(data=program, args=compile_time_args))
        )

    @property
    def __gt_allocator__(
        self,
    ) -> next_allocators.FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]:
        return self.allocator
