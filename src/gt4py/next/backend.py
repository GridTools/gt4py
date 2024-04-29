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
from typing import Any, Generic

from gt4py._core import definitions as core_defs
from gt4py.next import allocators as next_allocators
from gt4py.next.ffront import (
    foast_to_itir,
    foast_to_past,
    func_to_foast,
    func_to_past,
    past_process_args,
    past_to_itir,
    stages as ffront_stages,
)
from gt4py.next.ffront.past_passes import linters as past_linters
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import stages, workflow
from gt4py.next.program_processors import processor_interface as ppi


@dataclasses.dataclass(frozen=True)
class FopArgsInjector(workflow.Workflow):
    args: tuple[Any, ...] = dataclasses.field(default_factory=tuple)
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    from_fieldop: Any = None

    def __call__(self, inp: ffront_stages.FoastOperatorDefinition) -> ffront_stages.FoastClosure:
        return ffront_stages.FoastClosure(
            foast_op_def=inp,
            args=self.args,
            kwargs=self.kwargs,
            closure_vars={inp.foast_node.id: self.from_fieldop},
        )


@dataclasses.dataclass(frozen=True)
class FieldopTransformWorkflow(workflow.NamedStepSequence):
    """Modular workflow for transformations with access to intermediates."""

    func_to_foast: workflow.SkippableStep[
        ffront_stages.FieldOperatorDefinition | ffront_stages.FoastOperatorDefinition,
        ffront_stages.FoastOperatorDefinition,
    ] = dataclasses.field(
        default_factory=lambda: func_to_foast.OptionalFuncToFoastFactory(cached=True)
    )
    foast_inject_args: workflow.Workflow[
        ffront_stages.FoastOperatorDefinition, ffront_stages.FoastClosure
    ] = dataclasses.field(default_factory=FopArgsInjector)
    foast_to_past_closure: workflow.Workflow[
        ffront_stages.FoastClosure, ffront_stages.PastClosure
    ] = dataclasses.field(
        default_factory=lambda: foast_to_past.FoastToPastClosure(
            foast_to_past=workflow.CachedStep(
                foast_to_past.foast_to_past, hash_function=ffront_stages.fingerprint_stage
            )
        )
    )
    past_transform_args: workflow.Workflow[ffront_stages.PastClosure, ffront_stages.PastClosure] = (
        dataclasses.field(default=past_process_args.past_process_args)
    )
    past_to_itir: workflow.Workflow[ffront_stages.PastClosure, stages.ProgramCall] = (
        dataclasses.field(default_factory=past_to_itir.PastToItirFactory)
    )

    foast_to_itir: workflow.Workflow[ffront_stages.FoastOperatorDefinition, itir.Expr] = (
        dataclasses.field(
            default_factory=lambda: workflow.CachedStep(
                step=foast_to_itir.foast_to_itir, hash_function=ffront_stages.fingerprint_stage
            )
        )
    )

    @property
    def step_order(self) -> list[str]:
        return [
            "func_to_foast",
            "foast_inject_args",
            "foast_to_past_closure",
            "past_transform_args",
            "past_to_itir",
        ]


DEFAULT_FIELDOP_TRANSFORMS = FieldopTransformWorkflow()


@dataclasses.dataclass(frozen=True)
class ProgArgsInjector(workflow.Workflow):
    args: tuple[Any, ...] = dataclasses.field(default_factory=tuple)
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __call__(self, inp: ffront_stages.PastProgramDefinition) -> ffront_stages.PastClosure:
        return ffront_stages.PastClosure(
            past_node=inp.past_node,
            closure_vars=inp.closure_vars,
            grid_type=inp.grid_type,
            args=self.args,
            kwargs=self.kwargs,
        )


@dataclasses.dataclass(frozen=True)
class ProgramTransformWorkflow(workflow.NamedStepSequence):
    """Modular workflow for transformations with access to intermediates."""

    func_to_past: workflow.SkippableStep[
        ffront_stages.ProgramDefinition | ffront_stages.PastProgramDefinition,
        ffront_stages.PastProgramDefinition,
    ] = dataclasses.field(
        default_factory=lambda: func_to_past.OptionalFuncToPastFactory(cached=True)
    )
    past_lint: workflow.Workflow[
        ffront_stages.PastProgramDefinition, ffront_stages.PastProgramDefinition
    ] = dataclasses.field(default_factory=past_linters.LinterFactory)
    past_inject_args: workflow.Workflow[
        ffront_stages.PastProgramDefinition, ffront_stages.PastClosure
    ] = dataclasses.field(default_factory=ProgArgsInjector)
    past_transform_args: workflow.Workflow[ffront_stages.PastClosure, ffront_stages.PastClosure] = (
        dataclasses.field(default=past_process_args.past_process_args)
    )
    past_to_itir: workflow.Workflow[ffront_stages.PastClosure, stages.ProgramCall] = (
        dataclasses.field(default_factory=past_to_itir.PastToItirFactory)
    )


DEFAULT_PROG_TRANSFORMS = ProgramTransformWorkflow()


@dataclasses.dataclass(frozen=True)
class Backend(Generic[core_defs.DeviceTypeT]):
    executor: ppi.ProgramExecutor
    allocator: next_allocators.FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]
    transforms_fop: FieldopTransformWorkflow = DEFAULT_FIELDOP_TRANSFORMS
    transforms_prog: ProgramTransformWorkflow = DEFAULT_PROG_TRANSFORMS

    def __call__(
        self,
        program: ffront_stages.ProgramDefinition | ffront_stages.FieldOperatorDefinition,
        *args: tuple[Any],
        **kwargs: dict[str, Any],
    ) -> None:
        if isinstance(
            program, (ffront_stages.FieldOperatorDefinition, ffront_stages.FoastOperatorDefinition)
        ):
            offset_provider = kwargs.pop("offset_provider")
            from_fieldop = kwargs.pop("from_fieldop")
            transforms_fop = self.transforms_fop.replace(
                foast_inject_args=FopArgsInjector(
                    args=args, kwargs=kwargs, from_fieldop=from_fieldop
                )
            )
            program_call = transforms_fop(program)
            program_call = dataclasses.replace(
                program_call, kwargs=program_call.kwargs | {"offset_provider": offset_provider}
            )
        else:
            transforms_prog = self.transforms_prog.replace(
                past_inject_args=ProgArgsInjector(args=args, kwargs=kwargs)
            )
            program_call = transforms_prog(program)
        self.executor(program_call.program, *program_call.args, **program_call.kwargs)

    @property
    def __name__(self) -> str:
        return getattr(self.executor, "__name__", None) or repr(self)

    @property
    def __gt_allocator__(
        self,
    ) -> next_allocators.FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]:
        return self.allocator
