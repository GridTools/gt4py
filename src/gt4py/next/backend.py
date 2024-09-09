# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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


@workflow.make_step
def foast_to_foast_closure(
    inp: workflow.InputWithArgs[ffront_stages.FoastOperatorDefinition],
) -> ffront_stages.FoastClosure:
    from_fieldop = inp.kwargs.pop("from_fieldop")
    return ffront_stages.FoastClosure(
        foast_op_def=inp.data,
        args=inp.args,
        kwargs=inp.kwargs,
        closure_vars={inp.data.foast_node.id: from_fieldop},
    )


@dataclasses.dataclass(frozen=True)
class FieldopTransformWorkflow(workflow.NamedStepSequenceWithArgs):
    """Modular workflow for transformations with access to intermediates."""

    func_to_foast: workflow.SkippableStep[
        ffront_stages.FieldOperatorDefinition | ffront_stages.FoastOperatorDefinition,
        ffront_stages.FoastOperatorDefinition,
    ] = dataclasses.field(
        default_factory=lambda: func_to_foast.OptionalFuncToFoastFactory(cached=True)
    )
    foast_to_foast_closure: workflow.Workflow[
        workflow.InputWithArgs[ffront_stages.FoastOperatorDefinition], ffront_stages.FoastClosure
    ] = dataclasses.field(default=foast_to_foast_closure, metadata={"takes_args": True})
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
            "foast_to_foast_closure",
            "foast_to_past_closure",
            "past_transform_args",
            "past_to_itir",
        ]


DEFAULT_FIELDOP_TRANSFORMS = FieldopTransformWorkflow()


@dataclasses.dataclass(frozen=True)
class ProgramTransformWorkflow(workflow.NamedStepSequenceWithArgs):
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
    past_to_past_closure: workflow.Workflow[
        ffront_stages.PastProgramDefinition, ffront_stages.PastClosure
    ] = dataclasses.field(
        default=lambda inp: ffront_stages.PastClosure(
            past_node=inp.data.past_node,
            closure_vars=inp.data.closure_vars,
            grid_type=inp.data.grid_type,
            args=inp.args,
            kwargs=inp.kwargs,
        ),
        metadata={"takes_args": True},
    )
    past_transform_args: workflow.Workflow[ffront_stages.PastClosure, ffront_stages.PastClosure] = (
        dataclasses.field(
            default=past_process_args.past_process_args, metadata={"takes_args": False}
        )
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
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if isinstance(
            program, (ffront_stages.FieldOperatorDefinition, ffront_stages.FoastOperatorDefinition)
        ):
            offset_provider = kwargs.pop("offset_provider")
            from_fieldop = kwargs.pop("from_fieldop")
            program_call = self.transforms_fop(
                workflow.InputWithArgs(program, args, kwargs | {"from_fieldop": from_fieldop})
            )
            program_call = dataclasses.replace(
                program_call, kwargs=program_call.kwargs | {"offset_provider": offset_provider}
            )
        else:
            program_call = self.transforms_prog(workflow.InputWithArgs(program, args, kwargs))
        self.executor(program_call.program, *program_call.args, **program_call.kwargs)

    @property
    def __name__(self) -> str:
        return getattr(self.executor, "__name__", None) or repr(self)

    @property
    def __gt_allocator__(
        self,
    ) -> next_allocators.FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]:
        return self.allocator
