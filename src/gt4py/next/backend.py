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
    func_to_foast,
    func_to_past,
    past_process_args,
    past_to_itir,
    stages as ffront_stages,
)
from gt4py.next.ffront.past_passes import linters as past_linters
from gt4py.next.otf import recipes, workflow
from gt4py.next.program_processors import processor_interface as ppi


@dataclasses.dataclass(frozen=True)
class ArgsInjector(workflow.Workflow):
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


DEFAULT_FIELDOP_TRANSFORMS = recipes.FieldopTransformWorkflow(
    func_to_foast=func_to_foast.OptionalFuncToFoastFactory(cached=True),
    foast_to_itir=workflow.CachedStep(
        step=foast_to_itir.foast_to_itir, hash_function=ffront_stages.hash_foast_operator_definition
    ),
)


DEFAULT_TRANSFORMS = recipes.ProgramTransformWorkflow(
    func_to_past=func_to_past.OptionalFuncToPastFactory(cached=True),
    past_lint=past_linters.LinterFactory(),
    past_inject_args=ArgsInjector(),
    past_transform_args=past_process_args.past_process_args,
    past_to_itir=past_to_itir.PastToItirFactory(),
)


@dataclasses.dataclass(frozen=True)
class Backend(Generic[core_defs.DeviceTypeT]):
    executor: ppi.ProgramExecutor
    allocator: next_allocators.FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]
    transforms_prog: recipes.ProgramTransformWorkflow = DEFAULT_TRANSFORMS

    def __call__(
        self, program: ffront_stages.ProgramDefinition, *args: tuple[Any], **kwargs: dict[str, Any]
    ) -> None:
        transformer = self.transforms_prog.replace(
            past_inject_args=ArgsInjector(args=args, kwargs=kwargs)
        )
        program_call = transformer(program)
        self.executor(program_call.program, *program_call.args, **program_call.kwargs)

    @property
    def __name__(self) -> str:
        return getattr(self.executor, "__name__", None) or repr(self)

    @property
    def __gt_allocator__(
        self,
    ) -> next_allocators.FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]:
        return self.allocator
