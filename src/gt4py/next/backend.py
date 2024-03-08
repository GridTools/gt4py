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
from typing import Generic

from gt4py._core import definitions as core_defs
from gt4py.next import allocators as next_allocators
from gt4py.next.ffront import past_process_args_wf, past_to_itir_wf
from gt4py.next.otf import stages, workflow
from gt4py.next.program_processors import processor_interface as ppi


DEFAULT_TRANSFORMS: workflow.Workflow[stages.PastClosure, stages.ProgramCall] = (
    past_process_args_wf.past_process_args.chain(past_to_itir_wf.PastToItirFactory())
)


@dataclasses.dataclass(frozen=True)
class Backend(Generic[core_defs.DeviceTypeT]):
    executor: ppi.ProgramExecutor
    allocator: next_allocators.FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]
    transformer: workflow.Workflow[stages.PastClosure, stages.ProgramCall] = DEFAULT_TRANSFORMS

    def __call__(self, program: stages.PastClosure) -> None:
        program_call = self.transformer(program)
        self.executor(program_call.program, *program_call.args, **program_call.kwargs)

    @property
    def __name__(self) -> str:
        return getattr(self.executor, "__name__", None) or repr(self)

    @property
    def __gt_allocator__(
        self,
    ) -> next_allocators.FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]:
        return self.allocator
