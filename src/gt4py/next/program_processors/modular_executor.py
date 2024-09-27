# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import gt4py.next.program_processors.processor_interface as ppi
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import arguments, stages, workflow


@dataclasses.dataclass(frozen=True)
class ModularExecutor(ppi.ProgramExecutor):
    otf_workflow: workflow.Workflow[stages.AOTProgram, stages.CompiledProgram]
    name: Optional[str] = None

    def __call__(self, program: itir.FencilDefinition, *args: Any, **kwargs: Any) -> None:
        self.otf_workflow(
            stages.AOTProgram(
                data=program,
                args=arguments.CompileTimeArgs.from_concrete(*args, **kwargs),
            )
        )(*args, offset_provider=kwargs["offset_provider"])

    @property
    def __name__(self) -> str:
        return self.name or repr(self)

    @property
    def kind(self) -> type[ppi.ProgramExecutor]:
        return ppi.ProgramExecutor
