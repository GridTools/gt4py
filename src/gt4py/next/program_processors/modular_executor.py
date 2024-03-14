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
from typing import Any, Optional

import gt4py.next.program_processors.processor_interface as ppi
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import stages, workflow


@dataclasses.dataclass(frozen=True)
class ModularExecutor(ppi.ProgramExecutor):
    otf_workflow: workflow.Workflow[stages.ProgramCall, stages.CompiledProgram]
    name: Optional[str] = None

    def __call__(self, program: itir.FencilDefinition, *args: Any, **kwargs: Any) -> None:
        self.otf_workflow(stages.ProgramCall(program=program, args=args, kwargs=kwargs))(
            *args, offset_provider=kwargs["offset_provider"]
        )

    @property
    def __name__(self) -> str:
        return self.name or repr(self)

    @property
    def kind(self) -> type[ppi.ProgramExecutor]:
        return ppi.ProgramExecutor
