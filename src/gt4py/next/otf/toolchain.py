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
from typing import Generic

from gt4py.next.otf import workflow


PrgT = typing.TypeVar("PrgT")
ArgT = typing.TypeVar("ArgT")
StartT = typing.TypeVar("StartT")
EndT = typing.TypeVar("EndT")


@dataclasses.dataclass
class CompilableProgram(Generic[PrgT, ArgT]):
    data: PrgT
    args: ArgT


@dataclasses.dataclass(frozen=True)
class DataOnlyAdapter(
    workflow.ChainableWorkflowMixin,
    workflow.ReplaceEnabledWorkflowMixin,
    workflow.Workflow[CompilableProgram[StartT, ArgT], CompilableProgram[EndT, ArgT]],
    Generic[ArgT, StartT, EndT],
):
    step: workflow.Workflow[StartT, EndT]

    def __call__(self, inp: CompilableProgram[StartT, ArgT]) -> CompilableProgram[EndT, ArgT]:
        return CompilableProgram(data=self.step(inp.data), args=inp.args)


@dataclasses.dataclass(frozen=True)
class ArgsOnlyAdapter(
    workflow.ChainableWorkflowMixin,
    workflow.ReplaceEnabledWorkflowMixin,
    workflow.Workflow[CompilableProgram[PrgT, StartT], CompilableProgram[PrgT, EndT]],
    Generic[PrgT, StartT, EndT],
):
    step: workflow.Workflow[StartT, EndT]

    def __call__(self, inp: CompilableProgram[PrgT, StartT]) -> CompilableProgram[PrgT, EndT]:
        return CompilableProgram(data=inp.data, args=self.step(inp.args))


@dataclasses.dataclass(frozen=True)
class StripArgsAdapter(
    workflow.ChainableWorkflowMixin,
    workflow.ReplaceEnabledWorkflowMixin,
    workflow.Workflow[CompilableProgram[StartT, ArgT], EndT],
    Generic[ArgT, StartT, EndT],
):
    step: workflow.Workflow[StartT, EndT]

    def __call__(self, inp: CompilableProgram[StartT, ArgT]) -> EndT:
        return self.step(inp.data)
