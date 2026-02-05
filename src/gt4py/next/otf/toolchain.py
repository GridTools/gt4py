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
class CompilableArtifact(Generic[PrgT, ArgT]):
    data: PrgT
    args: ArgT


@dataclasses.dataclass(frozen=True)
class DataOnlyAdapter(
    workflow.ChainableWorkflowMixin,
    workflow.ReplaceEnabledWorkflowMixin,
    workflow.Workflow[CompilableArtifact[StartT, ArgT], CompilableArtifact[EndT, ArgT]],
    Generic[ArgT, StartT, EndT],
):
    step: workflow.Workflow[StartT, EndT]

    def __call__(self, inp: CompilableArtifact[StartT, ArgT]) -> CompilableArtifact[EndT, ArgT]:
        return CompilableArtifact(data=self.step(inp.data), args=inp.args)


@dataclasses.dataclass(frozen=True)
class ArgsOnlyAdapter(
    workflow.ChainableWorkflowMixin,
    workflow.ReplaceEnabledWorkflowMixin,
    workflow.Workflow[CompilableArtifact[PrgT, StartT], CompilableArtifact[PrgT, EndT]],
    Generic[PrgT, StartT, EndT],
):
    step: workflow.Workflow[StartT, EndT]

    def __call__(self, inp: CompilableArtifact[PrgT, StartT]) -> CompilableArtifact[PrgT, EndT]:
        return CompilableArtifact(data=inp.data, args=self.step(inp.args))


@dataclasses.dataclass(frozen=True)
class StripArgsAdapter(
    workflow.ChainableWorkflowMixin,
    workflow.ReplaceEnabledWorkflowMixin,
    workflow.Workflow[CompilableArtifact[StartT, ArgT], EndT],
    Generic[ArgT, StartT, EndT],
):
    step: workflow.Workflow[StartT, EndT]

    def __call__(self, inp: CompilableArtifact[StartT, ArgT]) -> EndT:
        return self.step(inp.data)
