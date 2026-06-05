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


S = typing.TypeVar("S")
T = typing.TypeVar("T")
DefT = typing.TypeVar("DefT")
ArgsT = typing.TypeVar("ArgsT")


@dataclasses.dataclass
class ConcreteArtifact(Generic[DefT, ArgsT]):
    data: DefT
    args: ArgsT


@dataclasses.dataclass(frozen=True)
class DataOnlyAdapter(
    workflow.ChainableWorkflowMixin,
    workflow.ReplaceEnabledWorkflowMixin,
    workflow.Workflow[ConcreteArtifact[S, ArgsT], ConcreteArtifact[T, ArgsT]],
    Generic[ArgsT, S, T],
):
    step: workflow.Workflow[S, T]

    def __call__(self, inp: ConcreteArtifact[S, ArgsT]) -> ConcreteArtifact[T, ArgsT]:
        return ConcreteArtifact(data=self.step(inp.data), args=inp.args)


@dataclasses.dataclass(frozen=True)
class ArgsOnlyAdapter(
    workflow.ChainableWorkflowMixin,
    workflow.ReplaceEnabledWorkflowMixin,
    workflow.Workflow[ConcreteArtifact[DefT, S], ConcreteArtifact[DefT, T]],
    Generic[DefT, S, T],
):
    step: workflow.Workflow[S, T]

    def __call__(self, inp: ConcreteArtifact[DefT, S]) -> ConcreteArtifact[DefT, T]:
        return ConcreteArtifact(data=inp.data, args=self.step(inp.args))


@dataclasses.dataclass(frozen=True)
class StripArgsAdapter(
    workflow.ChainableWorkflowMixin,
    workflow.ReplaceEnabledWorkflowMixin,
    workflow.Workflow[ConcreteArtifact[S, ArgsT], T],
    Generic[ArgsT, S, T],
):
    step: workflow.Workflow[S, T]

    def __call__(self, inp: ConcreteArtifact[S, ArgsT]) -> T:
        return self.step(inp.data)
