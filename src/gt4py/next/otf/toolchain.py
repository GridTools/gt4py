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


@dataclasses.dataclass(frozen=True)
class DataOnlyAdapter(
    workflow.ChainableWorkflowMixin,
    workflow.ReplaceEnabledWorkflowMixin,
    workflow.Workflow[workflow.ConcreteArtifact[S, ArgsT], workflow.ConcreteArtifact[T, ArgsT]],
    Generic[ArgsT, S, T],
):
    step: workflow.Workflow[S, T]

    def __call__(
        self, inp: workflow.ConcreteArtifact[S, ArgsT]
    ) -> workflow.ConcreteArtifact[T, ArgsT]:
        return workflow.ConcreteArtifact(data=self.step(inp.data), args=inp.args)


@dataclasses.dataclass(frozen=True)
class ArgsOnlyAdapter(
    workflow.ChainableWorkflowMixin,
    workflow.ReplaceEnabledWorkflowMixin,
    workflow.Workflow[workflow.ConcreteArtifact[DefT, S], workflow.ConcreteArtifact[DefT, T]],
    Generic[DefT, S, T],
):
    step: workflow.Workflow[S, T]

    def __call__(
        self, inp: workflow.ConcreteArtifact[DefT, S]
    ) -> workflow.ConcreteArtifact[DefT, T]:
        return workflow.ConcreteArtifact(data=inp.data, args=self.step(inp.args))


@dataclasses.dataclass(frozen=True)
class StripArgsAdapter(
    workflow.ChainableWorkflowMixin,
    workflow.ReplaceEnabledWorkflowMixin,
    workflow.Workflow[workflow.ConcreteArtifact[S, ArgsT], T],
    Generic[ArgsT, S, T],
):
    step: workflow.Workflow[S, T]

    def __call__(self, inp: workflow.ConcreteArtifact[S, ArgsT]) -> T:
        return self.step(inp.data)
