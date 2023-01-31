# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

import dataclasses

from gt4py.next.otf import workflow


@dataclasses.dataclass
class StageOne:
    x: int


@dataclasses.dataclass
class StageTwo:
    pre: StageOne
    y: str


def step_zero(inp: int) -> StageOne:
    return StageOne(inp)


def step_one(inp: StageOne) -> StageTwo:
    return StageTwo(inp, str(inp.x))


def step_two(inp: StageTwo) -> str:
    return inp.y


def test_single_step():
    step1: workflow.Workflow[StageOne, StageTwo] = workflow.Step(step_one)
    assert step1(StageOne(3)) == step_one(StageOne(3))


def test_chain_step_step():
    wf: workflow.Workflow[StageOne, str] = workflow.Step(step_one).chain(step_two)
    inp = StageOne(5)
    assert wf(inp) == step_two(step_one(inp))


def test_chain_combinedstep_step():
    initial_workflow: workflow.Workflow[int, StageTwo] = workflow.CombinedStep(step_zero, step_one)
    full_workflow: workflow.Workflow[int, str] = initial_workflow.chain(step_two)
    assert full_workflow(42) == "42"
