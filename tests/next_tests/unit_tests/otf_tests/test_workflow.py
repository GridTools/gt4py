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

import dataclasses

from gt4py.next.otf import workflow


@dataclasses.dataclass
class StageOne:
    x: int


@dataclasses.dataclass
class StageTwo:
    pre: StageOne
    y: str


@dataclasses.dataclass(frozen=True)
class NamedStepsExample(workflow.NamedStepSequence[int, str]):
    repeat: workflow.Workflow[int, list[int]]
    strify: workflow.Workflow[list[int], str]


class SingleStep(workflow.NamedStepSequence[int, StageTwo]):
    step: workflow.Workflow[int, StageTwo]


def step_one(inp: StageOne) -> StageTwo:
    return StageTwo(inp, str(inp.x))


def step_two(inp: StageTwo) -> str:
    return inp.y


def test_single_step():
    step1: workflow.Workflow[StageOne, StageTwo] = workflow.make_step(step_one)
    assert step1(StageOne(3)) == step_one(StageOne(3))


def test_chain_step_sequence():
    wf: workflow.Workflow[StageOne, str] = workflow.StepSequenc.from_step(step_one).chain(step_two)
    inp = StageOne(5)
    assert wf(inp) == step_two(step_one(inp))


def test_named_steps():
    """Test composing named steps"""

    wf = NamedStepsExample(repeat=lambda inp: [inp] * 3, strify=lambda inp: str(inp))
    assert wf.repeat(4) == [4, 4, 4]
    assert wf.strify([1, 2, 3]) == "[1, 2, 3]"
    assert wf(4) == "[4, 4, 4]"


def test_chain_from_named():
    initial_workflow: workflow.Workflow[StageOne, StageTwo] = SingleStep(step=step_one)
    full_workflow: workflow.Workflow[StageOne, str] = initial_workflow.chain(step_two)
    assert full_workflow(StageOne(42)) == "42"


def test_cached_with_hashing():
    def hashing(inp: list[int]) -> int:
        return hash(sum(inp))

    wf = workflow.CachedStep(step=lambda inp: inp + [1], hash_function=hashing)

    assert wf([1, 2, 3]) == [1, 2, 3, 1]
    assert wf([3, 2, 1]) == [1, 2, 3, 1]


def test_replace():
    """Test replacing a named step."""
    wf = NamedStepsExample(repeat=lambda inp: [inp] * 3, strify=lambda inp: str(inp))
    wf_repl = wf.replace(repeat=lambda inp: [inp] * 4)
    assert wf_repl(4) == "[4, 4, 4, 4]"
