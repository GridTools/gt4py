# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

from gt4py.next import utils
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


@dataclasses.dataclass(frozen=True)
class SingleStep(workflow.NamedStepSequence[int, StageTwo]):
    step: workflow.Workflow[int, StageTwo]


@dataclasses.dataclass(frozen=True)
class _StepWithValue:
    v: int

    def __call__(self, x: int) -> int:
        return x + self.v


def step_one(inp: StageOne) -> StageTwo:
    return StageTwo(inp, str(inp.x))


def step_two(inp: StageTwo) -> str:
    return inp.y


def test_single_step():
    step1: workflow.Workflow[StageOne, StageTwo] = workflow.make_step(step_one)
    assert step1(StageOne(3)) == step_one(StageOne(3))


def test_chain_step_sequence():
    wf: workflow.Workflow[StageOne, str] = workflow.StepSequence.start(step_one).chain(step_two)
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


def _append_one(inp: list[int]) -> list[int]:
    return [*inp, 1]


def test_cached_with_hashing():
    def hashing(inp: list[int]) -> int:
        return hash(sum(inp))

    wf = workflow.CachedStep(step=_append_one, key_function=hashing)

    assert wf([1, 2, 3]) == [1, 2, 3, 1]
    assert wf([3, 2, 1]) == [1, 2, 3, 1]


def test_replace():
    """Test replacing a named step."""
    wf = NamedStepsExample(repeat=lambda inp: [inp] * 3, strify=lambda inp: str(inp))
    wf_repl = wf.replace(repeat=lambda inp: [inp] * 4)
    assert wf_repl(4) == "[4, 4, 4, 4]"


def test_fingerprint_is_defined():
    assert callable(utils.stable_fingerprinter)
    assert isinstance(utils.stable_fingerprinter("hello"), str)


def test_fingerprint_is_stable_for_calls():
    assert utils.stable_fingerprinter("hello") == utils.stable_fingerprinter("hello")


def test_fingerprint_is_stable_for_dicts():
    d1 = {"b": 2, "a": 1}
    d2 = {"a": 1, "b": 2}
    assert utils.stable_fingerprinter(d1) == utils.stable_fingerprinter(d2)


def test_fingerprint_is_stable_for_sets():
    assert utils.stable_fingerprinter({3, 1, 2}) == utils.stable_fingerprinter({1, 2, 3})


def test_fingerprint_differs_for_different_objects():
    assert utils.stable_fingerprinter({"a": 1}) != utils.stable_fingerprinter({"a": 2})


def test_fingerprint_handles_modules():
    """Modules are unpicklable by default; the fingerprinter must serialize them by reference.

    The fingerprinter overrides built-in container types and therefore uses the pure-Python
    pickler, which (unlike the C extension) has no fast path for modules. The fingerprinter
    registers a module reducer so that data containing module references can still be hashed.
    """
    import os
    import sys

    # Does not raise and is deterministic.
    assert utils.stable_fingerprinter({"mod": os}) == utils.stable_fingerprinter({"mod": os})
    # Different modules produce different fingerprints.
    assert utils.stable_fingerprinter({"mod": os}) != utils.stable_fingerprinter({"mod": sys})


def test_cached_step_cache_key_includes_step_config():
    """Two CachedStep instances with different step configs produce different cache keys."""
    # Use dataclass steps with different configurations (v=1 vs v=2)
    cs1 = workflow.CachedStep(step=_StepWithValue(v=1), key_function=lambda x: x)
    cs2 = workflow.CachedStep(step=_StepWithValue(v=2), key_function=lambda x: x)
    assert cs1.cache_key(42) != cs2.cache_key(42)
