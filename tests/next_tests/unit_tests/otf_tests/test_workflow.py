# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

import pytest

from gt4py._core import filecache
from gt4py.next import fingerprinting
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

    wf = workflow.CachedStep(step=_append_one, input_fingerprinter=hashing)

    assert wf([1, 2, 3]) == [1, 2, 3, 1]
    assert wf([3, 2, 1]) == [1, 2, 3, 1]


def test_replace():
    """Test replacing a named step."""
    wf = NamedStepsExample(repeat=lambda inp: [inp] * 3, strify=lambda inp: str(inp))
    wf_repl = wf.replace(repeat=lambda inp: [inp] * 4)
    assert wf_repl(4) == "[4, 4, 4, 4]"


def test_fingerprint_is_defined():
    assert callable(fingerprinting.strict_fingerprinter)
    assert isinstance(fingerprinting.strict_fingerprinter("hello"), str)


def test_fingerprint_is_stable_for_calls():
    assert fingerprinting.strict_fingerprinter("hello") == fingerprinting.strict_fingerprinter(
        "hello"
    )


def test_fingerprint_is_stable_for_dicts():
    d1 = {"b": 2, "a": 1}
    d2 = {"a": 1, "b": 2}
    assert fingerprinting.strict_fingerprinter(d1) == fingerprinting.strict_fingerprinter(d2)

    # Dicts keyed by types occur in compile-time metadata (e.g. argument descriptors).
    d3 = {int: 1, str: 2}
    d4 = {str: 2, int: 1}
    assert fingerprinting.strict_fingerprinter(d3) == fingerprinting.strict_fingerprinter(d4)


def test_fingerprint_is_stable_for_sets():
    assert fingerprinting.strict_fingerprinter({3, 1, 2}) == fingerprinting.strict_fingerprinter(
        {1, 2, 3}
    )


def test_fingerprint_differs_for_different_objects():
    assert fingerprinting.strict_fingerprinter({"a": 1}) != fingerprinting.strict_fingerprinter(
        {"a": 2}
    )


def test_fingerprint_handles_modules():
    """Data containing module references must still be hashable (by qualified name)."""
    import os
    import sys

    # Does not raise and is deterministic.
    assert fingerprinting.strict_fingerprinter({"mod": os}) == fingerprinting.strict_fingerprinter(
        {"mod": os}
    )
    # Different modules produce different fingerprints.
    assert fingerprinting.strict_fingerprinter({"mod": os}) != fingerprinting.strict_fingerprinter(
        {"mod": sys}
    )


def test_cached_step_cache_key_includes_step_config():
    """Two CachedStep instances with different step configs produce different cache keys."""
    # Use dataclass steps with different configurations (v=1 vs v=2)
    cs1 = workflow.CachedStep(step=_StepWithValue(v=1), input_fingerprinter=lambda x: x)
    cs2 = workflow.CachedStep(step=_StepWithValue(v=2), input_fingerprinter=lambda x: x)
    assert cs1.cache_key(42) != cs2.cache_key(42)


def _identity(x: int) -> int:
    return x


def test_cached_step_lenient_fingerprinter_tolerates_non_importable_step():
    """The default ``lenient_fingerprinter`` hashes a non-importable step structurally."""
    # The step is a lambda (no importable qualified name). The default
    # ``step_fingerprinter`` (``lenient_fingerprinter``) lives within one
    # process, so it hashes the step structurally rather than raising. The
    # cache key is still a stable string and the step runs.
    cached = workflow.CachedStep(step=lambda inp: inp + 1, input_fingerprinter=_identity)
    assert isinstance(cached.cache_key(5), str)
    assert cached(5) == 6
    assert cached(5) == 6  # served from cache


def test_cached_step_strict_fingerprinter_rejects_non_importable_step(tmp_path):
    """``strict_fingerprinter`` requires reproducible cross-process keys, so a
    non-importable step must raise rather than risk a non-reproducible on-disk key.

    This is the fingerprinter a persistent ``FileCache`` must be paired with."""
    cached = workflow.CachedStep(
        step=lambda inp: inp + 1,
        input_fingerprinter=_identity,
        step_fingerprinter=fingerprinting.strict_fingerprinter,
        cache=filecache.FileCache(str(tmp_path)),
    )
    with pytest.raises(TypeError, match="not importable"):
        cached.cache_key(5)


def test_in_memory_factory_pairs_lenient_fingerprinter():
    """``CachedStep.in_memory`` wires the lenient fingerprinter and a dict store."""
    cached = workflow.CachedStep.in_memory(step=lambda inp: inp + 1, input_fingerprinter=_identity)
    assert cached.step_fingerprinter is fingerprinting.lenient_fingerprinter
    assert isinstance(cached.cache, dict)
    # A non-importable (lambda) step is tolerated by the lenient fingerprinter.
    assert cached(5) == 6
    assert cached(5) == 6  # served from cache


def test_persistent_factory_pairs_strict_fingerprinter(tmp_path):
    """``CachedStep.persistent`` wires the strict fingerprinter, so a non-importable
    step is rejected and persistent keys stay reproducible across processes."""
    cached = workflow.CachedStep.persistent(
        step=lambda inp: inp + 1,
        input_fingerprinter=_identity,
        cache=filecache.FileCache(str(tmp_path)),
    )
    assert cached.step_fingerprinter is fingerprinting.strict_fingerprinter
    with pytest.raises(TypeError, match="not importable"):
        cached.cache_key(5)


def test_cached_step_rejection_follows_fingerprinter_not_cache():
    """Rejection of non-importable steps is driven by ``step_fingerprinter``,
    not by the cache type.

    Even with the default in-memory ``dict`` cache, an explicit
    ``strict_fingerprinter`` rejects a non-importable step."""
    cached = workflow.CachedStep(
        step=lambda inp: inp + 1,
        input_fingerprinter=_identity,
        step_fingerprinter=fingerprinting.strict_fingerprinter,
    )
    with pytest.raises(TypeError, match="not importable"):
        cached.cache_key(5)
