from copy import deepcopy
from typing import Callable, Iterator

import pytest

from gt4py.analysis import TransformData
from gt4py.analysis.passes import (
    ComputeExtentsPass,
    InitInfoPass,
    MergeBlocksPass,
    NormalizeBlocksPass,
)


PassType = Callable[[TransformData], TransformData]


@pytest.fixture()
def init_pass() -> Iterator[PassType]:
    yield InitInfoPass().apply


@pytest.fixture()
def normalize_blocks_pass(init_pass: PassType) -> Iterator[PassType]:
    def inner(data: TransformData):
        return NormalizeBlocksPass().apply(init_pass(deepcopy(data)))

    yield inner


@pytest.fixture()
def compute_extents_pass(normalize_blocks_pass: PassType) -> Iterator[PassType]:
    def inner(data: TransformData):
        return ComputeExtentsPass().apply(normalize_blocks_pass(deepcopy(data)))

    yield inner


@pytest.fixture()
def merge_blocks_pass(compute_extents_pass: PassType) -> Iterator[PassType]:
    def inner(data: TransformData):
        return MergeBlocksPass().apply(compute_extents_pass(deepcopy(data)))

    yield inner
