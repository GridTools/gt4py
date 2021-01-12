from typing import Callable, Iterator

import pytest

from gt4py.analysis import TransformData
from gt4py.analysis.passes import (
    BuildIIRPass,
    ComputeExtentsPass,
    ComputeUsedSymbolsPass,
    DemoteLocalTemporariesToVariablesPass,
    InitInfoPass,
    MergeBlocksPass,
    NormalizeBlocksPass,
)


AnalysisPass = Callable[[TransformData], TransformData]


@pytest.fixture()
def init_pass() -> Iterator[AnalysisPass]:
    def inner(data: TransformData) -> TransformData:
        InitInfoPass.apply(data)
        return data

    yield inner


@pytest.fixture()
def normalize_blocks_pass(init_pass: AnalysisPass) -> Iterator[AnalysisPass]:
    def inner(data: TransformData) -> TransformData:
        NormalizeBlocksPass.apply(init_pass(data))
        return data

    yield inner


@pytest.fixture()
def compute_extents_pass(normalize_blocks_pass: AnalysisPass) -> Iterator[AnalysisPass]:
    def inner(data: TransformData) -> TransformData:
        ComputeExtentsPass.apply(normalize_blocks_pass(data))
        return data

    yield inner


@pytest.fixture()
def merge_blocks_pass(compute_extents_pass: AnalysisPass) -> Iterator[AnalysisPass]:
    def inner(data: TransformData) -> TransformData:
        MergeBlocksPass.apply(compute_extents_pass(data))
        return data

    yield inner


@pytest.fixture()
def compute_used_symbols_pass(
    merge_blocks_pass: AnalysisPass,
) -> Iterator[AnalysisPass]:
    def inner(data: TransformData) -> TransformData:
        ComputeUsedSymbolsPass.apply(merge_blocks_pass(data))
        return data

    yield inner


@pytest.fixture()
def build_iir_pass(compute_used_symbols_pass: AnalysisPass) -> Iterator[AnalysisPass]:
    def inner(data: TransformData) -> TransformData:
        BuildIIRPass.apply(compute_used_symbols_pass(data))
        return data

    yield inner


@pytest.fixture()
def demote_locals_pass(build_iir_pass: AnalysisPass) -> Iterator[AnalysisPass]:
    def inner(data: TransformData) -> TransformData:
        DemoteLocalTemporariesToVariablesPass.apply(build_iir_pass(data))
        return data

    yield inner
